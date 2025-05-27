import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import argparse
import numpy as np
import json
from tqdm import tqdm
from transformers import BertModel, AutoModel
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import time
from datetime import datetime

# 使用Optuna进行贝叶斯优化
import optuna
from optuna.samplers import TPESampler
import sqlite3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你的原有类定义保持不变
class ErrorDetectionDataset(Dataset):
    def __init__(
            self,
            data_path,
            coarse_labels={
                "字符级错误": 0,
                "成分残缺型错误": 1, 
                "成分赘余型错误": 2,
                "成分搭配不当型错误": 3
            },
            fine_labels={
                "缺字漏字": 0,
                "错别字错误": 1,
                "缺少标点": 2,
                "错用标点": 3,
                "主语不明": 4,
                "谓语残缺": 5,
                "宾语残缺": 6,
                "其他成分残缺": 7,
                "主语多余": 8,
                "虚词多余": 9,
                "其他成分多余": 10,
                "语序不当": 11,
                "动宾搭配不当": 12,
                "其他搭配不当": 13
            }
    ):
        self.data_path = data_path
        self.coarse_labels = coarse_labels
        self.fine_labels = fine_labels
        self._get_data()
    
    def _get_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.data = []
        for item in data:
            sent = item['sent']
            coarse_types = item['CourseGrainedErrorType']
            fine_types = item['FineGrainedErrorType']
            
            # 构建粗粒度标签（多标签）
            coarse_label = [0] * len(self.coarse_labels)
            for c_type in coarse_types:
                if c_type in self.coarse_labels:
                    coarse_label[self.coarse_labels[c_type]] = 1
            
            # 构建细粒度标签（多标签）
            fine_label = [0] * len(self.fine_labels)
            for f_type in fine_types:
                if f_type in self.fine_labels:
                    fine_label[self.fine_labels[f_type]] = 1
            
            self.data.append((sent, coarse_label, fine_label, item.get('sent_id', -1)))
    
    def __len__(self):
        return len(self.data)
    
    def get_coarse_labels(self):
        return self.coarse_labels
    
    def get_fine_labels(self):
        return self.fine_labels
    
    def __getitem__(self, idx):
        return self.data[idx]


class ErrorDetectionDataLoader:
    def __init__(
        self,
        dataset,
        batch_size=16,
        max_length=128,
        shuffle=True,
        drop_last=True,
        device=None,
        tokenizer_name='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5'
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            self.device = device
        
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=self.shuffle,
            drop_last=self.drop_last
        )
    
    def collate_fn(self, data):
        sents = [item[0] for item in data]
        coarse_labels = [item[1] for item in data]
        fine_labels = [item[2] for item in data]
        sent_ids = [item[3] for item in data]
        
        # 编码文本
        encoded = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=sents,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_length=True
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 处理标签
        if coarse_labels[0] == -1:
            coarse_labels = None
            fine_labels = None
        else:
            coarse_labels = torch.tensor(coarse_labels, dtype=torch.float).to(self.device)
            fine_labels = torch.tensor(fine_labels, dtype=torch.float).to(self.device)
        
        return input_ids, attention_mask, coarse_labels, fine_labels, sent_ids
    
    def __iter__(self):
        for data in self.loader:
            yield data
    
    def __len__(self):
        return len(self.loader)


class ExpertSpecializedMoEClassifier(nn.Module):
    """专家专门化的MoE分类器 - 每个专家专门处理特定类型的错误"""
    def __init__(
        self, 
        pretrained_model_name, 
        num_coarse_labels=4, 
        num_fine_labels=14, 
        dropout=0.2,
        expert_dim=512
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        hidden_size = self.bert.config.hidden_size
        
        # 为每个粗粒度类别创建专门的专家
        self.coarse_gate = nn.Linear(hidden_size, num_coarse_labels)
        
        # 每个粗粒度类别对应一个专家
        self.coarse_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, hidden_size),
                nn.Dropout(dropout)
            ) for _ in range(num_coarse_labels)
        ])
        
        # 粗粒度分类头
        self.coarse_classifier = nn.Linear(hidden_size, num_coarse_labels)
        
        # 为每个粗粒度类别的细粒度分类创建专门的分类头
        self.fine_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, len(fine_indices)) 
            for fine_indices in [
                [0, 1, 2, 3],    # 字符级错误 -> 4个细粒度类别
                [4, 5, 6, 7],    # 成分残缺型错误 -> 4个细粒度类别
                [8, 9, 10],      # 成分冗余型错误 -> 3个细粒度类别
                [11, 12, 13]     # 成分搭配不当型错误 -> 3个细粒度类别
            ]
        ])
        
        # 定义粗粒度类别和对应的细粒度索引的映射
        self.coarse_to_fine_indices = {
            0: [0, 1, 2, 3],    # 字符级错误
            1: [4, 5, 6, 7],    # 成分残缺型错误
            2: [8, 9, 10],      # 成分冗余型错误
            3: [11, 12, 13]     # 成分搭配不当型错误
        }
        
        self.num_fine_labels = num_fine_labels
        
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT编码
        if token_type_ids is not None:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0, :]  # 取[CLS]的输出
        pooled_output = self.dropout(pooled_output)
        
        # 粗粒度Gate网络
        gate_logits = self.coarse_gate(pooled_output)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # 计算每个专家的输出
        expert_outputs = []
        for i, expert in enumerate(self.coarse_experts):
            expert_output = expert(pooled_output)
            expert_outputs.append(expert_output)
        
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, hidden_size)
        
        # 基于gate权重组合专家输出
        weighted_output = torch.sum(
            gate_probs.unsqueeze(-1) * expert_outputs, dim=1
        )  # (batch_size, hidden_size)
        
        # 粗粒度分类
        coarse_logits = self.coarse_classifier(weighted_output)
        coarse_probs = torch.sigmoid(coarse_logits)
        
        # 细粒度分类 - 为每个粗粒度类别计算对应的细粒度预测
        fine_logits_list = []
        batch_size = pooled_output.size(0)
        
        for i, (coarse_idx, fine_indices) in enumerate(self.coarse_to_fine_indices.items()):
            # 使用对应专家的输出进行细粒度分类
            expert_output = expert_outputs[:, i, :]  # (batch_size, hidden_size)
            fine_logits = self.fine_classifiers[i](expert_output)  # (batch_size, num_fine_for_this_coarse)
            fine_logits_list.append(fine_logits)
        
        # 组合所有细粒度预测
        fine_probs = torch.zeros(batch_size, self.num_fine_labels).to(pooled_output.device)
        for i, (coarse_idx, fine_indices) in enumerate(self.coarse_to_fine_indices.items()):
            fine_logits = fine_logits_list[i]
            fine_probs_part = torch.sigmoid(fine_logits)
            fine_probs[:, fine_indices] = fine_probs_part
        
        # 计算负载均衡损失
        load_balance_loss = self.compute_load_balance_loss(gate_probs)
        
        return coarse_probs, fine_probs, load_balance_loss
    
    def compute_load_balance_loss(self, gate_probs):
        """计算负载均衡损失"""
        expert_usage = gate_probs.mean(dim=0)
        ideal_usage = 1.0 / gate_probs.size(1)
        load_balance_loss = torch.sum((expert_usage - ideal_usage) ** 2)
        return 0.01 * load_balance_loss
    
    def apply_hierarchical_constraint(self, coarse_preds, fine_preds):
        """应用层次约束"""
        constrained_fine_preds = fine_preds.clone()
        
        for i in range(coarse_preds.size(0)):
            for coarse_idx, fine_indices in self.coarse_to_fine_indices.items():
                if coarse_preds[i, coarse_idx] == 0:
                    constrained_fine_preds[i, fine_indices] = 0
        
        return constrained_fine_preds


def calculate_metrics(labels, predictions, average='micro'):
    """计算各种评估指标"""
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    sample_acc = accuracy_score(labels, predictions)
    
    return {
        'micro_f1': micro_f1 * 100,
        'macro_f1': macro_f1 * 100,
        'accuracy': sample_acc * 100
    }


def train_single_config_for_optuna(trial, base_config, results_dir):
    """为Optuna优化训练单个配置，返回完整的6个指标"""
    
    # 从trial中获取超参数 - 移除了原MoE相关的参数，只保留expert_dim
    params = {
        'dropout': trial.suggest_float('dropout', 0.0, 0.4),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
        'lr': trial.suggest_float('lr', 1e-6, 5e-5, log=True),
        'threshold': trial.suggest_float('threshold', 0.3, 0.8),
        'seed': trial.suggest_categorical('seed', [42, 3407, 2023]),
        'expert_dim': trial.suggest_categorical('expert_dim', [256, 512, 768, 1024])
    }
    
    # 设置随机种子
    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建配置对象
    class Config:
        pass
    
    configs = Config()
    
    # 设置基础配置
    for key, value in base_config.items():
        setattr(configs, key, value)
    
    # 设置优化参数
    for key, value in params.items():
        setattr(configs, key, value)
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(configs.checkpoint_dir, f"optuna_trial_{trial.number}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    configs.checkpoint_dir = checkpoint_dir
    
    try:
        # 加载数据集
        train_dataset = ErrorDetectionDataset(configs.data_path)
        val_dataset = ErrorDetectionDataset(configs.val_data_path)    

        # 创建数据加载器
        train_dataloader = ErrorDetectionDataLoader(
            dataset=train_dataset,
            batch_size=configs.batch_size,
            max_length=configs.max_length,
            shuffle=True,
            drop_last=True,
            device=configs.device,
            tokenizer_name=configs.model_name
        )

        val_dataloader = ErrorDetectionDataLoader(
            dataset=val_dataset,
            batch_size=configs.batch_size,
            max_length=configs.max_length,
            shuffle=False,
            drop_last=False,
            device=configs.device,
            tokenizer_name=configs.model_name
        )

        # 创建模型 - 使用新的ExpertSpecializedMoEClassifier
        model = ExpertSpecializedMoEClassifier(
            pretrained_model_name=configs.model_name,
            num_coarse_labels=configs.num_coarse_labels,
            num_fine_labels=configs.num_fine_labels,
            dropout=configs.dropout,
            expert_dim=configs.expert_dim
        ).to(configs.device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=configs.lr,
            weight_decay=getattr(configs, 'weight_decay', 0.01)
        )

        # 初始化最佳验证性能和6个指标
        best_metrics = {
            'hierarchical_final_micro_f1': 0,
            'hierarchical_final_macro_f1': 0,
            'hierarchical_coarse_micro_f1': 0,
            'hierarchical_fine_micro_f1': 0,
            'hierarchical_coarse_macro_f1': 0,
            'hierarchical_fine_macro_f1': 0,
            'epoch': 0
        }
        
        best_val_f1 = 0
        patience_counter = 0
        
        # 训练循环
        for epoch in range(configs.epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            
            for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in train_dataloader:
                optimizer.zero_grad()
                
                # 前向传播
                coarse_probs, fine_probs, load_balance_loss = model(input_ids, attention_mask)
                
                # 计算分类损失
                coarse_loss = criterion(coarse_probs, coarse_labels)
                fine_loss = criterion(fine_probs, fine_labels)
                classification_loss = coarse_loss + fine_loss
                
                # 总损失 = 分类损失 + 负载均衡损失
                total_loss = classification_loss + load_balance_loss
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                train_loss += classification_loss.item()

            # 验证阶段
            model.eval()
            all_coarse_preds = []
            all_coarse_labels = []
            all_fine_labels = []
            all_constrained_fine_preds = []

            with torch.no_grad():
                for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in val_dataloader:
                    # 前向传播
                    coarse_probs, fine_probs, load_balance_loss = model(input_ids, attention_mask)
                    
                    # 应用层次约束
                    coarse_preds = (coarse_probs > configs.threshold).float()
                    fine_preds = (fine_probs > configs.threshold).float()
                    constrained_fine_preds = model.apply_hierarchical_constraint(coarse_preds, fine_preds)
                    
                    # 收集预测结果
                    all_coarse_preds.extend(coarse_preds.cpu().numpy())
                    all_coarse_labels.extend(coarse_labels.cpu().numpy())
                    all_constrained_fine_preds.extend(constrained_fine_preds.cpu().numpy())
                    all_fine_labels.extend(fine_labels.cpu().numpy())
            
            # 计算验证指标
            val_coarse_metrics_micro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='micro')
            val_coarse_metrics_macro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='macro')
            val_constrained_fine_metrics_micro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='micro')
            val_constrained_fine_metrics_macro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='macro')
            
            # 计算6个指标
            hierarchical_final_micro_f1 = (val_constrained_fine_metrics_micro['micro_f1'] + val_coarse_metrics_micro['micro_f1']) / 2
            hierarchical_final_macro_f1 = (val_constrained_fine_metrics_macro['macro_f1'] + val_coarse_metrics_macro['macro_f1']) / 2
            hierarchical_coarse_micro_f1 = val_coarse_metrics_micro['micro_f1']
            hierarchical_fine_micro_f1 = val_constrained_fine_metrics_micro['micro_f1']
            hierarchical_coarse_macro_f1 = val_coarse_metrics_macro['macro_f1']
            hierarchical_fine_macro_f1 = val_constrained_fine_metrics_macro['macro_f1']
            
            current_metrics = {
                'hierarchical_final_micro_f1': hierarchical_final_micro_f1,
                'hierarchical_final_macro_f1': hierarchical_final_macro_f1,
                'hierarchical_coarse_micro_f1': hierarchical_coarse_micro_f1,
                'hierarchical_fine_micro_f1': hierarchical_fine_micro_f1,
                'hierarchical_coarse_macro_f1': hierarchical_coarse_macro_f1,
                'hierarchical_fine_macro_f1': hierarchical_fine_macro_f1,
                'epoch': epoch + 1
            }
            
            # 检查是否为最佳性能（使用Final micro f1作为主要指标）
            if hierarchical_final_micro_f1 > best_val_f1:
                best_val_f1 = hierarchical_final_micro_f1
                best_metrics = current_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停检查
            if patience_counter >= configs.patience:
                break
            
            # 报告中间结果给Optuna（用于剪枝）
            trial.report(hierarchical_final_micro_f1, epoch)
            
            # 检查是否应该剪枝
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # 保存当前试验结果到JSON
        trial_result = {
            'trial_number': trial.number,
            'params': params,
            'metrics': best_metrics,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存每轮结果
        with open(os.path.join(results_dir, f'trial_{trial.number}_result.json'), 'w', encoding='utf-8') as f:
            json.dump(trial_result, f, ensure_ascii=False, indent=2)
        
        # 清理检查点目录
        import shutil
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        
        return best_val_f1, best_metrics
        
    except Exception as e:
        print(f"Trial {trial.number} 失败: {str(e)}")
        
        # 保存失败的试验结果
        failed_result = {
            'trial_number': trial.number,
            'params': params,
            'metrics': None,
            'status': 'failed',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(results_dir, f'trial_{trial.number}_result.json'), 'w', encoding='utf-8') as f:
            json.dump(failed_result, f, ensure_ascii=False, indent=2)
        
        # 清理检查点目录
        import shutil
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
        
        raise optuna.exceptions.TrialPruned()


def optuna_hyperparameter_optimization():
    """使用Optuna进行超参数优化"""
    
    # 固定参数 - 请根据你的实际路径修改
    base_config = {
        'model_name': '/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5',
        'num_coarse_labels': 4,
        'num_fine_labels': 14,
        'max_length': 128,
        'epochs': 40,  # 减少epoch以加快调试
        'patience': 5,
        # 请修改为你的实际数据路径
        'data_path': '/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train.json',
        'val_data_path': '/mnt/cfs/huangzhiwei/NLP-WED0910/datas/val.json',
        'checkpoint_dir': 'optuna_checkpoints_expert_specialized_0527',
        'device': device,
        'weight_decay': 0.01
    }
    
    # 创建结果保存目录
    results_dir = 'hyperparameter_results_expert_specialized_0527'
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建数据库存储优化历史
    storage_name = f"sqlite:///{results_dir}/optuna_study.db"
    
    # 创建study对象 - 贝叶斯优化配置
    study = optuna.create_study(
        direction='maximize',  # 最大化目标函数
        sampler=TPESampler(
            seed=42,
            n_startup_trials=10,  # 前10次随机试验用于初始化
            n_ei_candidates=24,   # 每次考虑24个候选点
            multivariate=True,    # 考虑参数间相关性
        ),
        pruner=optuna.pruners.MedianPruner(  # 中位数剪枝器
            n_startup_trials=5,    # 前5个epoch不剪枝
            n_warmup_steps=10,     # 预热10步
            interval_steps=1,      # 每步检查一次
        ),
        study_name='expert_specialized_error_classification',
        storage=storage_name,
        load_if_exists=True  # 如果存在则加载
    )
    
    print("开始Optuna超参数优化...")
    print("🔧 使用专家专门化MoE分类器")
    print(f"数据库: {storage_name}")
    print("优化目标: Hierarchical Final Micro F1")
    print("跟踪6个指标:")
    print("  1. hierarchical_final_micro_f1 (主要指标)")
    print("  2. hierarchical_final_macro_f1")
    print("  3. hierarchical_coarse_micro_f1")
    print("  4. hierarchical_fine_micro_f1")
    print("  5. hierarchical_coarse_macro_f1")
    print("  6. hierarchical_fine_macro_f1")
    
    start_time = time.time()
    
    # 记录所有试验结果
    all_trials_results = []
    best_overall_result = {
        'best_score': 0,
        'best_params': {},
        'best_metrics': {},
        'trial_number': -1
    }
    
    # 定义回调函数来保存中间结果
    def save_callback(study, trial):
        nonlocal best_overall_result
        
        if trial.state == optuna.trial.TrialState.COMPLETE:
            # 读取该试验的详细结果
            trial_file = os.path.join(results_dir, f'trial_{trial.number}_result.json')
            if os.path.exists(trial_file):
                with open(trial_file, 'r', encoding='utf-8') as f:
                    trial_data = json.load(f)
                    all_trials_results.append(trial_data)
                    
                    # 检查是否是新的最佳结果
                    if trial.value > best_overall_result['best_score']:
                        best_overall_result = {
                            'best_score': trial.value,
                            'best_params': trial.params.copy(),
                            'best_metrics': trial_data['metrics'].copy(),
                            'trial_number': trial.number
                        }
            
            print(f"✅ Trial {trial.number} 完成: 得分 = {trial.value:.2f}%")
            print(f"📊 当前最佳: {study.best_value:.2f}% (Trial {study.best_trial.number})")
            
            # 保存当前最佳结果
            with open(os.path.join(results_dir, 'current_best_result.json'), 'w', encoding='utf-8') as f:
                json.dump(best_overall_result, f, ensure_ascii=False, indent=2)
        
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"✂️ Trial {trial.number} 被剪枝")
    
    try:
        # 开始优化
        def objective(trial):
            score, metrics = train_single_config_for_optuna(trial, base_config, results_dir)
            return score
        
        study.optimize(
            objective,
            n_trials=200,  # 可以根据需要调整试验次数
            timeout=None,  # 不设置时间限制
            callbacks=[save_callback]
        )
        
    except KeyboardInterrupt:
        print("优化被用户中断")
    
    total_time = time.time() - start_time
    
    # 保存所有试验的汇总结果
    all_trials_summary = {
        'model_type': 'ExpertSpecializedMoEClassifier',
        'optimization_completed': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'total_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'best_trial_number': study.best_trial.number if study.best_trial else -1,
        'best_score': study.best_value if study.best_value else 0,
        'all_trials_summary': all_trials_results
    }
    
    with open(os.path.join(results_dir, 'all_trials_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(all_trials_summary, f, ensure_ascii=False, indent=2)
    
    # 保存最佳参数组合和6个指标
    if best_overall_result['trial_number'] != -1:
        final_best_result = {
            'model_type': 'ExpertSpecializedMoEClassifier',
            'best_hyperparameters': best_overall_result['best_params'],
            'best_6_metrics': best_overall_result['best_metrics'],
            'trial_number': best_overall_result['trial_number'],
            'optimization_summary': {
                'total_trials': len(study.trials),
                'total_time_hours': total_time / 3600,
                'optimization_date': datetime.now().isoformat()
            }
        }
        
        with open(os.path.join(results_dir, 'BEST_RESULT.json'), 'w', encoding='utf-8') as f:
            json.dump(final_best_result, f, ensure_ascii=False, indent=2)
    
    # 打印最终结果
    print(f"\n{'='*80}")
    print("🎉 专家专门化MoE分类器超参数优化完成!")
    print(f"⏱️  总耗时: {total_time/3600:.2f} 小时")
    print(f"🧪 总试验数: {len(study.trials)}")
    print(f"✅ 完成试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"✂️ 剪枝试验数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    
    if best_overall_result['trial_number'] != -1:
        print(f"\n🏆 最佳结果 (Trial {best_overall_result['trial_number']}):")
        print(f"📈 Final Micro F1: {best_overall_result['best_score']:.2f}%")
        
        print(f"\n🔧 最佳超参数:")
        for param, value in best_overall_result['best_params'].items():
            print(f"   {param}: {value}")
        
        print(f"\n📊 完整6个指标:")
        metrics = best_overall_result['best_metrics']
        print(f"   1. Final Micro F1: {metrics['hierarchical_final_micro_f1']:.2f}%")
        print(f"   2. Final Macro F1: {metrics['hierarchical_final_macro_f1']:.2f}%")
        print(f"   3. Coarse Micro F1: {metrics['hierarchical_coarse_micro_f1']:.2f}%")
        print(f"   4. Fine Micro F1: {metrics['hierarchical_fine_micro_f1']:.2f}%")
        print(f"   5. Coarse Macro F1: {metrics['hierarchical_coarse_macro_f1']:.2f}%")
        print(f"   6. Fine Macro F1: {metrics['hierarchical_fine_macro_f1']:.2f}%")
        print(f"   🎯 最佳epoch: {metrics['epoch']}")
    
    print(f"\n📁 结果保存在: {results_dir}/")
    print(f"   - BEST_RESULT.json: 最佳参数和6个指标")
    print(f"   - all_trials_summary.json: 所有试验汇总")
    print(f"   - trial_X_result.json: 每个试验的详细结果")
    print(f"   - current_best_result.json: 实时最佳结果")
    
    return best_overall_result, study


if __name__ == '__main__':
    print("🚀 启动专家专门化MoE层次化错误分类超参数优化")
    print("🎯 主要评判指标: Final Micro F1")
    print("📊 跟踪6个关键指标")
    print("💾 每轮结果保存到JSON文件")
    print("🔧 使用专家专门化MoE分类器 - 每个专家专门处理特定类型的错误")
    
    # 检查数据路径
    data_paths = [
        '/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train.json',
        '/mnt/cfs/huangzhiwei/NLP-WED0910/datas/val.json'
    ]
    
    print("\n🔍 检查数据文件...")
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ {path} - 存在")
        else:
            print(f"❌ {path} - 不存在")
            print(f"请修改代码中的数据路径或确保文件存在")
            exit(1)
    
    # 检查Optuna是否安装
    try:
        import optuna
        print(f"✅ Optuna版本: {optuna.__version__}")
    except ImportError:
        print("❌ 请先安装Optuna: pip install optuna")
        exit(1)
    
    # 开始优化
    best_result, study = optuna_hyperparameter_optimization()
    
    print(f"\n🎊 优化完成! 最佳配置已保存到 hyperparameter_results_expert_specialized_0527/BEST_RESULT.json")