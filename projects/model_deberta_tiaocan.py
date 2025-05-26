import torch
import torch.nn as nn
import os
import random
import argparse
import numpy as np
import json
from tqdm import tqdm
from transformers import BertModel, AutoModel
# from transformers import AdamW
from torch.optim import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from itertools import product
import time
from datetime import datetime

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
        tokenizer_name='bert-base-chinese'
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


class HierarchicalErrorClassifier(nn.Module):
    def __init__(
        self, 
        pretrained_model_name, 
        num_coarse_labels=4, 
        num_fine_labels=14, 
        dropout=0.2,
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 粗粒度分类器
        self.coarse_classifier = nn.Linear(self.bert.config.hidden_size, num_coarse_labels)
        
        # 细粒度分类器
        self.fine_classifier = nn.Linear(self.bert.config.hidden_size, num_fine_labels)
        
        # 定义粗粒度类别和对应的细粒度索引的映射
        self.coarse_to_fine_indices = {
            0: [0, 1, 2, 3],    # 字符级错误
            1: [4, 5, 6, 7],    # 成分残缺型错误
            2: [8, 9, 10],      # 成分冗余型错误
            3: [11, 12, 13]     # 成分搭配不当型错误
        }
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT编码
        if token_type_ids is not None:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:,0,:]  # 取[CLS]的输出
        pooled_output = self.dropout(pooled_output)
        
        # 粗粒度分类
        coarse_logits = self.coarse_classifier(pooled_output)
        coarse_probs = torch.sigmoid(coarse_logits)
        
        # 细粒度分类
        fine_logits = self.fine_classifier(pooled_output)
        fine_probs = torch.sigmoid(fine_logits)
        
        return coarse_probs, fine_probs
    
    def apply_hierarchical_constraint(self, coarse_preds, fine_preds):
        """
        应用层次约束：如果粗粒度类别预测为负，则该粗粒度下的所有细粒度类别均设为负
        """
        constrained_fine_preds = fine_preds.clone()
        
        # 遍历每个样本
        for i in range(coarse_preds.size(0)):
            # 对每个粗粒度类别
            for coarse_idx, fine_indices in self.coarse_to_fine_indices.items():
                # 如果粗粒度为负，则对应的细粒度全部设为负
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


def train_single_config(configs, param_combination_desc=""):
    """训练单个配置并返回12个性能指标"""
    
    # 设置随机种子
    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建检查点目录
    checkpoint_dir = os.path.join(configs.checkpoint_dir, configs.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
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

    # 创建模型
    model = HierarchicalErrorClassifier(
        pretrained_model_name=configs.model_name,
        num_coarse_labels=configs.num_coarse_labels,
        num_fine_labels=configs.num_fine_labels,
        dropout=configs.dropout
    ).to(configs.device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.lr,
        weight_decay=getattr(configs, 'weight_decay', 0.01)
    )

    # 初始化最佳验证性能
    best_metrics = {
        'epoch': 0,
        'hierarchical_final_micro_f1': 0,
        'hierarchical_final_macro_f1': 0,
        'hierarchical_coarse_micro_f1': 0,
        'hierarchical_fine_micro_f1': 0,
        'hierarchical_coarse_macro_f1': 0,
        'hierarchical_fine_macro_f1': 0,
        'normal_final_micro_f1': 0,
        'normal_final_macro_f1': 0,
        'normal_coarse_micro_f1': 0,
        'normal_fine_micro_f1': 0,
        'normal_coarse_macro_f1': 0,
        'normal_fine_macro_f1': 0
    }
    
    patience_counter = 0
    best_val_f1 = 0
    
    print(f"开始训练配置: {param_combination_desc}")
    
    # 训练循环
    for epoch in range(configs.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        with tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f'Epoch {epoch + 1}/{configs.epochs}',
            unit='batch',
            ncols=80,
            leave=False
        ) as pbar:
            for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in pbar:
                optimizer.zero_grad()
                
                # 前向传播
                coarse_probs, fine_probs = model(input_ids, attention_mask)
                
                # 计算损失
                coarse_loss = criterion(coarse_probs, coarse_labels)
                fine_loss = criterion(fine_probs, fine_labels)
                loss = coarse_loss + fine_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.3f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        all_constrained_fine_preds = []

        with torch.no_grad():
            for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in val_dataloader:
                # 前向传播
                coarse_probs, fine_probs = model(input_ids, attention_mask)
                
                # 应用层次约束
                coarse_preds = (coarse_probs > configs.threshold).float()
                fine_preds = (fine_probs > configs.threshold).float()
                constrained_fine_preds = model.apply_hierarchical_constraint(coarse_preds, fine_preds)
                
                # 计算损失
                coarse_loss = criterion(coarse_probs, coarse_labels)
                fine_loss = criterion(fine_probs, fine_labels)
                loss = coarse_loss + fine_loss
                val_loss += loss.item()
                
                # 收集预测结果
                all_coarse_preds.extend(coarse_preds.cpu().numpy())
                all_coarse_labels.extend(coarse_labels.cpu().numpy())
                all_fine_preds.extend(fine_preds.cpu().numpy())
                all_constrained_fine_preds.extend(constrained_fine_preds.cpu().numpy())
                all_fine_labels.extend(fine_labels.cpu().numpy())
        
        # 计算验证指标
        val_coarse_metrics_micro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='micro')
        val_coarse_metrics_macro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='macro')
        val_fine_metrics_micro = calculate_metrics(all_fine_labels, all_fine_preds, average='micro')
        val_fine_metrics_macro = calculate_metrics(all_fine_labels, all_fine_preds, average='macro')
        val_constrained_fine_metrics_micro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='micro')
        val_constrained_fine_metrics_macro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='macro')
        
        # 计算12个指标
        # 层次分类方法的结果
        hierarchical_final_micro_f1 = (val_constrained_fine_metrics_micro['micro_f1'] + val_coarse_metrics_micro['micro_f1']) / 2
        hierarchical_final_macro_f1 = (val_constrained_fine_metrics_macro['macro_f1'] + val_coarse_metrics_macro['macro_f1']) / 2
        hierarchical_coarse_micro_f1 = val_coarse_metrics_micro['micro_f1']
        hierarchical_fine_micro_f1 = val_constrained_fine_metrics_micro['micro_f1']
        hierarchical_coarse_macro_f1 = val_coarse_metrics_macro['macro_f1']
        hierarchical_fine_macro_f1 = val_constrained_fine_metrics_macro['macro_f1']
        
        # 正常分类方法的结果
        normal_final_micro_f1 = (val_fine_metrics_micro['micro_f1'] + val_coarse_metrics_micro['micro_f1']) / 2
        normal_final_macro_f1 = (val_fine_metrics_macro['macro_f1'] + val_coarse_metrics_macro['macro_f1']) / 2
        normal_coarse_micro_f1 = val_coarse_metrics_micro['micro_f1']
        normal_fine_micro_f1 = val_fine_metrics_micro['micro_f1']
        normal_coarse_macro_f1 = val_coarse_metrics_macro['macro_f1']
        normal_fine_macro_f1 = val_fine_metrics_macro['macro_f1']
        
        # 检查是否为最佳性能（使用层次分类的final micro f1作为主要指标）
        current_main_metric = hierarchical_final_micro_f1
        
        if current_main_metric > best_val_f1:
            best_val_f1 = current_main_metric
            best_metrics.update({
                'epoch': epoch + 1,
                'hierarchical_final_micro_f1': hierarchical_final_micro_f1,
                'hierarchical_final_macro_f1': hierarchical_final_macro_f1,
                'hierarchical_coarse_micro_f1': hierarchical_coarse_micro_f1,
                'hierarchical_fine_micro_f1': hierarchical_fine_micro_f1,
                'hierarchical_coarse_macro_f1': hierarchical_coarse_macro_f1,
                'hierarchical_fine_macro_f1': hierarchical_fine_macro_f1,
                'normal_final_micro_f1': normal_final_micro_f1,
                'normal_final_macro_f1': normal_final_macro_f1,
                'normal_coarse_micro_f1': normal_coarse_micro_f1,
                'normal_fine_micro_f1': normal_fine_micro_f1,
                'normal_coarse_macro_f1': normal_coarse_macro_f1,
                'normal_fine_macro_f1': normal_fine_macro_f1
            })
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停检查
        if patience_counter >= configs.patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        # 简单的进度输出
        if (epoch + 1) % 3 == 0:
            print(f'Epoch {epoch + 1}: Hierarchical Final F1 = {hierarchical_final_micro_f1:.2f}%')
    
    return best_metrics


def grid_search_hyperparameters():
    """网格搜索超参数调优"""
    
    # 定义超参数搜索空间
    param_grid = {
        'dropout': [0.0, 0.1, 0.2, 0.3],
        'batch_size': [8, 16, 32],
        'lr': [1e-5, 2e-5, 3e-5],
        'threshold': [0.4, 0.5, 0.6, 0.7],
        'seed': [42, 3407]
    }
    
    # 固定参数
    base_config = {
        'model_name': '/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5',  # 改为通用模型，避免路径问题
        'num_coarse_labels': 4,
        'num_fine_labels': 14,
        'max_length': 128,
        'epochs': 40,  # 减少epoch数以节省时间
        'patience': 5,
        'data_path': '../datas/train.json',  # 请确保路径正确
        'val_data_path': '../datas/val.json',  # 请确保路径正确
        'checkpoint_dir': 'grid_search_checkpoints',
        # 'device': device,
        'weight_decay': 0.01
    }
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    print(f"总共有 {len(param_combinations)} 个参数组合需要测试")
    
    # 记录所有结果
    all_results = []
    best_result = {
        'hierarchical_final_micro_f1': 0,
        'params': {},
        'metrics': {}
    }
    
    # 创建结果保存目录
    results_dir = 'grid_search_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 开始网格搜索
    start_time = time.time()
    
    for i, param_combination in enumerate(param_combinations):
        print(f"\n{'='*80}")
        print(f"测试组合 {i+1}/{len(param_combinations)}")
        
        # 创建当前配置
        class Config:
            pass
        
        configs = Config()
        
        # 设置基础配置
        for key, value in base_config.items():
            setattr(configs, key, value)

        # 修正：确保 configs.device 被设置
        setattr(configs, 'device', device) # 使用脚本顶部的全局 device
        
        # 设置当前超参数组合
        current_params = {}
        for param_name, param_value in zip(param_names, param_combination):
            setattr(configs, param_name, param_value)
            current_params[param_name] = param_value
        
        # 设置实验名称
        configs.exp_name = f"grid_search_{i+1}"
        
        # 打印当前参数
        param_str = ", ".join([f"{k}={v}" for k, v in current_params.items()])
        print(f"参数: {param_str}")
        
        try:
            # 训练当前配置
            metrics = train_single_config(configs, param_str)
            
            # 记录结果
            result = {
                'combination_id': i + 1,
                'params': current_params,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)
            
            # 检查是否为最佳结果
            if metrics['hierarchical_final_micro_f1'] > best_result['hierarchical_final_micro_f1']:
                best_result = {
                    'hierarchical_final_micro_f1': metrics['hierarchical_final_micro_f1'],
                    'params': current_params.copy(),
                    'metrics': metrics.copy(),
                    'combination_id': i + 1
                }
                print(f"新的最佳结果! Hierarchical Final Micro F1: {metrics['hierarchical_final_micro_f1']:.2f}%")
            
            print(f"当前结果: Hierarchical Final Micro F1 = {metrics['hierarchical_final_micro_f1']:.2f}%")
            print(f"当前最佳: Hierarchical Final Micro F1 = {best_result['hierarchical_final_micro_f1']:.2f}%")
            
        except Exception as e:
            print(f"训练失败: {str(e)}")
            # 记录失败的配置
            result = {
                'combination_id': i + 1,
                'params': current_params,
                'metrics': None,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            all_results.append(result)
        
        # 每完成10个组合就保存一次中间结果
        if (i + 1) % 10 == 0:
            interim_results = {
                'completed_combinations': i + 1,
                'total_combinations': len(param_combinations),
                'best_result_so_far': best_result,
                'all_results': all_results
            }
            with open(os.path.join(results_dir, f'interim_results_{i+1}.json'), 'w', encoding='utf-8') as f:
                json.dump(interim_results, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    
    # 保存最终结果
    final_results = {
        'search_completed': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'total_combinations_tested': len(param_combinations),
        'best_hyperparameters': best_result['params'],
        'best_12_metrics': best_result['metrics'],
        'best_combination_id': best_result['combination_id'],
        'param_grid': param_grid,
        'base_config': base_config,
        'all_results': all_results
    }
    
    # 保存详细结果
    with open(os.path.join(results_dir, 'final_grid_search_results.json'), 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # 保存最佳配置的简化版本
    best_config_summary = {
        'best_hyperparameters': best_result['params'],
        'best_12_metrics': {
            '层次分类方法': {
                'final_micro_f1': best_result['metrics']['hierarchical_final_micro_f1'],
                'final_macro_f1': best_result['metrics']['hierarchical_final_macro_f1'],
                'coarse_micro_f1': best_result['metrics']['hierarchical_coarse_micro_f1'],
                'fine_micro_f1': best_result['metrics']['hierarchical_fine_micro_f1'],
                'coarse_macro_f1': best_result['metrics']['hierarchical_coarse_macro_f1'],
                'fine_macro_f1': best_result['metrics']['hierarchical_fine_macro_f1']
            },
            '正常分类方法': {
                'final_micro_f1': best_result['metrics']['normal_final_micro_f1'],
                'final_macro_f1': best_result['metrics']['normal_final_macro_f1'],
                'coarse_micro_f1': best_result['metrics']['normal_coarse_micro_f1'],
                'fine_micro_f1': best_result['metrics']['normal_fine_micro_f1'],
                'coarse_macro_f1': best_result['metrics']['normal_coarse_macro_f1'],
                'fine_macro_f1': best_result['metrics']['normal_fine_macro_f1']
            }
        },
        'best_epoch': best_result['metrics']['epoch']
    }
    
    with open(os.path.join(results_dir, 'best_hyperparameters_and_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(best_config_summary, f, ensure_ascii=False, indent=2)
    
    # 打印最终结果
    print(f"\n{'='*80}")
    print("网格搜索完成!")
    print(f"总耗时: {total_time/3600:.2f} 小时")
    print(f"测试了 {len(param_combinations)} 个参数组合")
    print(f"\n最佳超参数组合:")
    for param, value in best_result['params'].items():
        print(f"  {param}: {value}")
    
    print(f"\n最佳性能指标 (12个指标):")
    print("层次分类方法:")
    print(f"  Final Micro F1: {best_result['metrics']['hierarchical_final_micro_f1']:.2f}%")
    print(f"  Final Macro F1: {best_result['metrics']['hierarchical_final_macro_f1']:.2f}%")
    print(f"  Coarse Micro F1: {best_result['metrics']['hierarchical_coarse_micro_f1']:.2f}%")
    print(f"  Fine Micro F1: {best_result['metrics']['hierarchical_fine_micro_f1']:.2f}%")
    print(f"  Coarse Macro F1: {best_result['metrics']['hierarchical_coarse_macro_f1']:.2f}%")
    print(f"  Fine Macro F1: {best_result['metrics']['hierarchical_fine_macro_f1']:.2f}%")
    
    print("正常分类方法:")
    print(f"  Final Micro F1: {best_result['metrics']['normal_final_micro_f1']:.2f}%")
    print(f"  Final Macro F1: {best_result['metrics']['normal_final_macro_f1']:.2f}%")
    print(f"  Coarse Micro F1: {best_result['metrics']['normal_coarse_micro_f1']:.2f}%")
    print(f"  Fine Micro F1: {best_result['metrics']['normal_fine_micro_f1']:.2f}%")
    print(f"  Coarse Macro F1: {best_result['metrics']['normal_coarse_macro_f1']:.2f}%")
    print(f"  Fine Macro F1: {best_result['metrics']['normal_fine_macro_f1']:.2f}%")
    
    print(f"\n在第 {best_result['metrics']['epoch']} 个epoch达到最佳性能")
    
    return best_result, all_results


def analyze_grid_search_results(results_file='grid_search_results/final_grid_search_results.json'):
    """分析网格搜索结果"""
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    all_results = results['all_results']
    successful_results = [r for r in all_results if r['metrics'] is not None]
    
    if not successful_results:
        print("没有成功的实验结果")
        return
    
    print(f"\n网格搜索结果分析 (基于 {len(successful_results)} 个成功的实验)")
    print("="*80)
    
    # 按性能排序
    successful_results.sort(key=lambda x: x['metrics']['hierarchical_final_micro_f1'], reverse=True)
    
    # 显示top 5结果
    print("Top 5 配置:")
    for i, result in enumerate(successful_results[:5]):
        print(f"\n{i+1}. Hierarchical Final Micro F1: {result['metrics']['hierarchical_final_micro_f1']:.2f}%")
        params_str = ", ".join([f"{k}={v}" for k, v in result['params'].items()])
        print(f"   参数: {params_str}")
    
    # 分析每个超参数的影响
    print(f"\n超参数影响分析:")
    param_names = list(successful_results[0]['params'].keys())
    
    for param_name in param_names:
        param_performance = {}
        for result in successful_results:
            param_value = result['params'][param_name]
            if param_value not in param_performance:
                param_performance[param_value] = []
            param_performance[param_value].append(result['metrics']['hierarchical_final_micro_f1'])
        
        print(f"\n{param_name}:")
        for value, performances in param_performance.items():
            avg_perf = np.mean(performances)
            std_perf = np.std(performances)
            print(f"  {value}: {avg_perf:.2f}% ± {std_perf:.2f}% (n={len(performances)})")


if __name__ == '__main__':
    print("开始网格搜索超参数调优...")
    print("调优的超参数: dropout, batch_size, lr, threshold, seed")
    print("将记录12个性能指标并保存到JSON文件中")
    
    # 开始网格搜索
    best_result, all_results = grid_search_hyperparameters()
    
    # 分析结果
    print("\n正在分析结果...")
    analyze_grid_search_results()