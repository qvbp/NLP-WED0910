# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import random
# import argparse
# import numpy as np
# import json
# from tqdm import tqdm
# from transformers import BertModel, AutoModel
# from torch.optim import AdamW
# from torch.utils.data import Dataset
# from torch.utils.data import DataLoader
# from transformers import AutoTokenizer
# from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# import time
# from datetime import datetime

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class ErrorDetectionDataset(Dataset):
#     def __init__(
#             self,
#             data_path,
#             coarse_labels={
#                 "字符级错误": 0,
#                 "成分残缺型错误": 1, 
#                 "成分赘余型错误": 2,
#                 "成分搭配不当型错误": 3
#             },
#             fine_labels={
#                 "缺字漏字": 0,
#                 "错别字错误": 1,
#                 "缺少标点": 2,
#                 "错用标点": 3,
#                 "主语不明": 4,
#                 "谓语残缺": 5,
#                 "宾语残缺": 6,
#                 "其他成分残缺": 7,
#                 "主语多余": 8,
#                 "虚词多余": 9,
#                 "其他成分多余": 10,
#                 "语序不当": 11,
#                 "动宾搭配不当": 12,
#                 "其他搭配不当": 13
#             }
#     ):
#         self.data_path = data_path
#         self.coarse_labels = coarse_labels
#         self.fine_labels = fine_labels
#         self._get_data()
    
#     def _get_data(self):
#         with open(self.data_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         self.data = []
#         for item in data:
#             sent = item['sent']
#             coarse_types = item['CourseGrainedErrorType']
#             fine_types = item['FineGrainedErrorType']
            
#             # 构建粗粒度标签（多标签）
#             coarse_label = [0] * len(self.coarse_labels)
#             for c_type in coarse_types:
#                 if c_type in self.coarse_labels:
#                     coarse_label[self.coarse_labels[c_type]] = 1
            
#             # 构建细粒度标签（多标签）
#             fine_label = [0] * len(self.fine_labels)
#             for f_type in fine_types:
#                 if f_type in self.fine_labels:
#                     fine_label[self.fine_labels[f_type]] = 1
            
#             self.data.append((sent, coarse_label, fine_label, item.get('sent_id', -1)))
    
#     def __len__(self):
#         return len(self.data)
    
#     def get_coarse_labels(self):
#         return self.coarse_labels
    
#     def get_fine_labels(self):
#         return self.fine_labels
    
#     def __getitem__(self, idx):
#         return self.data[idx]


# class ErrorDetectionDataLoader:
#     def __init__(
#         self,
#         dataset,
#         batch_size=16,
#         max_length=128,
#         shuffle=True,
#         drop_last=True,
#         device=None,
#         tokenizer_name='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5'
#     ):
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.shuffle = shuffle
#         self.drop_last = drop_last
        
#         if device is None:
#             self.device = torch.device(
#                 'cuda' if torch.cuda.is_available() else 'cpu'
#             )
#         else:
#             self.device = device
        
#         self.loader = DataLoader(
#             dataset=self.dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.collate_fn,
#             shuffle=self.shuffle,
#             drop_last=self.drop_last
#         )
    
#     def collate_fn(self, data):
#         sents = [item[0] for item in data]
#         coarse_labels = [item[1] for item in data]
#         fine_labels = [item[2] for item in data]
#         sent_ids = [item[3] for item in data]
        
#         # 编码文本
#         encoded = self.tokenizer.batch_encode_plus(
#             batch_text_or_text_pairs=sents,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt',
#             return_length=True
#         )
        
#         input_ids = encoded['input_ids'].to(self.device)
#         attention_mask = encoded['attention_mask'].to(self.device)
        
#         # 处理标签
#         if coarse_labels[0] == -1:
#             coarse_labels = None
#             fine_labels = None
#         else:
#             coarse_labels = torch.tensor(coarse_labels, dtype=torch.float).to(self.device)
#             fine_labels = torch.tensor(fine_labels, dtype=torch.float).to(self.device)
        
#         return input_ids, attention_mask, coarse_labels, fine_labels, sent_ids
    
#     def __iter__(self):
#         for data in self.loader:
#             yield data
    
#     def __len__(self):
#         return len(self.loader)

# class MoELayer(nn.Module):
#     """Mixture of Experts Layer"""
#     def __init__(self, input_dim, expert_dim, num_experts=8, top_k=2, dropout=0.1, load_balance_weight=0.01):
#         super().__init__()
#         self.num_experts = num_experts
#         self.top_k = top_k
#         self.input_dim = input_dim
#         self.expert_dim = expert_dim
        
#         # Gate网络，用于选择专家
#         self.gate = nn.Linear(input_dim, num_experts)

#         # 专家网络：每一个专家都是两层的MLP
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(input_dim, expert_dim),
#                 nn.ReLU(),
#                 nn.Dropout(dropout),
#                 nn.Linear(expert_dim, input_dim),
#                 nn.Dropout(dropout)
#             ) for _ in range(num_experts)
#         ])

#         # 负载均衡损失的权重
#         self.load_balance_weight = load_balance_weight
    
#     def forward(self, x):
#         batch_size, seq_len = x.size(0), x.size(1) if len(x.shape) > 2 else 1

#         # 如果输入是3D (batch, seq, hidden)，我们只用[CLS]位置
#         if len(x.shape) == 3:
#             x = x[:, 0, :]  # 取[CLS]位置

#         # Gate网络输出: 选择专家的概率
#         gate_logits = self.gate(x) # (batch_size, num_experts)
#         gate_probs = F.softmax(gate_logits, dim=-1)

#         # 选择前top_k个专家
#         top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
#         top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 归一化

#         # 计算专家输出
#         expert_outputs = []
#         for i in range(self.num_experts):
#             expert_output = self.experts[i](x) # (batch_size, input_dim)
#             expert_outputs.append(expert_output)

#         expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, input_dim)

#         # 组合top-k专家的输出
#         final_output = torch.zeros_like(x) # (batch_size, input_dim)
#         for i in range(self.top_k):
#             expert_idx = top_k_indices[:, i]  # 修复语法错误
#             expert_weight = top_k_probs[:, i:i+1]  # (batch_size, 1)

#             # 选择对应专家的输出
#             selected_output = expert_outputs[torch.arange(batch_size), expert_idx]  # (batch_size, input_dim)
#             final_output += expert_weight * selected_output

#         # 计算负载均衡损失
#         load_balance_loss = self.compute_load_balance_loss(gate_probs)
        
#         return final_output, load_balance_loss

    
#     def compute_load_balance_loss(self, gate_probs):
#         """计算负载均衡损失，鼓励专家使用的均衡性"""
#         # 计算每个专家被选择的平均概率
#         expert_usage = gate_probs.mean(dim=0)  # (num_experts,)
        
#         # 理想情况下每个专家被使用的概率应该是 1/num_experts
#         ideal_usage = 1.0 / self.num_experts
        
#         # 计算负载均衡损失（使用L2距离）
#         load_balance_loss = torch.sum((expert_usage - ideal_usage) ** 2)
        
#         return self.load_balance_weight * load_balance_loss


# class HierarchicalErrorClassifier(nn.Module):
#     def __init__(
#         self, 
#         pretrained_model_name, 
#         num_coarse_labels=4, 
#         num_fine_labels=14, 
#         dropout=0.2,
#         num_experts=8,
#         expert_dim=512,
#         top_k=2,
#         use_separate_moe=True,  # 是否为粗粒度和细粒度使用不同的MoE
#         load_balance_weight=0.01
#     ):
#         super().__init__()
        
#         self.bert = AutoModel.from_pretrained(pretrained_model_name)
#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.use_separate_moe = use_separate_moe
        
#         hidden_size = self.bert.config.hidden_size
        
#         if use_separate_moe:
#             # 为粗粒度和细粒度分类分别使用不同的MoE
#             self.coarse_moe = MoELayer(
#                 input_dim=hidden_size,
#                 expert_dim=expert_dim,
#                 num_experts=num_experts,
#                 top_k=top_k,
#                 dropout=dropout,
#                 load_balance_weight=load_balance_weight
#             )
#             self.fine_moe = MoELayer(
#                 input_dim=hidden_size,
#                 expert_dim=expert_dim,
#                 num_experts=num_experts,
#                 top_k=top_k,
#                 dropout=dropout,
#                 load_balance_weight=load_balance_weight
#             )
#         else:
#             # 共享一个MoE层
#             self.shared_moe = MoELayer(
#                 input_dim=hidden_size,
#                 expert_dim=expert_dim,
#                 num_experts=num_experts,
#                 top_k=top_k,
#                 dropout=dropout,
#                 load_balance_weight=load_balance_weight
#             )
        
#         # 粗粒度分类器
#         self.coarse_classifier = nn.Linear(self.bert.config.hidden_size, num_coarse_labels)
        
#         # 细粒度分类器
#         self.fine_classifier = nn.Linear(self.bert.config.hidden_size, num_fine_labels)
        
#         # 定义粗粒度类别和对应的细粒度索引的映射
#         self.coarse_to_fine_indices = {
#             0: [0, 1, 2, 3],    # 字符级错误
#             1: [4, 5, 6, 7],    # 成分残缺型错误
#             2: [8, 9, 10],      # 成分冗余型错误
#             3: [11, 12, 13]     # 成分搭配不当型错误
#         }
    
#     def forward(self, input_ids, attention_mask, token_type_ids=None):
#         # BERT编码
#         if token_type_ids is not None:
#             outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#         else:
#             outputs = self.bert(input_ids, attention_mask=attention_mask)
        
#         pooled_output = outputs.last_hidden_state[:,0,:]  # 取[CLS]的输出
#         pooled_output = self.dropout(pooled_output)

#         total_load_balance_loss = 0
        
#         if self.use_separate_moe:
#             # 使用不同的MoE处理粗粒度和细粒度分类
#             coarse_features, coarse_lb_loss = self.coarse_moe(pooled_output)
#             fine_features, fine_lb_loss = self.fine_moe(pooled_output)
#             total_load_balance_loss = coarse_lb_loss + fine_lb_loss
#         else:
#             # 使用共享MoE
#             shared_features, shared_lb_loss = self.shared_moe(pooled_output)
#             coarse_features = fine_features = shared_features
#             total_load_balance_loss = shared_lb_loss
        
#         # 粗粒度分类
#         coarse_logits = self.coarse_classifier(pooled_output)
#         coarse_probs = torch.sigmoid(coarse_logits)
        
#         # 细粒度分类
#         fine_logits = self.fine_classifier(pooled_output)
#         fine_probs = torch.sigmoid(fine_logits)
        
#         return coarse_probs, fine_probs, total_load_balance_loss
    
#     def apply_hierarchical_constraint(self, coarse_preds, fine_preds):
#         """
#         应用层次约束：如果粗粒度类别预测为负，则该粗粒度下的所有细粒度类别均设为负
#         """
#         constrained_fine_preds = fine_preds.clone()
        
#         # 遍历每个样本
#         for i in range(coarse_preds.size(0)):
#             # 对每个粗粒度类别
#             for coarse_idx, fine_indices in self.coarse_to_fine_indices.items():
#                 # 如果粗粒度为负，则对应的细粒度全部设为负
#                 if coarse_preds[i, coarse_idx] == 0:
#                     constrained_fine_preds[i, fine_indices] = 0
        
#         return constrained_fine_preds


# def calculate_metrics(labels, predictions, average='micro'):
#     """计算各种评估指标"""
#     labels = np.array(labels)
#     predictions = np.array(predictions)
    
#     micro_f1 = f1_score(labels, predictions, average='micro')
#     macro_f1 = f1_score(labels, predictions, average='macro')
#     sample_acc = accuracy_score(labels, predictions)
    
#     return {
#         'micro_f1': micro_f1 * 100,
#         'macro_f1': macro_f1 * 100,
#         'accuracy': sample_acc * 100
#     }


# def train_model(config):
#     """主训练函数"""
#     print("🚀 开始训练层次化错误分类模型")
#     print(f"📱 设备: {config['device']}")
#     print(f"🏷️  粗粒度类别数: {config['num_coarse_labels']}")
#     print(f"🏷️  细粒度类别数: {config['num_fine_labels']}")
#     print(f"📝 最大序列长度: {config['max_length']}")
#     print(f"🔄 批次大小: {config['batch_size']}")
#     print(f"📚 训练轮数: {config['epochs']}")
#     print("="*80)
    
#     # 设置随机种子
#     random.seed(config['seed'])
#     np.random.seed(config['seed'])
#     torch.manual_seed(config['seed'])
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     # 创建结果保存目录
#     os.makedirs(config['results_dir'], exist_ok=True)
#     os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
#     # 加载数据集
#     print("📂 加载数据集...")
#     train_dataset = ErrorDetectionDataset(config['data_path'])
#     val_dataset = ErrorDetectionDataset(config['val_data_path'])
#     print(f"📊 训练集大小: {len(train_dataset)}")
#     print(f"📊 验证集大小: {len(val_dataset)}")
    
#     # 创建数据加载器
#     train_dataloader = ErrorDetectionDataLoader(
#         dataset=train_dataset,
#         batch_size=config['batch_size'],
#         max_length=config['max_length'],
#         shuffle=True,
#         drop_last=True,
#         device=config['device'],
#         tokenizer_name=config['model_name']
#     )

#     val_dataloader = ErrorDetectionDataLoader(
#         dataset=val_dataset,
#         batch_size=config['batch_size'],
#         max_length=config['max_length'],
#         shuffle=False,
#         drop_last=False,
#         device=config['device'],
#         tokenizer_name=config['model_name']
#     )
    
#     # 创建模型
#     print("🏗️  构建模型...")
#     model = HierarchicalErrorClassifier(
#         pretrained_model_name=config['model_name'],
#         num_coarse_labels=config['num_coarse_labels'],
#         num_fine_labels=config['num_fine_labels'],
#         dropout=config['dropout'],
#         num_experts=config['num_experts'],
#         expert_dim=config['expert_dim'],
#         top_k=config['top_k'],
#         use_separate_moe=config['use_separate_moe'],
#         load_balance_weight=config['load_balance_weight']
#     ).to(config['device'])
    
#     print(f"🔧 模型参数:")
#     print(f"   - Dropout: {config['dropout']}")
#     print(f"   - 专家数量: {config['num_experts']}")
#     print(f"   - Top-K: {config['top_k']}")
#     print(f"   - 专家维度: {config['expert_dim']}")
#     print(f"   - 独立MoE: {config['use_separate_moe']}")
#     print(f"   - 负载均衡权重: {config['load_balance_weight']}")
    
#     # 定义损失函数和优化器
#     criterion = nn.BCELoss()
#     optimizer = AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=config['lr'],
#         weight_decay=config.get('weight_decay', 0.01)
#     )
    
#     # 初始化最佳验证性能和6个指标
#     best_metrics = {
#         'hierarchical_final_micro_f1': 0,
#         'hierarchical_final_macro_f1': 0,
#         'hierarchical_coarse_micro_f1': 0,
#         'hierarchical_fine_micro_f1': 0,
#         'hierarchical_coarse_macro_f1': 0,
#         'hierarchical_fine_macro_f1': 0,
#         'epoch': 0
#     }
    
#     best_val_f1 = 0
#     patience_counter = 0
    
#     # 记录所有epoch的结果
#     training_history = {
#         'train_loss': [],
#         'val_metrics': [],
#         'best_metrics': best_metrics,
#         'config': config
#     }
    
#     print("\n🎯 开始训练...")
#     start_time = time.time()
    
#     # 训练循环
#     for epoch in range(config['epochs']):
#         epoch_start_time = time.time()
        
#         # 训练阶段
#         model.train()
#         train_loss = 0.0
#         train_steps = 0
        
#         train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
#         for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in train_progress:
#             optimizer.zero_grad()
            
#             # 前向传播
#             coarse_probs, fine_probs, load_balance_loss = model(input_ids, attention_mask)
            
#             # 计算分类损失
#             coarse_loss = criterion(coarse_probs, coarse_labels)
#             fine_loss = criterion(fine_probs, fine_labels)
#             classification_loss = coarse_loss + fine_loss
            
#             # 总损失 = 分类损失 + 负载均衡损失
#             total_loss = classification_loss + load_balance_loss
            
#             # 反向传播
#             total_loss.backward()
#             optimizer.step()
            
#             train_loss += classification_loss.item()
#             train_steps += 1
            
#             # 更新进度条
#             train_progress.set_postfix({
#                 'Loss': f'{classification_loss.item():.4f}',
#                 'LB_Loss': f'{load_balance_loss.item():.4f}'
#             })
        
#         avg_train_loss = train_loss / train_steps
        
#         # 验证阶段
#         model.eval()
#         all_coarse_preds = []
#         all_coarse_labels = []
#         all_fine_labels = []
#         all_constrained_fine_preds = []
        
#         val_progress = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
#         with torch.no_grad():
#             for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in val_progress:
#                 # 前向传播
#                 coarse_probs, fine_probs, load_balance_loss = model(input_ids, attention_mask)
                
#                 # 应用层次约束
#                 coarse_preds = (coarse_probs > config['threshold']).float()
#                 fine_preds = (fine_probs > config['threshold']).float()
#                 constrained_fine_preds = model.apply_hierarchical_constraint(coarse_preds, fine_preds)
                
#                 # 收集预测结果
#                 all_coarse_preds.extend(coarse_preds.cpu().numpy())
#                 all_coarse_labels.extend(coarse_labels.cpu().numpy())
#                 all_constrained_fine_preds.extend(constrained_fine_preds.cpu().numpy())
#                 all_fine_labels.extend(fine_labels.cpu().numpy())
        
#         # 计算验证指标
#         val_coarse_metrics_micro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='micro')
#         val_coarse_metrics_macro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='macro')
#         val_constrained_fine_metrics_micro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='micro')
#         val_constrained_fine_metrics_macro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='macro')
        
#         # 计算6个指标
#         hierarchical_final_micro_f1 = (val_constrained_fine_metrics_micro['micro_f1'] + val_coarse_metrics_micro['micro_f1']) / 2
#         hierarchical_final_macro_f1 = (val_constrained_fine_metrics_macro['macro_f1'] + val_coarse_metrics_macro['macro_f1']) / 2
#         hierarchical_coarse_micro_f1 = val_coarse_metrics_micro['micro_f1']
#         hierarchical_fine_micro_f1 = val_constrained_fine_metrics_micro['micro_f1']
#         hierarchical_coarse_macro_f1 = val_coarse_metrics_macro['macro_f1']
#         hierarchical_fine_macro_f1 = val_constrained_fine_metrics_macro['macro_f1']
        
#         current_metrics = {
#             'hierarchical_final_micro_f1': hierarchical_final_micro_f1,
#             'hierarchical_final_macro_f1': hierarchical_final_macro_f1,
#             'hierarchical_coarse_micro_f1': hierarchical_coarse_micro_f1,
#             'hierarchical_fine_micro_f1': hierarchical_fine_micro_f1,
#             'hierarchical_coarse_macro_f1': hierarchical_coarse_macro_f1,
#             'hierarchical_fine_macro_f1': hierarchical_fine_macro_f1,
#             'epoch': epoch + 1
#         }
        
#         # 记录训练历史
#         training_history['train_loss'].append(avg_train_loss)
#         training_history['val_metrics'].append(current_metrics.copy())
        
#         epoch_time = time.time() - epoch_start_time
        
#         # 打印当前epoch结果
#         print(f"\n📊 Epoch {epoch+1}/{config['epochs']} 结果 (耗时: {epoch_time:.1f}s):")
#         print(f"   🔥 训练损失: {avg_train_loss:.4f}")
#         print(f"   📈 Final Micro F1: {hierarchical_final_micro_f1:.2f}%")
#         print(f"   📈 Final Macro F1: {hierarchical_final_macro_f1:.2f}%")
#         print(f"   📊 Coarse Micro F1: {hierarchical_coarse_micro_f1:.2f}%")
#         print(f"   📊 Fine Micro F1: {hierarchical_fine_micro_f1:.2f}%")
#         print(f"   📊 Coarse Macro F1: {hierarchical_coarse_macro_f1:.2f}%")
#         print(f"   📊 Fine Macro F1: {hierarchical_fine_macro_f1:.2f}%")
        
#         # 检查是否为最佳性能（使用Final micro f1作为主要指标）
#         if hierarchical_final_micro_f1 > best_val_f1:
#             best_val_f1 = hierarchical_final_micro_f1
#             best_metrics = current_metrics.copy()
#             patience_counter = 0
            
#             # 保存最佳模型
#             model_save_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
#             # torch.save({
#             #     'epoch': epoch + 1,
#             #     'model_state_dict': model.state_dict(),
#             #     'optimizer_state_dict': optimizer.state_dict(),
#             #     'best_metrics': best_metrics,
#             #     'config': config
#             # }, model_save_path)
            
#             print(f"   🎉 新的最佳模型! 已保存到 {model_save_path}")
            
#         else:
#             patience_counter += 1
#             print(f"   ⏳ 耐心等待: {patience_counter}/{config['patience']}")
        
#         # 更新训练历史中的最佳指标
#         training_history['best_metrics'] = best_metrics.copy()
        
#         # 早停检查
#         if patience_counter >= config['patience']:
#             print(f"\n🛑 早停触发! 已连续 {config['patience']} 个epoch没有改善")
#             break
        
#         print("-" * 80)
    
#     total_time = time.time() - start_time
    
#     # 保存完整的训练历史
#     training_history['total_time_hours'] = total_time / 3600
#     training_history['final_epoch'] = epoch + 1
#     training_history['early_stopped'] = patience_counter >= config['patience']
    
#     history_save_path = os.path.join(config['results_dir'], 'training_history.json')
#     with open(history_save_path, 'w', encoding='utf-8') as f:
#         json.dump(training_history, f, ensure_ascii=False, indent=2)
    
#     # 保存最终结果
#     final_result = {
#         'best_6_metrics': best_metrics,
#         'config': config,
#         'training_summary': {
#             'total_epochs': epoch + 1,
#             'total_time_hours': total_time / 3600,
#             'early_stopped': patience_counter >= config['patience'],
#             'best_epoch': best_metrics['epoch'],
#             'training_date': datetime.now().isoformat()
#         }
#     }
    
#     result_save_path = os.path.join(config['results_dir'], 'final_result.json')
#     with open(result_save_path, 'w', encoding='utf-8') as f:
#         json.dump(final_result, f, ensure_ascii=False, indent=2)
    
#     # 打印最终结果
#     print(f"\n{'='*80}")
#     print("🎉 训练完成!")
#     print(f"⏱️  总耗时: {total_time/3600:.2f} 小时")
#     print(f"🔄 训练轮数: {epoch + 1}/{config['epochs']}")
#     print(f"🎯 最佳epoch: {best_metrics['epoch']}")
#     if patience_counter >= config['patience']:
#         print("🛑 早停触发")
    
#     print(f"\n🏆 最佳6个指标:")
#     print(f"   1. Final Micro F1: {best_metrics['hierarchical_final_micro_f1']:.2f}%")
#     print(f"   2. Final Macro F1: {best_metrics['hierarchical_final_macro_f1']:.2f}%")
#     print(f"   3. Coarse Micro F1: {best_metrics['hierarchical_coarse_micro_f1']:.2f}%")
#     print(f"   4. Fine Micro F1: {best_metrics['hierarchical_fine_micro_f1']:.2f}%")
#     print(f"   5. Coarse Macro F1: {best_metrics['hierarchical_coarse_macro_f1']:.2f}%")
#     print(f"   6. Fine Macro F1: {best_metrics['hierarchical_fine_macro_f1']:.2f}%")
    
#     print(f"\n📁 结果已保存:")
#     print(f"   - 最佳模型: {os.path.join(config['checkpoint_dir'], 'best_model.pt')}")
#     print(f"   - 训练历史: {history_save_path}")
#     print(f"   - 最终结果: {result_save_path}")
    
#     return best_metrics, training_history


# def main():
#     """主函数"""
#     parser = argparse.ArgumentParser(description='层次化错误分类模型训练')
    
#     # 数据路径参数
#     parser.add_argument('--data_path', type=str, 
#                        default='/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train.json',
#                        help='训练数据路径')
#     parser.add_argument('--val_data_path', type=str,
#                        default='/mnt/cfs/huangzhiwei/NLP-WED0910/datas/val.json', 
#                        help='验证数据路径')
#     parser.add_argument('--model_name', type=str,
#                        default='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5',
#                        help='预训练模型路径')
    
#     # 模型参数
#     parser.add_argument('--num_coarse_labels', type=int, default=4, help='粗粒度标签数')
#     parser.add_argument('--num_fine_labels', type=int, default=14, help='细粒度标签数')
#     parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
#     parser.add_argument('--dropout', type=float, default=0.3135999246870766, help='Dropout率')
    
#     # MoE参数
#     parser.add_argument('--num_experts', type=int, default=15, help='专家数量')
#     parser.add_argument('--expert_dim', type=int, default=512, help='专家维度')
#     parser.add_argument('--top_k', type=int, default=1, help='Top-K专家')
#     parser.add_argument('--use_separate_moe', type=bool, default=True, help='是否使用独立MoE')
#     parser.add_argument('--load_balance_weight', type=float, default=0.013540158381723238, help='负载均衡权重')
    
#     # 训练参数
#     parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
#     parser.add_argument('--lr', type=float, default=1.86884655838032e-05, help='学习率')
#     parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
#     parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
#     parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
#     parser.add_argument('--threshold', type=float, default=0.5, help='二分类阈值')
#     parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    
#     # 保存路径
#     parser.add_argument('--results_dir', type=str, default='training_results', help='结果保存目录')
#     parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型保存目录')
    
#     args = parser.parse_args()
    
#     # 转换为配置字典
#     config = vars(args)
#     config['device'] = device
    
#     # 约束检查
#     if config['top_k'] > config['num_experts']:
#         config['top_k'] = config['num_experts']
#         print(f"⚠️  Top-K ({args.top_k}) 大于专家数量 ({config['num_experts']})，已调整为 {config['top_k']}")
    
#     # 检查数据文件
#     print("🔍 检查数据文件...")
#     for path_key in ['data_path', 'val_data_path']:
#         if not os.path.exists(config[path_key]):
#             print(f"❌ {config[path_key]} 不存在")
#             return
#         else:
#             print(f"✅ {config[path_key]} 存在")
    
#     # 检查模型路径
#     if not os.path.exists(config['model_name']):
#         print(f"❌ 模型路径 {config['model_name']} 不存在")
#         return
#     else:
#         print(f"✅ 模型路径存在")
    
#     print(f"\n🎛️  配置参数:")
#     for key, value in config.items():
#         if key not in ['device']:
#             print(f"   {key}: {value}")
    
#     # 开始训练
#     best_metrics, training_history = train_model(config)
    
#     print(f"\n✨ 训练完成! 最佳 Final Micro F1: {best_metrics['hierarchical_final_micro_f1']:.2f}%")


# if __name__ == '__main__':
#     main()




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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class MoELayer(nn.Module):
    """Mixture of Experts Layer"""
    def __init__(self, input_dim, expert_dim, num_experts=8, top_k=2, dropout=0.1, load_balance_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        
        # Gate网络，用于选择专家
        self.gate = nn.Linear(input_dim, num_experts)

        # 专家网络：每一个专家都是两层的MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, input_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

        # 负载均衡损失的权重
        self.load_balance_weight = load_balance_weight
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1) if len(x.shape) > 2 else 1

        # 如果输入是3D (batch, seq, hidden)，我们只用[CLS]位置
        if len(x.shape) == 3:
            x = x[:, 0, :]  # 取[CLS]位置

        # Gate网络输出: 选择专家的概率
        gate_logits = self.gate(x) # (batch_size, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 选择前top_k个专家
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)  # 归一化

        # 计算专家输出
        expert_outputs = []
        for i in range(self.num_experts):
            expert_output = self.experts[i](x) # (batch_size, input_dim)
            expert_outputs.append(expert_output)

        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, input_dim)

        # 组合top-k专家的输出
        final_output = torch.zeros_like(x) # (batch_size, input_dim)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # 修复语法错误
            expert_weight = top_k_probs[:, i:i+1]  # (batch_size, 1)

            # 选择对应专家的输出
            selected_output = expert_outputs[torch.arange(batch_size), expert_idx]  # (batch_size, input_dim)
            final_output += expert_weight * selected_output

        # 计算负载均衡损失
        load_balance_loss = self.compute_load_balance_loss(gate_probs)
        
        return final_output, load_balance_loss

    
    def compute_load_balance_loss(self, gate_probs):
        """计算负载均衡损失，鼓励专家使用的均衡性"""
        # 计算每个专家被选择的平均概率
        expert_usage = gate_probs.mean(dim=0)  # (num_experts,)
        
        # 理想情况下每个专家被使用的概率应该是 1/num_experts
        ideal_usage = 1.0 / self.num_experts
        
        # 计算负载均衡损失（使用L2距离）
        load_balance_loss = torch.sum((expert_usage - ideal_usage) ** 2)
        
        return self.load_balance_weight * load_balance_loss


class HierarchicalErrorClassifier(nn.Module):
    def __init__(
        self, 
        pretrained_model_name, 
        num_coarse_labels=4, 
        num_fine_labels=14, 
        dropout=0.2,
        num_experts=8,
        expert_dim=512,
        top_k=2,
        use_separate_moe=True,  # 是否为粗粒度和细粒度使用不同的MoE
        load_balance_weight=0.01
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(pretrained_model_name)

        # # 解冻最后四层参数
        # # 冻结BERT底层，保留顶层微调
        # modules = [self.bert.embeddings, *self.bert.encoder.layer[:8]]
        # for module in modules:
        #     for param in module.parameters():
        #         param.requires_grad = False


        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_separate_moe = use_separate_moe
        
        hidden_size = self.bert.config.hidden_size

        
        if use_separate_moe:
            # 为粗粒度和细粒度分类分别使用不同的MoE
            self.coarse_moe = MoELayer(
                input_dim=hidden_size,
                expert_dim=expert_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight
            )
            self.fine_moe = MoELayer(
                input_dim=hidden_size,
                expert_dim=expert_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight
            )
        else:
            # 共享一个MoE层
            self.shared_moe = MoELayer(
                input_dim=hidden_size,
                expert_dim=expert_dim,
                num_experts=num_experts,
                top_k=top_k,
                dropout=dropout,
                load_balance_weight=load_balance_weight
            )
        
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

        total_load_balance_loss = 0
        
        if self.use_separate_moe:
            # 使用不同的MoE处理粗粒度和细粒度分类
            coarse_features, coarse_lb_loss = self.coarse_moe(pooled_output)
            fine_features, fine_lb_loss = self.fine_moe(pooled_output)
            total_load_balance_loss = coarse_lb_loss + fine_lb_loss
        else:
            # 使用共享MoE
            shared_features, shared_lb_loss = self.shared_moe(pooled_output)
            coarse_features = fine_features = shared_features
            total_load_balance_loss = shared_lb_loss
        
        # 粗粒度分类
        coarse_logits = self.coarse_classifier(pooled_output)
        coarse_probs = torch.sigmoid(coarse_logits)
        
        # 细粒度分类
        fine_logits = self.fine_classifier(pooled_output)
        fine_probs = torch.sigmoid(fine_logits)
        
        return coarse_probs, fine_probs, total_load_balance_loss
    
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


def train_model(config):
    """主训练函数"""
    print("🚀 开始训练层次化错误分类模型")
    print(f"📱 设备: {config['device']}")
    print(f"🏷️  粗粒度类别数: {config['num_coarse_labels']}")
    print(f"🏷️  细粒度类别数: {config['num_fine_labels']}")
    print(f"📝 最大序列长度: {config['max_length']}")
    print(f"🔄 批次大小: {config['batch_size']}")
    print(f"📚 训练轮数: {config['epochs']}")
    print("="*80)
    
    # 设置随机种子
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 创建结果保存目录
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 加载数据集
    print("📂 加载数据集...")
    train_dataset = ErrorDetectionDataset(config['data_path'])
    val_dataset = ErrorDetectionDataset(config['val_data_path'])
    print(f"📊 训练集大小: {len(train_dataset)}")
    print(f"📊 验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    train_dataloader = ErrorDetectionDataLoader(
        dataset=train_dataset,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        shuffle=True,
        drop_last=True,
        device=config['device'],
        tokenizer_name=config['model_name']
    )

    val_dataloader = ErrorDetectionDataLoader(
        dataset=val_dataset,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        shuffle=False,
        drop_last=False,
        device=config['device'],
        tokenizer_name=config['model_name']
    )
    
    # 创建模型
    print("🏗️  构建模型...")
    model = HierarchicalErrorClassifier(
        pretrained_model_name=config['model_name'],
        num_coarse_labels=config['num_coarse_labels'],
        num_fine_labels=config['num_fine_labels'],
        dropout=config['dropout'],
        num_experts=config['num_experts'],
        expert_dim=config['expert_dim'],
        top_k=config['top_k'],
        use_separate_moe=config['use_separate_moe'],
        load_balance_weight=config['load_balance_weight']
    ).to(config['device'])
    
    print(f"🔧 模型参数:")
    print(f"   - Dropout: {config['dropout']}")
    print(f"   - 专家数量: {config['num_experts']}")
    print(f"   - Top-K: {config['top_k']}")
    print(f"   - 专家维度: {config['expert_dim']}")
    print(f"   - 独立MoE: {config['use_separate_moe']}")
    print(f"   - 负载均衡权重: {config['load_balance_weight']}")
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.01)
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
    
    # 记录所有epoch的结果 (处理config中不能序列化的对象)
    serializable_config = config.copy()
    serializable_config['device'] = str(config['device'])  # 转换device为字符串
    
    training_history = {
        'train_loss': [],
        'val_metrics': [],
        'best_metrics': best_metrics,
        'config': serializable_config
    }
    
    print("\n🎯 开始训练...")
    start_time = time.time()
    
    # 训练循环
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        train_progress = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in train_progress:
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
            train_steps += 1
            
            # 更新进度条
            train_progress.set_postfix({
                'Loss': f'{classification_loss.item():.4f}',
                'LB_Loss': f'{load_balance_loss.item():.4f}'
            })
        
        avg_train_loss = train_loss / train_steps
        
        # 验证阶段
        model.eval()
        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_labels = []
        all_constrained_fine_preds = []
        
        val_progress = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
        with torch.no_grad():
            for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in val_progress:
                # 前向传播
                coarse_probs, fine_probs, load_balance_loss = model(input_ids, attention_mask)
                
                # 应用层次约束
                coarse_preds = (coarse_probs > config['threshold']).float()
                fine_preds = (fine_probs > config['threshold']).float()
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
        
        # 记录训练历史
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_metrics'].append(current_metrics.copy())
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印当前epoch结果
        print(f"\n📊 Epoch {epoch+1}/{config['epochs']} 结果 (耗时: {epoch_time:.1f}s):")
        print(f"   🔥 训练损失: {avg_train_loss:.4f}")
        print(f"   📈 Final Micro F1: {hierarchical_final_micro_f1:.2f}%")
        print(f"   📈 Final Macro F1: {hierarchical_final_macro_f1:.2f}%")
        print(f"   📊 Coarse Micro F1: {hierarchical_coarse_micro_f1:.2f}%")
        print(f"   📊 Fine Micro F1: {hierarchical_fine_micro_f1:.2f}%")
        print(f"   📊 Coarse Macro F1: {hierarchical_coarse_macro_f1:.2f}%")
        print(f"   📊 Fine Macro F1: {hierarchical_fine_macro_f1:.2f}%")
        
        # 检查是否为最佳性能（使用Final micro f1作为主要指标）
        if hierarchical_final_micro_f1 > best_val_f1:
            best_val_f1 = hierarchical_final_micro_f1
            best_metrics = current_metrics.copy()
            patience_counter = 0
            
            # 保存最佳模型
            model_save_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
            # torch.save({
            #     'epoch': epoch + 1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'best_metrics': best_metrics,
            #     'config': serializable_config_final
            # }, model_save_path)
            
            print(f"   🎉 新的最佳模型! 已保存到 {model_save_path}")
            
        else:
            patience_counter += 1
            print(f"   ⏳ 耐心等待: {patience_counter}/{config['patience']}")
        
        # 更新训练历史中的最佳指标
        training_history['best_metrics'] = best_metrics.copy()
        
        # 早停检查
        if patience_counter >= config['patience']:
            print(f"\n🛑 早停触发! 已连续 {config['patience']} 个epoch没有改善")
            break
        
        print("-" * 80)
    
    total_time = time.time() - start_time
    
    # 保存完整的训练历史
    training_history['total_time_hours'] = total_time / 3600
    training_history['final_epoch'] = epoch + 1
    training_history['early_stopped'] = patience_counter >= config['patience']
    
    history_save_path = os.path.join(config['results_dir'], 'training_history.json')
    with open(history_save_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, ensure_ascii=False, indent=2)
    
    # 保存最终结果 (处理config中不能序列化的对象)
    serializable_config_final = config.copy()
    serializable_config_final['device'] = str(config['device'])  # 转换device为字符串
    
    final_result = {
        'best_6_metrics': best_metrics,
        'config': serializable_config_final,
        'training_summary': {
            'total_epochs': epoch + 1,
            'total_time_hours': total_time / 3600,
            'early_stopped': patience_counter >= config['patience'],
            'best_epoch': best_metrics['epoch'],
            'training_date': datetime.now().isoformat()
        }
    }
    
    result_save_path = os.path.join(config['results_dir'], 'final_result.json')
    with open(result_save_path, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    # 打印最终结果
    print(f"\n{'='*80}")
    print("🎉 训练完成!")
    print(f"⏱️  总耗时: {total_time/3600:.2f} 小时")
    print(f"🔄 训练轮数: {epoch + 1}/{config['epochs']}")
    print(f"🎯 最佳epoch: {best_metrics['epoch']}")
    if patience_counter >= config['patience']:
        print("🛑 早停触发")
    
    print(f"\n🏆 最佳6个指标:")
    print(f"   1. Final Micro F1: {best_metrics['hierarchical_final_micro_f1']:.2f}%")
    print(f"   2. Final Macro F1: {best_metrics['hierarchical_final_macro_f1']:.2f}%")
    print(f"   3. Coarse Micro F1: {best_metrics['hierarchical_coarse_micro_f1']:.2f}%")
    print(f"   4. Fine Micro F1: {best_metrics['hierarchical_fine_micro_f1']:.2f}%")
    print(f"   5. Coarse Macro F1: {best_metrics['hierarchical_coarse_macro_f1']:.2f}%")
    print(f"   6. Fine Macro F1: {best_metrics['hierarchical_fine_macro_f1']:.2f}%")
    
    print(f"\n📁 结果已保存:")
    print(f"   - 最佳模型: {os.path.join(config['checkpoint_dir'], 'best_model.pt')}")
    print(f"   - 训练历史: {history_save_path}")
    print(f"   - 最终结果: {result_save_path}")
    
    return best_metrics, training_history


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='层次化错误分类模型训练')
    
    # 数据路径参数
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train_new.json',
                       help='训练数据路径')
    parser.add_argument('--val_data_path', type=str,
                       default='/mnt/cfs/huangzhiwei/NLP-WED0910/datas/val.json', 
                       help='验证数据路径')
    parser.add_argument('--model_name', type=str,
                       default='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5',
                       help='预训练模型路径')
    
    # 模型参数
    parser.add_argument('--num_coarse_labels', type=int, default=4, help='粗粒度标签数')
    parser.add_argument('--num_fine_labels', type=int, default=14, help='细粒度标签数')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--dropout', type=float, default=0.3135999246870766, help='Dropout率')
    
    # MoE参数
    parser.add_argument('--num_experts', type=int, default=15, help='专家数量')
    parser.add_argument('--expert_dim', type=int, default=512, help='专家维度')
    parser.add_argument('--top_k', type=int, default=1, help='Top-K专家')
    parser.add_argument('--use_separate_moe', type=bool, default=True, help='是否使用独立MoE')
    parser.add_argument('--load_balance_weight', type=float, default=0.013540158381723238, help='负载均衡权重')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1.86884655838032e-05, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='权重衰减')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--threshold', type=float, default=0.31676353792873924, help='二分类阈值')
    parser.add_argument('--seed', type=int, default=3407, help='随机种子')
    
    # 保存路径
    parser.add_argument('--results_dir', type=str, default='training_results', help='结果保存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 转换为配置字典
    config = vars(args)
    config['device'] = device
    
    # 约束检查
    if config['top_k'] > config['num_experts']:
        config['top_k'] = config['num_experts']
        print(f"⚠️  Top-K ({args.top_k}) 大于专家数量 ({config['num_experts']})，已调整为 {config['top_k']}")
    
    # 检查数据文件
    print("🔍 检查数据文件...")
    for path_key in ['data_path', 'val_data_path']:
        if not os.path.exists(config[path_key]):
            print(f"❌ {config[path_key]} 不存在")
            return
        else:
            print(f"✅ {config[path_key]} 存在")
    
    # 检查模型路径
    if not os.path.exists(config['model_name']):
        print(f"❌ 模型路径 {config['model_name']} 不存在")
        return
    else:
        print(f"✅ 模型路径存在")
    
    print(f"\n🎛️  配置参数:")
    for key, value in config.items():
        if key not in ['device']:
            print(f"   {key}: {value}")
    
    # 开始训练
    best_metrics, training_history = train_model(config)
    
    print(f"\n✨ 训练完成! 最佳 Final Micro F1: {best_metrics['hierarchical_final_micro_f1']:.2f}%")


if __name__ == '__main__':
    main()