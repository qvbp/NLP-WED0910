import torch
import torch.nn as nn
import os
import wandb
import random
import argparse
import numpy as np
import json
from tqdm import tqdm
from transformers import BertModel, AutoModel
from transformers import AdamW
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
        tokenizer_name='/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base'
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
        # token_type_ids = encoded.get('token_type_ids', None)
        
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
        # corase_probs = coarse_logits
        
        # # 细粒度分类
        fine_logits = self.fine_classifier(pooled_output)
        fine_probs = torch.sigmoid(fine_logits)
        
        # 细粒度分类，包含14个补充信息
        # fine_logits = torch.matmul(pooled_output, self.fine_description.T)
        # fine_logits = self.fine_classifier(pooled_output)
        # fine_probs = torch.sigmoid(fine_logits)
        # fine_probs = fine_logits
        
        return coarse_probs, fine_probs
    
    def apply_hierarchical_constraint(self, coarse_preds, fine_preds):
        """
        应用层次约束：如果粗粒度类别预测为负，则该粗粒度下的所有细粒度类别均设为负
        
        Args:
            coarse_preds: 粗粒度预测结果，shape [batch_size, num_coarse_labels]
            fine_preds: 细粒度预测结果，shape [batch_size, num_fine_labels]
            
        Returns:
            应用约束后的细粒度预测结果
        """
        constrained_fine_preds = fine_preds.clone()
        
        # 遍历每个样本
        for i in range(coarse_preds.size(0)):  # 遍历每个样本（第一个维度是批次大小）
            # 对每个粗粒度类别
            for coarse_idx, fine_indices in self.coarse_to_fine_indices.items():
                # 如果粗粒度为负，则对应的细粒度全部设为负
                if coarse_preds[i, coarse_idx] == 0:
                    constrained_fine_preds[i, fine_indices] = 0
        
        return constrained_fine_preds


# 导入自定义模块
# from error_detection_dataset import ErrorDetectionDataset
# from error_detection_dataloader import ErrorDetectionDataLoader
# from hierarchical_classifier_model import HierarchicalErrorClassifier

# 如果分项目的时候就可以使用这个参数解析函数
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5')
    parser.add_argument('--num_coarse_labels', type=int, default=4)
    parser.add_argument('--num_fine_labels', type=int, default=14)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--freeze_pooler', action='store_true', help='Flag to freeze the pooler layer')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--project', type=str, default='hierarchical_error_detection')
    parser.add_argument('--entity', type=str, default='akccc')
    parser.add_argument('--name', type=str, required=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_path', type=str, default='../datas/train.json')
    parser.add_argument('--val_data_path', type=str, default='../datas/val.json')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5)
    
    return parser.parse_args()



# # 如果在Jupyter Notebook中运行，可以使用这个自定义参数函数替代argparser
# def get_default_configs():
#     """在Jupyter环境中使用的默认配置，避免argparse解析错误"""
#     class Args:
#         def __init__(self):
#             # self.model_name = '/mnt/cfs/huangzhiwei/BAE2025/models/deberta-v3-base'
#             self.model_name = '/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5'
#             self.num_coarse_labels = 4
#             self.num_fine_labels = 14
#             self.dropout = 0.2
#             self.freeze_pooler = False
#             self.batch_size = 8
#             self.max_length = 128
#             self.lr = 2e-5
#             self.epochs = 40
#             self.device = device
#             self.project = 'hierarchical_error_detection'
#             self.name = None
#             self.seed = 42
#             self.data_path = '../datas/train.json'
#             self.val_data_path = '../datas/val.json'
#             self.checkpoint_dir = 'checkpoints'
#             self.threshold = 0.5
#             self.patience = 5
#             self.exp_name = 'default_run'
#             self.weight_decay = 0.01
#     return Args()


def calculate_metrics(labels, predictions, average='micro'):
    """
    计算各种评估指标
    
    Args:
        labels: 真实标签
        predictions: 预测标签
        average: 平均方法，'micro'或'macro'
        
    Returns:
        包含各种指标的字典
    """
    # 将数组转换为numpy格式以确保兼容性
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    # 计算微平均和宏平均的F1分数
    micro_f1 = f1_score(labels, predictions, average='micro')
    macro_f1 = f1_score(labels, predictions, average='macro')
    
    # 计算样本级别的准确率（每个样本的所有标签都要正确）
    sample_acc = accuracy_score(labels, predictions)
    
    return {
        'micro_f1': micro_f1 * 100,  # 转换为百分比
        'macro_f1': macro_f1 * 100,
        'accuracy': sample_acc * 100
    }

def train(configs):
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

    # 定义损失函数
    criterion = nn.BCELoss()

    # 定义优化器
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs.lr,
        weight_decay=configs.weight_decay
    )

    # 初始化最佳验证损失和早停计数器
    best_val_f1 = 0
    patience_counter = 0
    
    # 获取标签映射，用于后续预测结果记录
    coarse_label_map = {v: k for k, v in val_dataset.get_coarse_labels().items()}
    fine_label_map = {v: k for k, v in val_dataset.get_fine_labels().items()}
    
    # print("coarse_label_map:", coarse_label_map)
    # print("fine_label_map:", fine_label_map)    
    # # 终止运行，用来debug
    # return
    
    
    # 训练循环
    for epoch in range(configs.epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        all_constrained_fine_preds = []
        
        with tqdm(
            train_dataloader,
            total=len(train_dataloader),
            desc=f'Epoch {epoch + 1}/{configs.epochs}',
            unit='batch',
            ncols=100
        ) as pbar:
            for input_ids, attention_mask, coarse_labels, fine_labels, sent_ids in pbar:
                optimizer.zero_grad()
                
                # 前向传播
                coarse_probs, fine_probs = model(input_ids, attention_mask)
                
                # print("coarse_probs:", coarse_probs)
                
                # 计算损失
                coarse_loss = criterion(coarse_probs, coarse_labels)
                fine_loss = criterion(fine_probs, fine_labels)
                loss = coarse_loss + fine_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 收集预测结果
                coarse_preds = (coarse_probs > configs.threshold).float().cpu().numpy()
                fine_preds = (fine_probs > configs.threshold).float().cpu().numpy()
                constrained_fine_preds = model.apply_hierarchical_constraint(
                    (coarse_probs > configs.threshold).float(), 
                    (fine_probs > configs.threshold).float()
                ).cpu().numpy()
                
                all_coarse_preds.extend(coarse_preds)
                all_coarse_labels.extend(coarse_labels.cpu().numpy())
                all_fine_preds.extend(fine_preds)
                all_fine_labels.extend(fine_labels.cpu().numpy())
                all_constrained_fine_preds.extend(constrained_fine_preds)
                
                # 更新进度条
                pbar.set_postfix(
                    loss=f'{loss.item():.3f}',
                    coarse_loss=f'{coarse_loss.item():.3f}',
                    fine_loss=f'{fine_loss.item():.3f}'
                )
        
        # 计算训练指标
        train_loss = train_loss / len(train_dataloader)
        
        # 计算各种评估指标
        train_coarse_metrics_micro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='micro')
        train_coarse_metrics_macro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='macro')
        train_fine_metrics_micro = calculate_metrics(all_fine_labels, all_fine_preds, average='micro')
        train_fine_metrics_macro = calculate_metrics(all_fine_labels, all_fine_preds, average='macro')
        train_constrained_fine_metrics_micro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='micro')
        train_constrained_fine_metrics_macro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='macro')
        
        # 保存模型文件
        # checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')
        # torch.save(model.state_dict(), checkpoint_path)
        

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_coarse_preds = []
        all_coarse_labels = []
        all_fine_preds = []
        all_fine_labels = []
        all_constrained_fine_preds = []
        
        # 记录验证集和测试集每个句子的真实标签和预测结果
        val_sentence_predictions = []

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
                
                # 记录当前 batch 中每个样本的预测结果和真实标签
                coarse_preds_np = coarse_preds.cpu().numpy()
                fine_preds_np = constrained_fine_preds.cpu().numpy()
                
                # 获取当前批次的原始句子
                # 由于sent_ids可能是数字或者直接是句子标识符，视情况处理
                # 这里我们使用索引从数据集中获取句子
                batch_sentences = []
                for sid in sent_ids:
                    # 如果sent_ids是数字索引
                    if isinstance(sid, int) or (isinstance(sid, torch.Tensor) and sid.numel() == 1):
                        idx = sid if isinstance(sid, int) else sid.item()
                        if idx >= 0 and idx < len(val_dataset):
                            batch_sentences.append(val_dataset.data[idx][0])
                        else:
                            # 如果索引无效，使用一个默认值
                            batch_sentences.append(f"[Unknown sentence with ID {sid}]")
                    else:
                        # 如果sent_ids本身就是句子标识符（字符串）
                        batch_sentences.append(str(sid))
                
                for i in range(len(batch_sentences)):
                    coarse_indices = np.where(coarse_preds_np[i] == 1)[0]
                    fine_indices = np.where(fine_preds_np[i] == 1)[0]
                    predicted_coarse = [coarse_label_map[idx] for idx in coarse_indices]
                    predicted_fine = [fine_label_map[idx] for idx in fine_indices]
                    
                    # 真实标签
                    true_coarse_indices = np.where(coarse_labels[i].cpu().numpy() == 1)[0]
                    true_fine_indices = np.where(fine_labels[i].cpu().numpy() == 1)[0]
                    true_coarse = [coarse_label_map[idx] for idx in true_coarse_indices]
                    true_fine = [fine_label_map[idx] for idx in true_fine_indices]

                    val_sentence_predictions.append({
                        'sentence': batch_sentences[i],
                        'predicted_coarse': predicted_coarse,
                        'predicted_fine': predicted_fine,
                        'true_coarse': true_coarse,
                        'true_fine': true_fine
                    })
                    
        # 保存验证集预测结果和真实标签到文件
        val_pred_file = os.path.join(checkpoint_dir, f"val_predictions_epoch_{epoch+1}.json")
        with open(val_pred_file, "w", encoding="utf-8") as f:
            json.dump(val_sentence_predictions, f, ensure_ascii=False, indent=4)
        
        # 计算验证指标
        val_loss = val_loss / len(val_dataloader)
        
        # 计算各种评估指标
        val_coarse_metrics_micro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='micro')
        val_coarse_metrics_macro = calculate_metrics(all_coarse_labels, all_coarse_preds, average='macro')
        val_fine_metrics_micro = calculate_metrics(all_fine_labels, all_fine_preds, average='micro')
        val_fine_metrics_macro = calculate_metrics(all_fine_labels, all_fine_preds, average='macro')
        val_constrained_fine_metrics_micro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='micro')
        val_constrained_fine_metrics_macro = calculate_metrics(all_fine_labels, all_constrained_fine_preds, average='macro')
        
        # 输出训练和验证指标
        print(f'\nEpoch {epoch+1}/{configs.epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Train Coarse-grained Metrics:')
        print(f'    Micro F1: {train_coarse_metrics_micro["micro_f1"]:.2f}')
        print(f'    Macro F1: {train_coarse_metrics_macro["macro_f1"]:.2f}')
        print(f'    Accuracy: {train_coarse_metrics_micro["accuracy"]:.2f}')
        print(f'  Train Fine-grained Metrics (Unconstrained):')
        print(f'    Micro F1: {train_fine_metrics_micro["micro_f1"]:.2f}')
        print(f'    Macro F1: {train_fine_metrics_macro["macro_f1"]:.2f}')
        print(f'    Accuracy: {train_fine_metrics_micro["accuracy"]:.2f}')
        print(f'  Train Fine-grained Metrics (Constrained):')
        print(f'    Micro F1: {train_constrained_fine_metrics_micro["micro_f1"]:.2f}')
        print(f'    Macro F1: {train_constrained_fine_metrics_macro["macro_f1"]:.2f}')
        print(f'    Accuracy: {train_constrained_fine_metrics_micro["accuracy"]:.2f}')
        print("+"*50)
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Coarse-grained Metrics:')
        print(f'    Micro F1: {val_coarse_metrics_micro["micro_f1"]:.2f}')
        print(f'    Macro F1: {val_coarse_metrics_macro["macro_f1"]:.2f}')
        print(f'    Accuracy: {val_coarse_metrics_micro["accuracy"]:.2f}')
        print(f'  Val Fine-grained Metrics (Unconstrained):')
        print(f'    Micro F1: {val_fine_metrics_micro["micro_f1"]:.2f}')
        print(f'    Macro F1: {val_fine_metrics_macro["macro_f1"]:.2f}')
        print(f'    Accuracy: {val_fine_metrics_micro["accuracy"]:.2f}')
        print(f'  Val Fine-grained Metrics (Constrained):')
        print(f'    Micro F1: {val_constrained_fine_metrics_micro["micro_f1"]:.2f}')
        print(f'    Macro F1: {val_constrained_fine_metrics_macro["macro_f1"]:.2f}')
        print(f'    Accuracy: {val_constrained_fine_metrics_micro["accuracy"]:.2f}')
        
        # 检查是否保存最佳模型并应用早停
        f1_score = (val_coarse_metrics_micro["micro_f1"] + val_constrained_fine_metrics_micro["micro_f1"])/2
        if f1_score > best_val_f1:

            # 以表格形式输出所有指标（与给定的评估表格格式一致）
            print("\n")
            print("******************这是层次分类方法的结果********************")
            print("\n")
            print("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
            print("| {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} |".format(
                "Final", "Final", "Course-", "Fine-grained", "Course-", "Fine-grained"))
            print("| {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} |".format(
                "micro f1", "macro f1", "grained micro f1", "micro f1", "grained macro f1", "macro f1"))
            print("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
            print("| {:<13.2f} | {:<13.2f} | {:<13.2f} | {:<13.2f} | {:<13.2f} | {:<13.2f} |".format(
                (val_constrained_fine_metrics_micro['micro_f1'] + val_coarse_metrics_micro['micro_f1']) / 2,
                (val_constrained_fine_metrics_macro['macro_f1'] + val_coarse_metrics_macro['macro_f1']) / 2,
                val_coarse_metrics_micro['micro_f1'],
                val_constrained_fine_metrics_micro['micro_f1'],
                val_coarse_metrics_macro['macro_f1'],
                val_constrained_fine_metrics_macro['macro_f1']
            ))
            print("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
            print("\n")

            print("\n")
            print("******************这是正常4和14分类方法的结果********************")
            print("\n")
            print("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
            print("| {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} |".format(
                "Final", "Final", "Course-", "Fine-grained", "Course-", "Fine-grained"))
            print("| {:<13} | {:<13} | {:<13} | {:<13} | {:<13} | {:<13} |".format(
                "micro f1", "macro f1", "grained micro f1", "micro f1", "grained macro f1", "macro f1"))
            print("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
            print("| {:<13.2f} | {:<13.2f} | {:<13.2f} | {:<13.2f} | {:<13.2f} | {:<13.2f} |".format(
                (val_fine_metrics_micro['micro_f1'] + val_coarse_metrics_micro['micro_f1']) / 2,
                (val_fine_metrics_macro['macro_f1'] + val_coarse_metrics_macro['macro_f1']) / 2,
                val_coarse_metrics_micro['micro_f1'],
                val_fine_metrics_micro['micro_f1'],
                val_coarse_metrics_macro['macro_f1'],
                val_fine_metrics_macro['macro_f1']
            ))
            print("+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+" + "-"*15 + "+")
            print("\n")

            best_val_f1 = f1_score
            # torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= configs.patience:
                print('Early stopping triggered.')
                break

        
# 在以下主函数中添加判断Jupyter环境的逻辑
if __name__ == '__main__':
    # 判断是否在Jupyter环境中运行
    try:
        # 检查是否在Jupyter中运行
        get_ipython = globals().get('get_ipython', None)
        if get_ipython and 'IPKernelApp' in get_ipython().config:
            # 在Jupyter环境中运行，使用默认配置
            print("Running in Jupyter environment, using default configs")
            configs = get_default_configs()
        else:
            # 在命令行环境中运行，使用argparse
            configs = argparser()
    except:
        # 任何异常都使用argparse处理
        configs = argparser()
    
    # 设置实验名称
    if configs.name is None:
        configs.exp_name = \
            f'{os.path.basename(configs.model_name)}' + \
            f'{"_fp" if configs.freeze_pooler else ""}' + \
            f'_b{configs.batch_size}_e{configs.epochs}' + \
            f'_len{configs.max_length}_lr{configs.lr}'
    else:
        configs.exp_name = configs.name
    
    # 设置设备
    if configs.device is None:
        configs.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    # 调用训练函数
    train(configs)