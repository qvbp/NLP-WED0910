import json
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def load_data(validation_file, prediction_file):
    """加载验证集和预测结果文件"""
    with open(validation_file, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
        
    with open(prediction_file, 'r', encoding='utf-8') as f:
        prediction_data = json.load(f)
        
    # 确保validation_data是列表
    if isinstance(validation_data, dict) and 'predictions' in validation_data:
        validation_data = validation_data['predictions']
    
    # 确保prediction_data是列表
    if isinstance(prediction_data, dict) and 'predictions' in prediction_data:
        prediction_data = prediction_data['predictions']
        
    return validation_data, prediction_data

def align_data_by_sent_id(validation_data, prediction_data):
    """根据sent_id对齐验证集和预测结果"""
    # 创建验证集的sent_id到数据的映射
    val_dict = {item['sent_id']: item for item in validation_data}
    
    aligned_val = []
    aligned_pred = []
    
    for pred_item in prediction_data:
        sent_id = pred_item['sent_id']
        if sent_id in val_dict:
            aligned_val.append(val_dict[sent_id])
            aligned_pred.append(pred_item)
    
    return aligned_val, aligned_pred

def extract_labels(data, label_type):
    """提取指定类型的标签"""
    labels = []
    for item in data:
        if label_type in item:
            labels.append(item[label_type])
        else:
            labels.append([])  # 如果没有该字段，使用空列表
    return labels

def calculate_micro_macro_f1(y_true_labels, y_pred_labels):
    """计算micro和macro F1分数"""
    # 获取所有可能的标签
    all_labels = set()
    for labels in y_true_labels + y_pred_labels:
        all_labels.update(labels)
    all_labels = sorted(list(all_labels))
    
    # 转换为多标签二进制格式
    mlb = MultiLabelBinarizer(classes=all_labels)
    y_true_binary = mlb.fit_transform(y_true_labels)
    y_pred_binary = mlb.transform(y_pred_labels)
    
    # 计算micro和macro F1
    micro_f1 = f1_score(y_true_binary, y_pred_binary, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true_binary, y_pred_binary, average='macro', zero_division=0)
    
    return micro_f1, macro_f1

def calculate_all_metrics(validation_file, prediction_file):
    """计算所有6个F1指标"""
    # 加载数据
    validation_data, prediction_data = load_data(validation_file, prediction_file)
    
    # 对齐数据
    aligned_val, aligned_pred = align_data_by_sent_id(validation_data, prediction_data)
    
    print(f"对齐后的数据量: {len(aligned_val)}")
    
    # 提取coarse标签 (CourseGrainedErrorType)
    val_coarse = extract_labels(aligned_val, 'CourseGrainedErrorType')
    pred_coarse = extract_labels(aligned_pred, 'CourseGrainedErrorType')
    
    # 提取fine标签 (FineGrainedErrorType)
    val_fine = extract_labels(aligned_val, 'FineGrainedErrorType')
    pred_fine = extract_labels(aligned_pred, 'FineGrainedErrorType')
    
    # 计算coarse的micro和macro F1
    coarse_micro_f1, coarse_macro_f1 = calculate_micro_macro_f1(val_coarse, pred_coarse)
    
    # 计算fine的micro和macro F1
    fine_micro_f1, fine_macro_f1 = calculate_micro_macro_f1(val_fine, pred_fine)
    
    # 计算final的micro和macro F1 (两个F1的平均)
    final_micro_f1 = (coarse_micro_f1 + fine_micro_f1) / 2
    final_macro_f1 = (coarse_macro_f1 + fine_macro_f1) / 2
    
    # 返回结果
    results = {
        'final_micro_f1': final_micro_f1*100,
        'final_macro_f1': final_macro_f1*100,
        'coarse_micro_f1': coarse_micro_f1*100,
        'fine_micro_f1': fine_micro_f1*100,
        'coarse_macro_f1': coarse_macro_f1*100,
        'fine_macro_f1': fine_macro_f1*100
    }
    
    return results

def main():
    """主函数"""
    # 设置文件路径 - 交换了文件路径，因为看起来它们被错误地交换了
    prediction_file = '/mnt/cfs/huangzhiwei/NLP-WED0910/qwen3-256b-data/val_result_update.json'  # 预测结果文件
    validation_file = '/mnt/cfs/huangzhiwei/NLP-WED0910/datas/val.json'  # 验证集文件
    
    try:
        # 计算所有指标
        results = calculate_all_metrics(validation_file, prediction_file)
        
        # 打印结果
        print("\n=== F1 指标结果 ===")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        
        # 保存结果到文件
        with open('f1_metrics_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("\n结果已保存到 f1_metrics_results.json")
        
    except Exception as e:
        print(f"计算过程中出现错误: {e}")

if __name__ == "__main__":
    main()
