import torch
from transformers import BertTokenizer, BertForMaskedLM
import torch.nn.functional as F

class BERTMaskPredictor:
    def __init__(self, model_name='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5'):
        """
        初始化BERT掩码预测器
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.model.eval()
        
    def predict_mask(self, text, top_k=5):
        """
        预测掩码位置的词
        
        Args:
            text: 包含[MASK]的文本
            top_k: 返回前k个预测结果
        
        Returns:
            预测结果列表
        """
        # 编码文本
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # 找到[MASK]的位置
        mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]
        
        if len(mask_token_index) == 0:
            return "文本中没有找到[MASK]标记"
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits
        
        # 获取[MASK]位置的预测
        mask_predictions = predictions[0, mask_token_index, :]
        
        # 获取top-k预测
        top_k_tokens = torch.topk(mask_predictions, top_k, dim=1)
        
        results = []
        for i, mask_pos in enumerate(mask_token_index):
            mask_results = []
            for j in range(top_k):
                token_id = top_k_tokens.indices[i][j].item()
                token = self.tokenizer.decode([token_id])
                prob = F.softmax(mask_predictions[i], dim=0)[token_id].item()
                mask_results.append({
                    'token': token,
                    'probability': prob,
                    'score': top_k_tokens.values[i][j].item()
                })
            results.append(mask_results)
        
        return results
    
    def auto_mask_and_predict(self, original_text, mask_word, top_k=5):
        """
        自动将指定词替换为[MASK]并预测
        
        Args:
            original_text: 原始文本
            mask_word: 要掩码的词
            top_k: 返回前k个预测结果
        """
        # 替换指定词为[MASK]
        masked_text = original_text.replace(mask_word, '[MASK]')
        
        print(f"原文: {original_text}")
        print(f"掩码后: {masked_text}")
        print("-" * 50)
        
        # 预测
        results = self.predict_mask(masked_text, top_k)
        
        return results
    
    def batch_predict(self, texts, top_k=5):
        """
        批量预测多个包含[MASK]的文本
        """
        all_results = []
        for text in texts:
            result = self.predict_mask(text, top_k)
            all_results.append({
                'text': text,
                'predictions': result
            })
        return all_results

# 使用示例
def main():
    # 初始化预测器
    predictor = BERTMaskPredictor()
    
    # 示例1：直接预测包含[MASK]的文本
    print("=== 示例1：直接预测 ===")
    text_with_mask = "周末时，老师邀请我与其他的同学去为母校的同学们[MASK]一位著名的冬奥运动员。"
    results = predictor.predict_mask(text_with_mask, top_k=5)
    
    print(f"输入文本: {text_with_mask}")
    print("预测结果:")
    for i, mask_results in enumerate(results):
        print(f"[MASK] 位置 {i+1}:")
        for j, pred in enumerate(mask_results):
            print(f"  {j+1}. {pred['token']} (概率: {pred['probability']:.4f})")
    
    print("\n" + "="*60 + "\n")
    
    # 示例2：自动掩码并预测
    print("=== 示例2：自动掩码预测 ===")
    original_text = "周末时，老师邀请我与其他的同学去为母校的同学们举行一位著名的冬奥运动员。"
    mask_word = "举行"
    
    results = predictor.auto_mask_and_predict(original_text, mask_word, top_k=5)
    
    print("预测结果:")
    for i, mask_results in enumerate(results):
        for j, pred in enumerate(mask_results):
            print(f"  {j+1}. {pred['token']} (概率: {pred['probability']:.4f})")
    
    print("\n" + "="*60 + "\n")
    
    # 示例3：批量预测
    print("=== 示例3：批量预测 ===")
    batch_texts = [
        "今天天气很[MASK]。",
        "我喜欢[MASK]音乐。",
        "这本书非常[MASK]。"
    ]
    
    batch_results = predictor.batch_predict(batch_texts, top_k=3)
    
    for item in batch_results:
        print(f"文本: {item['text']}")
        print("预测:")
        for i, mask_results in enumerate(item['predictions']):
            for j, pred in enumerate(mask_results):
                print(f"  {j+1}. {pred['token']} (概率: {pred['probability']:.4f})")
        print("-" * 30)

# 如果需要在分类任务中使用MLM作为辅助任务
class BERTClassifierWithMLM:
    def __init__(self, model_name='/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5', num_classes=2):
        """
        结合MLM和分类任务的BERT模型
        """
        from transformers import BertForSequenceClassification
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.classifier = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.mlm_model = BertForMaskedLM.from_pretrained(model_name)
        
    def train_with_mlm(self, texts, labels, mlm_texts, learning_rate=2e-5):
        """
        同时训练分类和MLM任务
        这可以作为多任务学习或继续预训练的方式
        """
        # 这里是一个框架示例，实际实现需要更详细的训练循环
        print("多任务训练框架 - 需要根据具体需求实现训练循环")
        pass

if __name__ == "__main__":
    main()