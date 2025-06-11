import json
import requests
import time
import random
from typing import List, Dict, Any
import os
from tqdm import tqdm

class QwenLabelPredictor:
    """使用千问API进行病句类型预测"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 定义标签体系
        self.coarse_labels = [
            "字符级错误",
            "成分残缺型错误", 
            "成分赘余型错误",
            "成分搭配不当型错误"
        ]
        
        self.fine_labels = [
            "错别字错误", "缺字漏字", "缺少标点", "错用标点",
            "主语不明", "谓语残缺", "宾语残缺", "其他成分残缺",
            "主语多余", "虚词多余", "其他成分多余",
            "语序不当", "动宾搭配不当", "其他搭配不当"
        ]
    
    def create_prediction_prompt(self, texts: List[str], include_probability: bool = True) -> str:
        """创建预测prompt"""
        
        base_prompt = """你是一位资深的中文语法专家和语言学教授，专门研究中小学作文中的病句识别与分析。你具有深厚的汉语语法理论功底和丰富的病句诊断经验，能够准确识别各种类型的语法错误，包括字符错误、语法结构问题、语义搭配不当等。

## 任务定义
中小学作文病句类型识别是一个多标签分类问题，需要你运用专业的语言学知识，准确预测一条句子包含哪些类型的病句。病句类型标签涵盖词法、句法、语义错误，本次识别任务共定义了4个粗粒度错误类型和14个细粒度错误类型。

## 分析步骤
请按以下步骤进行专业分析：
1. **语义理解**：仔细阅读句子，准确理解句子要表达的基本含义
2. **结构分析**：分析句子的语法结构（主谓宾、定状补等成分）
3. **字符检查**：检查是否存在错别字、缺漏字、标点符号错误
4. **成分诊断**：检查是否存在成分残缺、赘余或搭配不当问题
5. **综合判断**：根据分析结果确定所有相关的错误类型

## 标签体系说明
### 粗粒度错误类型（4类）：
1. **字符级错误**：包括错别字、缺字漏字、缺少标点、标点错误等字符层面的错误
2. **成分残缺型错误**：句子中缺少某些必要成分，如主语不明、谓语残缺、宾语残缺以及其他成分残缺等
3. **成分赘余型错误**：句子中有多余成分，如主语多余、虚词多余以及其他成分多余等
4. **成分搭配不当型错误**：句子成分之间搭配不当，如语序不当，动宾搭配不当以及其他搭配不当等

### 细粒度错误类型（14类）：
**字符级错误（4个子类）：**
1. **错别字错误**：句子中出现错别字（需要修改或删除）
2. **缺字漏字**：句子中缺少字（需要添加）
3. **缺少标点**：句子缺少必要标点，应该断句的地方没有使用标点把句子断开
4. **错用标点**：标点使用错误，如本来应该使用句号或分号却使用逗号

**成分残缺型错误（4个子类）：**
5. **主语不明**：句子缺少主语或主语不明确，修改时要增加主语或使主语显现
6. **谓语残缺**：句子缺少谓语，修改时要增加谓语
7. **宾语残缺**：句子缺少宾语，修改时要增加宾语
8. **其他成分残缺**：句子缺少其他必要成分，修改时要增加主语谓语宾语之外的其他情况

**成分赘余型错误（4个子类）：**
9. **主语多余**：句子主语重复或多余，修改时要删掉多余主语
10. **虚词多余**：句子中虚词使用多余，助词"的"、"所"的多余，修改时删掉虚词
11. **其他成分多余**：除主语、谓语、虚词之外的成分多余，修改时删掉多余成分

**成分搭配不当型错误（3个子类）：**
12. **语序不当**：句子中词语或子句的顺序不合理，修改时调换某几个词汇或子句的顺序
13. **动宾搭配不当**：谓语和宾语搭配不当，修改时要用其他词替换句子的谓语或宾语
14. **其他搭配不当**：其他成分搭配不当，除动宾、语序不当之外的其他搭配不当情况

## 专业判断原则
- **准确性原则**：如果句子语法正确、表达清晰、无任何错误，则CourseGrainedErrorType和FineGrainedErrorType都应该为空数组[]
- **完整性原则**：如果存在错误，请准确标注所有相关的错误类型，不要遗漏
- **层次性原则**：注意粗粒度和细粒度错误的对应关系，细粒度错误必须与相应的粗粒度错误对应
- **专业性原则**：基于语言学专业知识进行判断，避免主观臆断

## 预测示例

**示例1：字符级错误**
输入："这本书的内客很有趣。"
分析：「内客」应为「内容」，存在错别字
输出：{
  "sent": "这本书的内客很有趣。",
  "CourseGrainedErrorType": ["字符级错误"],
  "FineGrainedErrorType": ["错别字错误"]""" + ("""，
  "coarse_probabilities": {
    "字符级错误": 0.95,
    "成分残缺型错误": 0.05,
    "成分赘余型错误": 0.02,
    "成分搭配不当型错误": 0.03
  },
  "fine_probabilities": {
    "错别字错误": 0.95,
    "缺字漏字": 0.05,
    "缺少标点": 0.02,
    "错用标点": 0.02,
    "主语不明": 0.05,
    "谓语残缺": 0.02,
    "宾语残缺": 0.02,
    "其他成分残缺": 0.03,
    "主语多余": 0.02,
    "虚词多余": 0.02,
    "其他成分多余": 0.02,
    "语序不当": 0.03,
    "动宾搭配不当": 0.03,
    "其他搭配不当": 0.03
  }""" if include_probability else "") + """
}

**示例2：成分残缺型错误**
输入："作为一名英语课代表，对徐老师的认识可能比同学更清楚。"
分析：缺少主语，「对徐老师的认识」前应有主语「我」；「比同学更清楚」缺少比较对象的宾语「的认识」
输出：{
  "sent": "作为一名英语课代表，对徐老师的认识可能比同学更清楚。",
  "CourseGrainedErrorType": ["成分残缺型错误"],
  "FineGrainedErrorType": ["主语不明", "其他成分残缺"]""" + ("""，
  "coarse_probabilities": {
    "字符级错误": 0.05,
    "成分残缺型错误": 0.92,
    "成分赘余型错误": 0.03,
    "成分搭配不当型错误": 0.08
  },
  "fine_probabilities": {
    "错别字错误": 0.02,
    "缺字漏字": 0.05,
    "缺少标点": 0.03,
    "错用标点": 0.02,
    "主语不明": 0.88,
    "谓语残缺": 0.05,
    "宾语残缺": 0.05,
    "其他成分残缺": 0.75,
    "主语多余": 0.02,
    "虚词多余": 0.02,
    "其他成分多余": 0.03,
    "语序不当": 0.05,
    "动宾搭配不当": 0.03,
    "其他搭配不当": 0.05
  }""" if include_probability else "") + """
}

**示例3：正确句子**
输入："他跑得很快，像一阵风一样。"
分析：句子结构完整，语法正确，表达清晰，无任何错误
输出：{
  "sent": "他跑得很快，像一阵风一样。",
  "CourseGrainedErrorType": [],
  "FineGrainedErrorType": []""" + ("""，
  "coarse_probabilities": {
    "字符级错误": 0.02,
    "成分残缺型错误": 0.03,
    "成分赘余型错误": 0.02,
    "成分搭配不当型错误": 0.03
  },
  "fine_probabilities": {
    "错别字错误": 0.02,
    "缺字漏字": 0.02,
    "缺少标点": 0.01,
    "错用标点": 0.01,
    "主语不明": 0.02,
    "谓语残缺": 0.01,
    "宾语残缺": 0.01,
    "其他成分残缺": 0.02,
    "主语多余": 0.01,
    "虚词多余": 0.01,
    "其他成分多余": 0.02,
    "语序不当": 0.02,
    "动宾搭配不当": 0.02,
    "其他搭配不当": 0.02
  }""" if include_probability else "") + """
}

## 预测任务
请运用你的专业知识，对以下句子进行准确的病句类型预测：

"""
        
        # 添加待预测的句子
        for i, text in enumerate(texts, 1):
            base_prompt += f"{i}. \"{text}\"\n"
        
        base_prompt += f"""
## 输出要求
请返回JSON数组格式的预测结果，每个句子一个JSON对象，包含：
- **sent**: 原句子
- **CourseGrainedErrorType**: 粗粒度错误类型数组（如果是正确句子则为空数组[]）
- **FineGrainedErrorType**: 细粒度错误类型数组（如果是正确句子则为空数组[]）"""

        if include_probability:
            base_prompt += """
- **coarse_probabilities**: 每个粗粒度错误类型的概率（0-1之间的浮点数）
- **fine_probabilities**: 每个细粒度错误类型的概率（0-1之间的浮点数）"""

        base_prompt += """

## 重要提醒
1. **准确判断**：运用专业语言学知识，仔细分析每个句子的语法结构
2. **完整标注**：如果句子没有错误，CourseGrainedErrorType和FineGrainedErrorType必须为空数组[]
3. **格式规范**：只返回标准JSON数组，确保可以被Python json.loads()正确解析
4. **层次对应**：确保细粒度错误类型与对应的粗粒度错误类型匹配
5. **专业严谨**：基于语言学理论进行判断，保持高度的专业性和准确性

请开始分析："""
        
        return base_prompt
        
    def predict_single(self, text: str, include_probability: bool = True, 
                      max_retries: int = 3) -> Dict[str, Any]:
        """预测单个文本的标签"""
        
        prompt = self.create_prediction_prompt([text], include_probability)
        
        data = {
            "model": "qwen-plus-latest",
            "input": {
                "messages": [
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.1,
                "top_p": 0.8,
                "max_tokens": 4000
            }
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(self.url, headers=self.headers, json=data, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                if 'output' in result and 'text' in result['output']:
                    generated_text = result['output']['text']
                    
                    # 解析JSON响应
                    try:
                        start_idx = generated_text.find('[')
                        end_idx = generated_text.rfind(']') + 1
                        
                        if start_idx != -1 and end_idx != 0:
                            json_text = generated_text[start_idx:end_idx]
                            predictions = json.loads(json_text)
                            
                            if len(predictions) >= 1:
                                return predictions[0]  # 返回第一个预测结果
                        else:
                            print(f"未找到有效的JSON格式，尝试第 {attempt + 1} 次")
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(2)
                            continue
                        else:
                            print("原始响应:", generated_text[:500])
                            
                else:
                    print(f"API调用失败 (尝试 {attempt + 1}/{max_retries}):", result)
                    
            except requests.exceptions.RequestException as e:
                print(f"请求失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
            
            except Exception as e:
                print(f"处理出错 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
        
        # 如果所有尝试都失败，返回空结果
        print(f"所有尝试都失败，为文本返回空预测结果: {text[:30]}...")
        return {"sent": text, "CourseGrainedErrorType": [], "FineGrainedErrorType": []}
    
    def load_existing_results(self, output_file: str) -> Dict[str, Any]:
        """加载已有的预测结果"""
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ 加载已有结果：{len(data['predictions'])} 条记录")
                return data
            except Exception as e:
                print(f"⚠️ 加载已有结果失败: {e}，将重新开始")
                return {"predictions": [], "completed_indices": []}
        else:
            return {"predictions": [], "completed_indices": []}
    
    def save_single_result(self, output_file: str, prediction: Dict[str, Any], 
                          completed_index: int):
        """保存单个预测结果（增量保存）"""
        
        # 加载现有数据
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                data = {"predictions": [], "completed_indices": []}
        else:
            data = {"predictions": [], "completed_indices": []}
        
        # 添加新结果
        data["predictions"].append(prediction)
        data["completed_indices"].append(completed_index)
        data["total_completed"] = len(data["predictions"])
        data["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def predict_texts_incremental(self, texts: List[str], 
                                 include_probability: bool = True,
                                 output_file: str = "incremental_predictions.json",
                                 sleep_between_requests: float = 1.0) -> List[Dict[str, Any]]:
        """增量预测文本列表的标签（支持断点续传）"""
        
        print(f"🚀 开始增量预测 {len(texts)} 个文本")
        print(f"📁 输出文件: {output_file}")
        print(f"⏱️ 请求间隔: {sleep_between_requests}秒")
        
        # 加载已有结果
        existing_data = self.load_existing_results(output_file)
        completed_indices = set(existing_data.get("completed_indices", []))
        all_predictions = existing_data.get("predictions", [])
        
        # 创建需要预测的索引列表
        remaining_indices = [i for i in range(len(texts)) if i not in completed_indices]
        
        if not remaining_indices:
            print("✅ 所有文本都已预测完成！")
            return all_predictions
        
        print(f"📊 待处理文本: {len(remaining_indices)} 个")
        print(f"📊 已完成文本: {len(completed_indices)} 个")
        
        # 开始增量预测
        for idx in tqdm(remaining_indices, desc="增量预测进度"):
            text = texts[idx]
            
            print(f"\n🔍 预测第 {idx+1}/{len(texts)} 个文本: {text[:50]}...")
            
            # 预测单个文本
            prediction = self.predict_single(text, include_probability)
            
            # 添加序号
            prediction['sent_id'] = idx + 1
            prediction['original_index'] = idx
            
            # 立即保存结果
            self.save_single_result(output_file, prediction, idx)
            
            # 添加到内存中的结果列表
            all_predictions.append(prediction)
            
            print(f"✅ 已保存预测结果 ({len(all_predictions)}/{len(texts)})")
            
            # 请求间隔
            if idx < remaining_indices[-1]:  # 不是最后一个
                time.sleep(sleep_between_requests)
        
        print(f"\n🎉 增量预测完成！共处理 {len(all_predictions)} 个文本")
        print(f"📁 结果已保存到: {output_file}")
        
        return all_predictions
    
    def get_prediction_status(self, output_file: str, total_texts: int):
        """查看预测进度状态"""
        if not os.path.exists(output_file):
            print(f"❌ 文件 {output_file} 不存在")
            return
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            completed = len(data.get("predictions", []))
            progress = completed / total_texts * 100 if total_texts > 0 else 0
            
            print(f"\n📊 预测进度状态:")
            print(f"   总文本数: {total_texts}")
            print(f"   已完成: {completed}")
            print(f"   进度: {progress:.1f}%")
            print(f"   最后更新: {data.get('last_update', '未知')}")
            
            if completed < total_texts:
                print(f"   剩余: {total_texts - completed} 个")
            else:
                print(f"   ✅ 全部完成！")
                
        except Exception as e:
            print(f"❌ 读取状态失败: {e}")
    
    def resume_prediction(self, texts: List[str], output_file: str):
        """断点续传预测"""
        print(f"🔄 尝试从 {output_file} 恢复预测...")
        
        # 检查当前状态
        self.get_prediction_status(output_file, len(texts))
        
        # 继续预测
        return self.predict_texts_incremental(texts, output_file=output_file)

    def analyze_predictions(self, predictions: List[Dict[str, Any]]):
        """分析预测结果"""
        print(f"\n📈 预测结果分析:")
        print(f"   总预测数: {len(predictions)}")
        
        # 统计各类错误类型
        coarse_counts = {}
        fine_counts = {}
        
        for pred in predictions:
            for coarse_type in pred.get("CourseGrainedErrorType", []):
                coarse_counts[coarse_type] = coarse_counts.get(coarse_type, 0) + 1
            
            for fine_type in pred.get("FineGrainedErrorType", []):
                fine_counts[fine_type] = fine_counts.get(fine_type, 0) + 1
        
        print(f"\n粗粒度错误类型统计:")
        for error_type, count in coarse_counts.items():
            print(f"   {error_type}: {count}")
        
        print(f"\n细粒度错误类型统计:")
        for error_type, count in fine_counts.items():
            print(f"   {error_type}: {count}")


def load_texts_from_json(file_path: str):
    """从JSON文件加载文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取sent字段
    texts = [item['sent'] for item in data]
    print(f"✅ 从 {file_path} 加载了 {len(texts)} 个文本")
    return texts

def main_with_file():
    """从文件加载数据的主函数"""
    
    api_key = "sk-abd72eb342014765b8e401394e870ba6"
    predictor = QwenLabelPredictor(api_key)
    
    # 从JSON文件加载文本
    input_file = "/mnt/cfs/huangzhiwei/NLP-WED0910/qwen3-256b-data/data_predict_again.json"  # 你的输入文件
    output_file = "/mnt/cfs/huangzhiwei/NLP-WED0910/qwen3-256b-data/val_1.json"  # 输出文件
    
    # 加载数据
    texts = load_texts_from_json(input_file)
    
    # 检查当前状态
    predictor.get_prediction_status(output_file, len(texts))
    
    # 开始或恢复增量预测
    predictions = predictor.predict_texts_incremental(
        texts=texts,
        include_probability=True,
        output_file=output_file,
        sleep_between_requests=1.0  # 每次请求间隔1秒
    )
    
    # 分析结果
    predictor.analyze_predictions(predictions)
    
    print(f"\n✅ 增量预测完成！结果已保存到 {output_file}")

if __name__ == "__main__":
    main_with_file()