# Qwen API 分词验证代码（支持JSON文件批量处理）
import requests
import json
import os
import time
from typing import Optional, List, Dict, Any
from datetime import datetime

class QwenTokenizerTest:
    def __init__(self, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"):
        """
        初始化Qwen API调用器
        
        Args:
            api_key: 你的API密钥
            base_url: API基础URL
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def test_tokenization_via_api(self, text: str, model: str = "qwen-turbo") -> dict:
        """
        通过API调用测试分词
        
        Args:
            text: 要测试的文本
            model: 模型名称
            
        Returns:
            包含测试结果的字典
        """
        # 构造请求，要求模型返回分词信息
        prompt = f"""请对以下文本进行分词处理，并返回以下信息：
1. 原始文本
2. 分词后的tokens列表
3. 对tokens进行decode后的文本

文本：{text}

请按照以下格式返回：
原始文本: [原始文本]
分词tokens: [token列表]
解码后文本: [解码文本]
"""
        
        payload = {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }
        
        try:
            response = requests.post(self.base_url, headers=self.headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            output_text = result["output"]["choices"][0]["message"]["content"]
            
            # 解析返回的结果
            lines = output_text.strip().split('\n')
            parsed_result = {}
            
            for line in lines:
                if line.startswith("原始文本:"):
                    parsed_result["original"] = line.split(":", 1)[1].strip()
                elif line.startswith("解码后文本:"):
                    parsed_result["decoded"] = line.split(":", 1)[1].strip()
            
            # 检查是否一致
            is_consistent = parsed_result.get("original", "").strip() == parsed_result.get("decoded", "").strip()
            
            return {
                "success": True,
                "original_text": text,
                "api_response": output_text,
                "parsed_result": parsed_result,
                "is_consistent": is_consistent,
                "error": None
            }
            
        except Exception as e:
            return {
                "success": False,
                "original_text": text,
                "api_response": None,
                "parsed_result": None,
                "is_consistent": False,
                "error": str(e)
            }
    
    def process_json_file(self, json_file_path: str, model: str = "qwen-plus-latest", max_samples: int = None) -> List[Dict[str, Any]]:
        """
        处理JSON文件中的句子进行分词测试
        
        Args:
            json_file_path: JSON文件路径
            model: 模型名称
            max_samples: 最大处理样本数，None表示处理全部
            
        Returns:
            测试结果列表
        """
        try:
            # 读取JSON文件
            print(f"📂 正在读取JSON文件: {json_file_path}")
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                raise ValueError("JSON文件应包含一个列表")
            
            print(f"📊 总共有 {len(data)} 条数据")
            
            # 限制处理数量
            if max_samples:
                data = data[:max_samples]
                print(f"🔢 限制处理前 {max_samples} 条数据")
            
            results = []
            failed_count = 0
            
            for i, item in enumerate(data, 1):
                if 'sent' not in item:
                    print(f"⚠️  第 {i} 条数据缺少'sent'字段，跳过")
                    continue
                
                sent_text = item['sent']
                sent_id = item.get('sent_id', f'unknown_{i}')
                
                print(f"\n🔄 处理第 {i}/{len(data)} 条 (ID: {sent_id})")
                print(f"📝 句子: {sent_text}")
                
                # 进行分词测试
                test_result = self.test_tokenization_via_api(sent_text, model)
                
                # 添加原始数据信息
                result = {
                    "sent_id": sent_id,
                    "original_item": item,
                    "tokenization_test": test_result,
                    "index": i
                }
                
                results.append(result)
                
                if test_result["success"]:
                    status = "✅ 通过" if test_result["is_consistent"] else "❌ 不一致"
                    print(f"结果: {status}")
                else:
                    failed_count += 1
                    print(f"❌ API调用失败: {test_result['error']}")
                
                # 避免API调用过快
                time.sleep(1)  # 等待1秒
            
            # 输出统计信息
            total = len(results)
            successful = sum(1 for r in results if r["tokenization_test"]["success"])
            consistent = sum(1 for r in results if r["tokenization_test"]["success"] and r["tokenization_test"]["is_consistent"])
            
            print(f"\n=== 处理完成统计 ===")
            print(f"📊 总处理数: {total}")
            print(f"✅ API成功数: {successful}")
            print(f"❌ API失败数: {failed_count}")
            print(f"🎯 一致性通过: {consistent}/{successful}")
            
            return results
            
        except Exception as e:
            print(f"❌ 处理JSON文件失败: {str(e)}")
            return []
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = None):
        """
        保存测试结果到JSON文件
        
        Args:
            results: 测试结果列表
            output_path: 输出文件路径，None则自动生成
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"qwen_tokenization_results_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")

def test_qwen_tokenization_from_json():
    """
    从JSON文件测试Qwen分词的主函数
    """
    # 你需要在这里填入你的API密钥
    API_KEY = "sk-abd72eb342014765b8e401394e870ba6"
    
    # JSON文件路径 - 修改为你的JSON文件路径
    JSON_FILE_PATH = "/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train.json"
    
    # 创建测试器
    tester = QwenTokenizerTest(API_KEY)
    
    print("=== Qwen API JSON文件分词一致性测试 ===\n")
    
    # 处理JSON文件
    results = tester.process_json_file(
        json_file_path=JSON_FILE_PATH,
        max_samples=10  # 限制处理前10条，可以修改或设为None处理全部
    )
    
    if results:
        # 保存结果
        tester.save_results(results)
        
        # 显示问题句子
        problematic_sentences = []
        for result in results:
            test_result = result["tokenization_test"]
            if test_result["success"] and not test_result["is_consistent"]:
                problematic_sentences.append({
                    "sent_id": result["sent_id"],
                    "original": test_result["original_text"],
                    "decoded": test_result.get("parsed_result", {}).get("decoded", ""),
                    "error_types": result["original_item"].get("CourseGrainedErrorType", []),
                    "fine_error_types": result["original_item"].get("FineGrainedErrorType", [])
                })
        
        if problematic_sentences:
            print(f"\n=== 发现 {len(problematic_sentences)} 个分词不一致的句子 ===")
            for item in problematic_sentences:
                print(f"ID {item['sent_id']}:")
                print(f"  原文: {item['original']}")
                print(f"  解码: {item['decoded']}")
                if item['error_types']:
                    print(f"  错误类型: {', '.join(item['error_types'])}")
                print()
        else:
            print("\n🎉 所有句子分词都一致！")

def test_qwen_tokenization():
    """
    测试Qwen分词的主函数（单句测试）
    """
    # 你需要在这里填入你的API密钥
    API_KEY = "your_api_key_here"
    
    # 创建测试器
    tester = QwenTokenizerTest(API_KEY)
    
    # 测试用例
    test_cases = [
        "你好，世界！",
        "这是一个测试句子。",
        "Hello, how are you today?",
        "混合语言测试：Hello世界123！",
        "特殊字符：@#$%^&*()",
    ]
    
    print("=== Qwen API 分词一致性测试 ===\n")
    
    for i, text in enumerate(test_cases, 1):
        print(f"测试 {i}: {text}")
        result = tester.test_tokenization_via_api(text)
        
        if result["success"]:
            print(f"✅ API调用成功")
            print(f"📝 API返回:\n{result['api_response']}")
            print(f"🔍 一致性检查: {'✅ 通过' if result['is_consistent'] else '❌ 失败'}")
        else:
            print(f"❌ API调用失败: {result['error']}")
        
        print("-" * 50)

if __name__ == "__main__":
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 单句测试")
    print("2. JSON文件批量测试")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        test_qwen_tokenization()
    elif choice == "2":
        test_qwen_tokenization_from_json()
    else:
        print("无效选择，默认执行单句测试")
        test_qwen_tokenization()