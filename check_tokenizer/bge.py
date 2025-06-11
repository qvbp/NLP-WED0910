# 本地BERT模型分词验证代码（支持JSON文件批量处理）
from transformers import AutoTokenizer, BertTokenizer
import torch
import json
import os
from typing import List, Dict, Any
from datetime import datetime

class BertTokenizerTest:
    def __init__(self, model_path: str, tokenizer_type: str = "auto"):
        """
        初始化BERT Tokenizer测试器
        
        Args:
            model_path: 本地模型路径或Hugging Face模型名称
            tokenizer_type: tokenizer类型 ("auto", "bert")
        """
        self.model_path = model_path
        
        try:
            if tokenizer_type == "auto":
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            elif tokenizer_type == "bert":
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
            else:
                raise ValueError(f"不支持的tokenizer类型: {tokenizer_type}")
                
            print(f"✅ 成功加载tokenizer: {model_path}")
            print(f"📊 词汇表大小: {self.tokenizer.vocab_size}")
            
        except Exception as e:
            raise Exception(f"加载tokenizer失败: {str(e)}")
    
    def test_tokenization(self, text: str, verbose: bool = True) -> Dict[str, Any]:
        """
        测试分词和编解码过程
        
        Args:
            text: 要测试的文本
            verbose: 是否显示详细信息
            
        Returns:
            包含测试结果的字典
        """
        try:
            # 1. 分词 (tokenize)
            tokens = self.tokenizer.tokenize(text)
            
            # 2. 转换为token IDs (encode)
            token_ids = self.tokenizer.encode(text, add_special_tokens=True)
            
            # 3. 不添加特殊tokens的编码
            token_ids_no_special = self.tokenizer.encode(text, add_special_tokens=False)
            
            # 4. 解码 (decode)
            decoded_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            decoded_text_no_special = self.tokenizer.decode(token_ids_no_special, skip_special_tokens=False)
            
            # 5. 检查一致性
            is_consistent = text.strip() == decoded_text.strip()
            is_consistent_no_special = text.strip() == decoded_text_no_special.strip()
            
            # 6. 获取特殊tokens
            special_tokens = []
            if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token:
                special_tokens.append(f"CLS: {self.tokenizer.cls_token}")
            if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token:
                special_tokens.append(f"SEP: {self.tokenizer.sep_token}")
            if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token:
                special_tokens.append(f"PAD: {self.tokenizer.pad_token}")
            if hasattr(self.tokenizer, 'unk_token') and self.tokenizer.unk_token:
                special_tokens.append(f"UNK: {self.tokenizer.unk_token}")
            
            result = {
                "original_text": text,
                "tokens": tokens,
                "token_ids": token_ids,
                "token_ids_no_special": token_ids_no_special,
                "decoded_text": decoded_text,
                "decoded_text_no_special": decoded_text_no_special,
                "is_consistent": is_consistent,
                "is_consistent_no_special": is_consistent_no_special,
                "special_tokens": special_tokens,
                "vocab_size": self.tokenizer.vocab_size
            }
            
            if verbose:
                self._print_result(result)
            
            return result
            
        except Exception as e:
            error_result = {
                "original_text": text,
                "error": str(e),
                "success": False
            }
            if verbose:
                print(f"❌ 分词测试失败: {str(e)}")
            return error_result
    
    def _print_result(self, result: Dict[str, Any]):
        """打印测试结果"""
        print(f"📝 原始文本: {result['original_text']}")
        print(f"🔤 分词结果: {result['tokens']}")
        print(f"🔢 Token IDs (含特殊tokens): {result['token_ids']}")
        print(f"🔢 Token IDs (不含特殊tokens): {result['token_ids_no_special']}")
        print(f"🔄 解码文本 (含特殊tokens): '{result['decoded_text']}'")
        print(f"🔄 解码文本 (不含特殊tokens): '{result['decoded_text_no_special']}'")
        print(f"✅ 一致性检查 (含特殊tokens): {'通过' if result['is_consistent'] else '失败'}")
        print(f"✅ 一致性检查 (不含特殊tokens): {'通过' if result['is_consistent_no_special'] else '失败'}")
        print(f"🎯 特殊tokens: {', '.join(result['special_tokens'])}")
    
    def process_json_file(self, json_file_path: str, verbose: bool = False, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        处理JSON文件中的句子进行分词测试
        
        Args:
            json_file_path: JSON文件路径
            verbose: 是否显示详细信息
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
                
                if verbose or i % 50 == 0:  # 每50条显示一次进度
                    print(f"\n🔄 处理第 {i}/{len(data)} 条 (ID: {sent_id})")
                    if verbose:
                        print(f"📝 句子: {sent_text}")
                
                # 进行分词测试
                test_result = self.test_tokenization(sent_text, verbose=verbose)
                
                # 添加原始数据信息
                result = {
                    "sent_id": sent_id,
                    "original_item": item,
                    "tokenization_test": test_result,
                    "index": i
                }
                
                results.append(result)
                
                if 'error' in test_result:
                    failed_count += 1
                    if verbose:
                        print(f"❌ 分词测试失败")
                elif verbose:
                    consistent_no_special = test_result.get('is_consistent_no_special', False)
                    status = "✅ 通过" if consistent_no_special else "❌ 不一致"
                    print(f"结果: {status}")
            
            # 输出统计信息
            total = len(results)
            successful = sum(1 for r in results if 'error' not in r["tokenization_test"])
            consistent = sum(1 for r in results if 'error' not in r["tokenization_test"] and r["tokenization_test"].get("is_consistent_no_special", False))
            consistent_with_special = sum(1 for r in results if 'error' not in r["tokenization_test"] and r["tokenization_test"].get("is_consistent", False))
            
            print(f"\n=== 处理完成统计 ===")
            print(f"📊 总处理数: {total}")
            print(f"✅ 分词成功数: {successful}")
            print(f"❌ 分词失败数: {failed_count}")
            print(f"🎯 一致性通过 (不含特殊tokens): {consistent}/{successful}")
            print(f"🎯 一致性通过 (含特殊tokens): {consistent_with_special}/{successful}")
            
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
            output_path = f"bert_tokenization_results_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 结果已保存到: {output_path}")
        except Exception as e:
            print(f"❌ 保存结果失败: {str(e)}")
    
    def generate_summary_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        生成测试摘要报告
        
        Args:
            results: 测试结果列表
            
        Returns:
            摘要报告字典
        """
        total = len(results)
        if total == 0:
            return {"error": "没有测试结果"}
        
        successful = [r for r in results if 'error' not in r["tokenization_test"]]
        failed = [r for r in results if 'error' in r["tokenization_test"]]
        
        consistent_no_special = [r for r in successful if r["tokenization_test"].get("is_consistent_no_special", False)]
        inconsistent_no_special = [r for r in successful if not r["tokenization_test"].get("is_consistent_no_special", False)]
        
        # 分析不一致的句子
        problematic_sentences = []
        for result in inconsistent_no_special:
            test_result = result["tokenization_test"]
            problematic_sentences.append({
                "sent_id": result["sent_id"],
                "original": test_result["original_text"],
                "decoded": test_result["decoded_text_no_special"],
                "tokens": test_result["tokens"],
                "error_types": result["original_item"].get("CourseGrainedErrorType", []),
                "fine_error_types": result["original_item"].get("FineGrainedErrorType", [])
            })
        
        summary = {
            "total_samples": total,
            "successful_tokenization": len(successful),
            "failed_tokenization": len(failed),
            "consistent_samples": len(consistent_no_special),
            "inconsistent_samples": len(inconsistent_no_special),
            "consistency_rate": len(consistent_no_special) / len(successful) if successful else 0,
            "problematic_sentences": problematic_sentences[:10],  # 只显示前10个问题句子
            "problematic_count": len(problematic_sentences)
        }
        
        return summary
    
    def batch_test(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量测试多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            测试结果列表
        """
        results = []
        for i, text in enumerate(texts, 1):
            print(f"\n=== 测试 {i}/{len(texts)} ===")
            result = self.test_tokenization(text)
            results.append(result)
        return results

def test_bert_tokenization_from_json():
    """
    从JSON文件测试BERT分词的主函数
    """
    # JSON文件路径 - 修改为你的JSON文件路径
    JSON_FILE_PATH = "/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train.json"
    
    # 你可以修改这里的模型路径
    model_options = [
        # "bert-base-chinese",           # 中文BERT
        # "bert-base-uncased",          # 英文BERT
        # "hfl/chinese-bert-wwm-ext",   # 中文BERT WWM
        # 或者你的本地模型路径
        # "/path/to/your/local/bert/model"
        "/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5"
    ]
    
    # 选择一个模型进行测试
    model_path = model_options[0]  # 默认使用中文BERT
    
    try:
        # 创建测试器
        print(f"🚀 正在加载模型: {model_path}")
        tester = BertTokenizerTest(model_path)
        
        print(f"\n=== BERT JSON文件分词一致性测试 ===")
        print(f"🏷️  模型: {model_path}")
        print(f"📂 文件: {JSON_FILE_PATH}")
        
        # 处理JSON文件
        results = tester.process_json_file(
            json_file_path=JSON_FILE_PATH,
            verbose=False,  # 设为True可看详细信息
            max_samples=100  # 限制处理前100条，可以修改或设为None处理全部
        )
        
        if results:
            # 生成摘要报告
            summary = tester.generate_summary_report(results)
            
            print(f"\n=== 详细测试摘要 ===")
            print(f"📊 总样本数: {summary['total_samples']}")
            print(f"✅ 分词成功: {summary['successful_tokenization']}")
            print(f"❌ 分词失败: {summary['failed_tokenization']}")
            print(f"🎯 一致性通过: {summary['consistent_samples']}")
            print(f"⚠️  一致性失败: {summary['inconsistent_samples']}")
            print(f"📈 一致性成功率: {summary['consistency_rate']:.2%}")
            
            # 保存结果
            tester.save_results(results)
            
            # 显示问题句子
            if summary['problematic_sentences']:
                print(f"\n=== 前{len(summary['problematic_sentences'])}个分词不一致的句子 ===")
                for i, item in enumerate(summary['problematic_sentences'], 1):
                    print(f"\n{i}. ID {item['sent_id']}:")
                    print(f"   原文: {item['original']}")
                    print(f"   解码: {item['decoded']}")
                    print(f"   分词: {item['tokens']}")
                    if item['error_types']:
                        print(f"   错误类型: {', '.join(item['error_types'])}")
                
                if summary['problematic_count'] > len(summary['problematic_sentences']):
                    print(f"\n... 还有 {summary['problematic_count'] - len(summary['problematic_sentences'])} 个问题句子，详见保存的结果文件")
            else:
                print("\n🎉 所有句子分词都一致！")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print("💡 提示: 请检查模型路径和JSON文件路径是否正确")
        print("💡 安装命令: pip install transformers torch")
        return None

def test_bert_tokenization():
    """
    测试BERT分词的主函数（单句测试）
    """
    # 你可以修改这里的模型路径
    # 可以是本地路径，也可以是Hugging Face模型名称
    model_options = [
        # "bert-base-chinese",           # 中文BERT
        # "bert-base-uncased",          # 英文BERT
        # "hfl/chinese-bert-wwm-ext",   # 中文BERT WWM
        # 或者你的本地模型路径
        # "/path/to/your/local/bert/model"
        '/mnt/cfs/huangzhiwei/NLP-WED0910/projects/models/bge-large-zh-v1.5'
    ]
    
    # 选择一个模型进行测试
    model_path = model_options[0]  # 默认使用中文BERT
    
    try:
        # 创建测试器
        print(f"🚀 正在加载模型: {model_path}")
        tester = BertTokenizerTest(model_path)
        
        # 测试用例
        test_cases = [
            "你好，世界！",
            "这是一个测试句子。",
            "Hello, how are you today?",
            "混合语言测试：Hello世界123！",
            "特殊字符：@#$%^&*()",
            "很长的句子测试：人工智能技术在近年来取得了显著的进展，特别是在自然语言处理领域。",
            "[UNK]测试未知词汇",
        ]
        
        print(f"\n=== BERT本地分词一致性测试 ===")
        print(f"🏷️  模型: {model_path}")
        
        # 批量测试
        results = tester.batch_test(test_cases)
        
        # 统计结果
        total_tests = len(results)
        consistent_with_special = sum(1 for r in results if r.get('is_consistent', False))
        consistent_no_special = sum(1 for r in results if r.get('is_consistent_no_special', False))
        
        print(f"\n=== 测试总结 ===")
        print(f"📊 总测试数: {total_tests}")
        print(f"✅ 一致性通过 (含特殊tokens): {consistent_with_special}/{total_tests}")
        print(f"✅ 一致性通过 (不含特殊tokens): {consistent_no_special}/{total_tests}")
        
        return results
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print("💡 提示: 请检查模型路径是否正确，或确保已安装transformers库")
        print("💡 安装命令: pip install transformers torch")
        return None

# 额外的实用函数
def compare_tokenizers(text: str, model_paths: List[str]):
    """
    比较不同tokenizer的分词结果
    
    Args:
        text: 要比较的文本
        model_paths: 模型路径列表
    """
    print(f"🔍 比较不同tokenizer对文本的分词结果")
    print(f"📝 测试文本: {text}")
    print("=" * 60)
    
    for model_path in model_paths:
        try:
            print(f"\n🏷️  模型: {model_path}")
            tester = BertTokenizerTest(model_path)
            result = tester.test_tokenization(text, verbose=False)
            
            print(f"🔤 分词: {result['tokens']}")
            print(f"🔢 Token数量: {len(result['tokens'])}")
            print(f"✅ 一致性: {'通过' if result['is_consistent_no_special'] else '失败'}")
            
        except Exception as e:
            print(f"❌ {model_path}: 加载失败 - {str(e)}")

if __name__ == "__main__":
    # 选择测试模式
    print("请选择测试模式:")
    print("1. 单句测试")
    print("2. JSON文件批量测试")
    print("3. 比较不同模型")
    
    choice = input("请输入选择 (1, 2 或 3): ").strip()
    
    if choice == "1":
        test_bert_tokenization()
    elif choice == "2":
        test_bert_tokenization_from_json()
    elif choice == "3":
        # 比较不同模型的分词结果
        compare_tokenizers(
            "这是一个测试句子",
            ["bert-base-chinese", "hfl/chinese-bert-wwm-ext"]
        )
    else:
        print("无效选择，默认执行单句测试")
        test_bert_tokenization()