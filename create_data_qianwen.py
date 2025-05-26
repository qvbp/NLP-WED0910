import json
import requests
import time
import random

def generate_sentence_errors(api_key, num_samples=200):
    """
    使用千问API生成病句识别数据
    """
    
    # 定义错误类型轮换策略
    error_type_batches = [
        {
            "focus": "字符级错误",
            "specific_types": ["错别字错误", "缺字漏字", "缺少标点", "错用标点"],
            "note": "重点生成字符层面的错误，每个句子可能有1-3种错误类型"
        },
        {
            "focus": "成分残缺型错误", 
            "specific_types": ["主语不明", "谓语残缺", "宾语残缺", "其他成分残缺"],
            "note": "重点生成句子成分缺失的错误，可能与字符级错误组合"
        },
        {
            "focus": "成分赘余型错误",
            "specific_types": ["主语多余", "谓语多余", "虚词多余", "其他成分多余"],
            "note": "重点生成句子成分多余的错误，有的句子只有1种，有的可能有2-3种"
        },
        {
            "focus": "成分搭配不当型错误",
            "specific_types": ["动宾搭配不当", "其他搭配不当", "语序不当"],
            "note": "重点生成搭配和语序错误，可能与其他类型组合"
        },
        {
            "focus": "混合错误类型",
            "specific_types": ["多种粗粒度错误组合", "复杂细粒度错误组合"],
            "note": "生成复杂的多错误类型组合，模拟真实学生作文中的多重错误"
        }
    ]
    
    # Few-shot prompt模板
    few_shot_prompt = """
## 任务定义
中小学作文病句类型识别是一个多标签分类问题，预测一条句子是哪些类型的病句。病句类型标签词时包含词法、句法、语义错误，本次识别任务共定义了个粗粒度错误类型和14个细粒度错误类型。

## 标签说明
### 粗粒度错误类型：
1. 字符级错误：包括错别字、缺字漏字、缺少标点、标点错误等字符层面的错误
2. 成分残缺型错误：句子中缺少某些必要成分，如主语不明、谓语残缺、宾语残缺以及其他成分残缺等
3. 成分搭配不当型错误：句子成分之间搭配不当，如语序不当，动宾搭配不当以及其他搭配不当等
4. 成分赘余型错误：句子中有多余成分，如主语多余，虚词多余以及其他成分多余等

### 细粒度错误类型：
1. 错别字错误：句子中出现错别字（需要修改或删除）
2. 缺字漏字：句子中缺少字（需要添加）
3. 缺少标点：句子缺少必要标点，应该断句的地方没有使用标点把句子断开
4. 错用标点：标点使用错误，如本来因应使用句号或分号却使用逗号
5. 主语不明：句子缺少主语或主语不明确，修改是要增加主语或使主语显现
6. 谓语残缺：句子缺少谓语，修改是要增加谓语
7. 宾语残缺：句子缺少宾语，修改是要增加宾语
8. 其他成分残缺：句子缺少其他必要成分，修改是要增加主语谓语宾语之外的其他情况
9. 主语多余：句子主语重复或多余，一般是句子较长，前一个主语说出后，紧接着有一个较长、较复杂的状语，后面又接了指向同一个事物的主语；修改是要删掉主语
11. 虚词多余：句子中虚词使用多余，助词”的“、”所“的多余，需改是删掉虚词
12. 其他成分多余：除主语、虚词之外的成分多余，修改是删掉除主语、虚词之外的词
13. 语序不当：句子中词语或子句的顺序不合理，修改是调换某几个词汇或子句的顺序
14. 动宾搭配不当：谓语和宾语搭配不当，修改是要用其他词替换句子的谓语或宾语
15. 其他搭配不当：其他成分搭配不当，除动宾、语序不当之外的其他搭配不当情况，修改是要用其他词替换句中的某个成分

## 数据示例

示例1 - 成分残缺型错误：
输入句子："作为一名英语课代表，对徐老师的认识可能比同学更清楚。"
输出：{
  "sent_id": 5059,
  "sent": "作为一名英语课代表，对徐老师的认识可能比同学更清楚。",
  "CourseGrainedErrorType": ["成分残缺型错误"],
  "FineGrainedErrorType": ["主语不明", "其他成分残缺"]
}

示例2 - 成分搭配不当型错误：
输入句子："因而，保存良好的家风，摒弃有害而无益的家风，是有助于人成长的一大益事。"
输出：{
  "sent_id": 254,
  "sent": "因而，保存良好的家风，摒弃有害而无益的家风，是有助于人成长的一大益事。",
  "CourseGrainedErrorType": ["成分搭配不当型错误"],
  "FineGrainedErrorType": ["动宾搭配不当"]
}

示例3 - 多种错误类型组合：
输入句子："早上阳光明媚，如我的心灵，丝毫不被外界的阴云所引响，只觉得蔚蓝的天空上漂浮着的云，星星点点素雅的小花，让人倍感舒爽到了中午，远处传来一丝丝的锣鼓声。"
输出：{
  "sent_id": 1004,
  "sent": "早上阳光明媚，如我的心灵，丝毫不被外界的阴云所引响，只觉得蔚蓝的天空上漂浮着的云，星星点点素雅的小花，让人倍感舒爽到了中午，远处传来一丝丝的锣鼓声。",
  "CourseGrainedErrorType": ["字符级错误", "成分残缺型错误"],
  "FineGrainedErrorType": ["缺少标点", "主语不明", "错别字错误"]
}

示例4 - 单一错误类型：
输入句子："这本书的内客很有趣。"
输出：{
  "sent_id": 1005,
  "sent": "这本书的内客很有趣。",
  "CourseGrainedErrorType": ["字符级错误"],
  "FineGrainedErrorType": ["错别字错误"]
}

示例5 - 复杂错误组合：
输入句子："狗狗事件不值一提了，但我奶奶您辛苦了，我以念念不忘的这件事来以此激励自己坚持走下去不让您老人家失望。"
输出：{
  "sent_id": 1006,
  "sent": "狗狗事件不值一提了，但我奶奶您辛苦了，我以念念不忘的这件事来以此激励自己坚持走下去不让您老人家失望。",
  "CourseGrainedErrorType": ["字符级错误", "成分赘余型错误"],
  "FineGrainedErrorType": ["缺少标点", "主语多余", "其他成分多余"]
}

## 要求
请生成5个新的中小学作文病句数据，要求：
1. 句子内容应该是中小学生写作风格
2. 每个句子都应该包含明确的语法错误
3. **重要：每个句子的错误类型数量应该多样化**：
   - 有的句子可能只有1种粗粒度错误 + 1种细粒度错误
   - 有的句子可能有2-3种粗粒度错误 + 2-4种细粒度错误
   - 不要固定生成相同数量的错误类型
4. **错误类型必须多样化**，尽量涵盖不同的粗粒度和细粒度错误，避免重复上述示例中的错误类型组合
5. 包含以下未出现的错误类型：缺字漏字、虚词多余、谓语残缺、宾语残缺、谓语多余、语序不当、其他搭配不当等
6. 返回标准JSON格式，包含sent_id, sent, CourseGrainedErrorType, FineGrainedErrorType字段
7. sent_id使用随机数字
8. **每批生成的数据不能重复相同的错误类型组合**
9. **CourseGrainedErrorType和FineGrainedErrorType都是数组，可能包含1-多个元素**

请直接返回JSON数组格式的数据，不需要其他解释。
"""

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    # url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    all_generated_data = []
    batch_size = 5  # 每次生成5条数据
    num_batches = (num_samples + batch_size - 1) // batch_size  # 向上取整
    
    print(f"准备生成 {num_samples} 条数据，分 {num_batches} 批次进行...")
    
    for batch in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_generated_data))
        
        # 轮换错误类型重点
        current_focus = error_type_batches[batch % len(error_type_batches)]
        
        # 根据当前批次调整prompt
        focused_prompt = few_shot_prompt.replace(
            "请生成5个新的中小学作文病句数据",
            f"请生成{current_batch_size}个新的中小学作文病句数据，本批次重点生成【{current_focus['focus']}】类型的错误。\n"
            f"注意：{current_focus['note']}\n"
            f"细粒度错误类型优先选择：{', '.join(current_focus['specific_types'][:4])}\n"
            f"**关键要求：每个句子的错误类型数量要多样化，有些句子1种错误，有些句子2-4种错误组合**"
        )
        
        data = {
            "model": "qwen-max",  # 或者使用 "qwen-plus", "qwen-max"
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": focused_prompt
                    }
                ]
            },
            "parameters": {
                "temperature": 0.9,  # 提高随机性
                "top_p": 0.95,
                "max_tokens": 2000
            }
        }
        
        try:
            print(f"正在生成第 {batch + 1}/{num_batches} 批数据...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if 'output' in result and 'text' in result['output']:
                generated_text = result['output']['text']
                
                # 尝试解析JSON
                try:
                    # 清理文本，提取JSON部分
                    start_idx = generated_text.find('[')
                    end_idx = generated_text.rfind(']') + 1
                    
                    if start_idx != -1 and end_idx != 0:
                        json_text = generated_text[start_idx:end_idx]
                        batch_data = json.loads(json_text)
                        
                        # 确保sent_id是唯一的
                        for item in batch_data:
                            item['sent_id'] = random.randint(10000, 99999)
                        
                        all_generated_data.extend(batch_data)
                        print(f"成功生成 {len(batch_data)} 条数据")
                    else:
                        print(f"第 {batch + 1} 批数据格式不正确，跳过")
                        
                except json.JSONDecodeError as e:
                    print(f"第 {batch + 1} 批JSON解析失败: {e}")
                    print("原始响应:", generated_text[:200] + "...")
                    
            else:
                print(f"第 {batch + 1} 批API调用失败:", result)
                
        except requests.exceptions.RequestException as e:
            print(f"第 {batch + 1} 批请求失败: {e}")
            
        except Exception as e:
            print(f"第 {batch + 1} 批处理出错: {e}")
        
        # 添加延迟避免请求过于频繁
        if batch < num_batches - 1:  # 最后一批不需要等待
            time.sleep(2)
    
    print(f"总共成功生成 {len(all_generated_data)} 条数据")
    return all_generated_data

def analyze_data_diversity(data):
    """分析生成数据的多样性"""
    from collections import Counter
    
    coarse_types = []
    fine_types = []
    error_type_counts = []
    
    for item in data:
        coarse_list = item.get('CourseGrainedErrorType', [])
        fine_list = item.get('FineGrainedErrorType', [])
        
        coarse_types.extend(coarse_list)
        fine_types.extend(fine_list)
        
        # 统计每个句子的错误类型数量
        error_type_counts.append({
            'coarse_count': len(coarse_list),
            'fine_count': len(fine_list),
            'total_count': len(coarse_list) + len(fine_list)
        })
    
    print("\n=== 数据多样性分析 ===")
    print(f"总计生成 {len(data)} 条数据")
    
    # 分析错误类型数量分布
    coarse_count_dist = Counter([item['coarse_count'] for item in error_type_counts])
    fine_count_dist = Counter([item['fine_count'] for item in error_type_counts])
    
    print(f"\n粗粒度错误类型数量分布:")
    for count, freq in sorted(coarse_count_dist.items()):
        print(f"  {count}种错误: {freq} 个句子")
    
    print(f"\n细粒度错误类型数量分布:")
    for count, freq in sorted(fine_count_dist.items()):
        print(f"  {count}种错误: {freq} 个句子")
    
    print("\n粗粒度错误类型分布:")
    coarse_counter = Counter(coarse_types)
    for error_type, count in coarse_counter.most_common():
        print(f"  {error_type}: {count} 次")
    
    print(f"\n细粒度错误类型分布:")
    fine_counter = Counter(fine_types)
    for error_type, count in fine_counter.most_common():
        print(f"  {error_type}: {count} 次")
    
    print(f"\n覆盖的细粒度错误类型数: {len(fine_counter)}/15")
    
    # 检查未覆盖的错误类型
    all_fine_types = {
        "错别字错误", "缺字漏字", "缺少标点", "错用标点", "主语不明", 
        "谓语残缺", "宾语残缺", "其他成分残缺", "主语多余", "谓语多余",
        "虚词多余", "其他成分多余", "语序不当", "动宾搭配不当", "其他搭配不当"
    }
    
    missing_types = all_fine_types - set(fine_counter.keys())
    if missing_types:
        print(f"\n未覆盖的错误类型: {', '.join(missing_types)}")
    else:
        print(f"\n✅ 已覆盖所有细粒度错误类型！")
    
    # 检查数量多样性
    if len(coarse_count_dist) >= 3 and len(fine_count_dist) >= 3:
        print(f"\n✅ 错误类型数量分布多样化！")
    else:
        print(f"\n⚠️  错误类型数量分布可能不够多样化")

def save_data(data, filename="generated_sentence_errors_strong.json"):
    """保存生成的数据到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已保存到 {filename}")

def main():
    # 替换为你的千问API密钥
    api_key = "sk-abd72eb342014765b8e401394e870ba6"
    
    # 生成200条数据
    generated_data = generate_sentence_errors(api_key, num_samples=200)
    
    # 保存数据
    if generated_data:
        save_data(generated_data)
        
        # 分析数据多样性
        analyze_data_diversity(generated_data)
        
        # 显示部分示例数据
        print("\n生成的数据示例:")
        for i, item in enumerate(generated_data[:3]):
            print(f"{i+1}. {json.dumps(item, ensure_ascii=False, indent=2)}")
    
    return generated_data

if __name__ == "__main__":
    main()