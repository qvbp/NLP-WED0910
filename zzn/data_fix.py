import json
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain_openai import OpenAI
from tqdm import tqdm
from langchain_openai import ChatOpenAI  # 改这里

system_template = """\\no_think"""
user_template = """你是一名专业的语文老师，下面是一个病句，请你输出正确的句子。请注意，输出的句子必须是正确的中文语法，并且与原句的意思相同。

存在的粗粒度错误：{course_error}
存在的细粒度错误：{fine_grained_error}
原句：{original_sentence}
你只需要输出修正后的句子，不要输出其他内容（包括“修正后的句子：”等前缀和后缀）。
"""

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        
    return data


def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main():
    file_path = "/mnt/cfs/huangzhiwei/NLP-WED0910/datas/train.json"
    file_save_path = "/mnt/cfs/huangzhiwei/NLP-WED0910/zzn/train_fixed.json"
    data = read_json(file_path)
    print(f"Loaded {len(data)} items from {file_path}")
    
    # llm = OpenAI(
    #     base_url="https://api.siliconflow.cn/v1",
    #     api_key="sk-ytzrzjokeuswcvksiayajjawsmelqrytzopgqsgjacyjtqel",
    #     # model="Qwen/Qwen3-235B-A22B",
    #     model="Pro/deepseek-ai/DeepSeek-V3",
    #     temperature=0.1,
    #     max_tokens=256,
    # )
    # llm = OpenAI(
    #     base_url="http://localhost:1234/v1",
    #     api_key="none",
    #     model="qwen3-30b-a3b",
    #     # model="qwen3-8b",
    #     temperature=0.1,
    #     max_tokens=256,
    # )
    # llm = OpenAI(
    #     # base_url="http://qwen3-235b-a22b.bd-ai-llm.mlops-infer.tal.com/v1",
    #     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    #     # api_key="EMPTY",
    #     api_key="sk-abd72eb342014765b8e401394e870ba6",
    #     # model="Qwen3-235B-A22B",
    #     model="qwen-plus-latest",
    #     temperature=0.1,
    #     max_tokens=256,
    # )

    # 改为：
    llm = ChatOpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key="sk-abd72eb342014765b8e401394e870ba6",
        model="qwen-plus-latest",
        temperature=0.1,
        max_tokens=256,
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(user_template)
    ])
    
    chain = prompt | llm

    for item in tqdm(data):
        sent = item["sent"]
        response = chain.invoke(dict(
            course_error=item.get("CourseGrainedErrorType", []),
            fine_grained_error=item.get("FineGrainedErrorType", []),
            original_sentence=sent
        ))

        print(response)
        # item["fixed_sent"] = response.strip()
        item["fixed_sent"] = response.content.strip()
        
        
    write_json(data, file_save_path)


if __name__ == "__main__":
    main()
