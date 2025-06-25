# 中文病句检测项目

## 📖 项目简介

本项目致力于中文病句检测任务，采用多层次错误分类体系对中文文本进行病句识别与纠正。通过深度学习模型和集成学习方法，实现了对中文语法错误的准确识别和分类。

### 🎯 任务描述
- **粗粒度错误分类**：4类主要错误类型
- **细粒度错误分类**：14类详细错误子类型
- **当前性能**：宏观F1值达到 **58.21%**

## 📁 项目结构

```
NLP-WED0910/
├── datas/                           # 数据目录
│   ├── 训练集/                      # 原始训练数据
│   ├── 验证集/                      # 验证数据集
│   └── 数据增强后的数据集/          # 扩展训练数据
├── projects/                        # 模型代码目录
│   ├── moe+bge+qwen256_distillation.py  # 最佳性能模型
│   ├── best_hyperparameter/         # 最佳超参数配置
│   │   └── bge+qwen256.json        # 超参数文件
│   └── 字符级集成/                  # 集成学习代码
├── hzw-3.9+.yml                    # 环境配置文件
└── README.md                       # 项目说明文档
```

## 🛠️ 环境配置

### 系统要求
- Python 3.9+
- CUDA (推荐用于GPU加速)

### 安装步骤

1. **创建并激活conda环境**
```bash
# 创建环境
conda env create -f hzw-3.9+.yml

# 激活环境
conda activate [环境名称]
```

2. **验证安装**
```bash
python --version
# 确保输出 Python 3.9.x 或更高版本
```

## 📊 数据说明

### 数据集组成
- **训练数据**：位于 `/datas/训练集/` 目录
- **验证数据**：位于 `/datas/验证集/` 目录，用于模型性能评估
- **增强数据**：位于 `/datas/数据增强后的数据集/` 目录，通过数据增强技术生成的扩展训练集

### 错误类型体系
- **粗粒度分类**：4类主要语法错误类型
- **细粒度分类**：14类详细错误子类型

## 🚀 快速开始

### 训练模型
```bash
cd projects/
python moe+bge+qwen256_distillation.py
```

### 使用预训练模型
所有模型相关代码和运行脚本均位于 `/projects/` 目录下。

## 🏆 复现最佳性能

### 步骤1：配置最佳超参数
1. 查看最佳超参数文件：`best_hyperparameter/bge+qwen256.json`
2. 手动将其中的参数值填入到 `moe+bge+qwen256_distillation.py` 的相应位置

### 步骤2：基础模型训练
```bash
cd projects/
python moe+bge+qwen256_distillation.py
```

### 步骤3：集成学习
```bash
cd 字符级集成/
# 运行集成学习脚本，整合qwen3-235b的字符级错误结果
python ensemble_integration.py
```

### 关键组件
1. **核心模型**：`moe+bge+qwen256_distillation.py`
2. **最佳超参数存储**：`best_hyperparameter/bge+qwen256.json`（需手动填入代码）
3. **集成学习**：整合qwen3-235b字符级错误检测结果
4. **集成代码**：位于 `字符级集成/` 目录

## 📈 性能指标

### 最佳性能
- **Micro F1值（Final）**：**58.21%**
- **错误分类层次**：粗粒度(4类) + 细粒度(14类)

### 完整实验结果对比

<table align="center">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center; vertical-align: middle;">Model</th>
      <th colspan="3" style="text-align: center;">Micro F1 (%)</th>
      <th colspan="3" style="text-align: center;">Macro F1 (%)</th>
    </tr>
    <tr>
      <th style="text-align: center;">Final</th>
      <th style="text-align: center;">Coarse</th>
      <th style="text-align: center;">Fine</th>
      <th style="text-align: center;">Final</th>
      <th style="text-align: center;">Coarse</th>
      <th style="text-align: center;">Fine</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">BGE-Base</td>
      <td style="text-align: center;">44.43</td>
      <td style="text-align: center;">59.52</td>
      <td style="text-align: center;">29.33</td>
      <td style="text-align: center;">25.54</td>
      <td style="text-align: center;">43.21</td>
      <td style="text-align: center;">7.87</td>
    </tr>
    <tr>
      <td style="text-align: center;">BGE-Large</td>
      <td style="text-align: center;">50.32</td>
      <td style="text-align: center;">64.29</td>
      <td style="text-align: center;">36.36</td>
      <td style="text-align: center;">33.27</td>
      <td style="text-align: center;">44.52</td>
      <td style="text-align: center;">22.02</td>
    </tr>
    <tr>
      <td style="text-align: center;">WWM-Base</td>
      <td style="text-align: center;">43.49</td>
      <td style="text-align: center;">54.29</td>
      <td style="text-align: center;">32.69</td>
      <td style="text-align: center;">14.24</td>
      <td style="text-align: center;">20.65</td>
      <td style="text-align: center;">7.84</td>
    </tr>
    <tr>
      <td style="text-align: center;">WWM-Large</td>
      <td style="text-align: center;">48.30</td>
      <td style="text-align: center;">63.77</td>
      <td style="text-align: center;">32.84</td>
      <td style="text-align: center;">29.72</td>
      <td style="text-align: center;">40.30</td>
      <td style="text-align: center;">19.15</td>
    </tr>
    <tr>
      <td style="text-align: center;">Qwen3-235B-A22B</td>
      <td style="text-align: center;">30.86</td>
      <td style="text-align: center;">42.11</td>
      <td style="text-align: center;">19.61</td>
      <td style="text-align: center;">31.45</td>
      <td style="text-align: center;">43.22</td>
      <td style="text-align: center;">19.68</td>
    </tr>
    <tr style="background-color: #f0f9ff;">
      <td style="text-align: center; font-weight: bold;">Ours (本方法)</td>
      <td style="text-align: center; font-weight: bold; color: #1d4ed8;">58.21</td>
      <td style="text-align: center; font-weight: bold; color: #1d4ed8;">75.68</td>
      <td style="text-align: center; font-weight: bold; color: #1d4ed8;">40.74</td>
      <td style="text-align: center; font-weight: bold;">31.63</td>
      <td style="text-align: center; font-weight: bold;">41.60</td>
      <td style="text-align: center; font-weight: bold;">21.67</td>
    </tr>
  </tbody>
</table>

### 性能说明
- **Final**: 最终综合评价指标
- **Coarse**: 粗粒度分类性能（4类主要错误）
- **Fine**: 细粒度分类性能（14类详细错误子类型）
- 本方法在 **Micro F1** 各项指标上均取得最佳性能

## 🔧 模型架构

- **基础模型**：MOE + BGE + Qwen256 蒸馏架构
- **增强策略**：数据增强 + 知识蒸馏
- **集成方法**：字符级错误检测结果集成

## 📝 使用说明

### 基本用法
```python
# 导入模型
from projects.moe_bge_qwen256_distillation import ErrorDetectionModel

# 初始化模型（使用最佳超参数，需先从bge+qwen256.json中查看并手动设置）
model = ErrorDetectionModel()

# 检测病句
text = "这个问题很困难解决。"
result = model.detect(text)
print(result)
```

### 超参数配置说明
最佳超参数存储在 `best_hyperparameter/bge+qwen256.json` 文件中，使用时需要：
1. 打开该JSON文件查看参数值
2. 手动将参数填入到 `moe+bge+qwen256_distillation.py` 的相应位置
3. 保存代码后运行训练

**注意**：运行模型前请确保已正确配置环境并下载必要的预训练模型文件。
