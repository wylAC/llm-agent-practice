# LLM Agent Practice

两个「最小可运行」的 LLM 智能体实践项目，用于跑通 **Function Calling** 与 **RAG（检索增强生成）** 的全流程。代码注释面向初学者，每个关键步骤都标注了对应的概念。

## 项目结构

```
├── fc_demo.py        # Function Calling demo：模型声明调用工具，程序执行
├── rag_demo.py       # RAG demo：字面检索(TF-IDF) + 语义检索(bge) 双模式
├── corpus/           # RAG 的语料（3 篇示例文档）
└── requirements.txt  # 依赖
```

## 环境准备

```bash
pip install -r requirements.txt
# 或手动安装：openai、numpy（RAG 必装）；torch、sentence-transformers（语义检索可选）
```

需要任一 OpenAI 兼容平台的 API key（生成部分用）：

```powershell
# Windows PowerShell
$env:DEEPSEEK_API_KEY="sk-你的key"
```

## 1. Function Calling demo

演示四步流程：程序把工具清单发给模型 → 模型输出 JSON 声明调用 → 程序执行真实函数 → 结果回填再生成。

```bash
python fc_demo.py --question "北京今天天气怎么样？"
python fc_demo.py --question "东京现在几点？"     # 模型会自动在 get_weather / get_time 中选择
python fc_demo.py --mock                        # 无 key 的本地模拟模式
```

关键理解：**模型只声明调用，不执行任何代码**；执行者是宿主程序。

## 2. RAG demo

全流程：文档切分 → 向量化 → 建索引 → 查询向量化 → top-k 检索 → 拼 prompt → LLM 生成。

```bash
python rag_demo.py --question "小明有哪些竞赛经历？"          # 完整 RAG（需 key）
python rag_demo.py --question "明年春节放假安排是什么？"           # 语料外问题 → 模型答「不知道」
python rag_demo.py --question "介绍一下小明的获奖情况" --compare  # 字面 vs 语义检索对比
python rag_demo.py --mode tfidf  # 强制字面检索
python rag_demo.py --mode embed  # 强制语义检索（需 torch + sentence-transformers）
```

### 字面检索 vs 语义检索（真实对比）

问题「介绍一下小明的获奖情况」（语料中无「获奖」二字）：

| 检索方式 | 第一名分数 | 表现 |
| --- | --- | --- |
| TF-IDF 字面检索 | 0.099 | 仅靠专有名词「小明」命中，同义改写抓瞎 |
| bge 语义检索 | 0.684 | 理解「获奖情况」≈「竞赛奖项」，稳定命中 |

结论：专有名词字面匹配强、同义改写语义强，**两者互补 → 混合检索**。

### 技术点

- TF-IDF 手写实现（中文 bigram），向量相似度用余弦
- bge-small-zh-v1.5 语义向量（首次运行自动从镜像下载）
- prompt 约束忠实性：语料外问题回答「不知道」而非编造

## 免责声明

本项目为学习实践用途，语料为示例文本。
