# LangGraph-RAG-Agent-with-Evaluation

一个基于 **LangGraph** 构建的工程化 RAG Agent 项目，集成了 **检索增强生成（RAG）+ 答案质量评估（Evaluation）+ 自动数据样本沉淀**，用于模拟真实业务场景中对大模型回答 **可控性、可靠性与可迭代性** 的要求。

> 本项目侧重 **Agent 工作流设计、数据标注与评估逻辑**，可作为 AI Agent / 数据标注 / Prompt & Agent 工程相关岗位的项目展示。

---

## ✨ 项目核心特性

* **LangGraph Agent 工作流**：

  * LLM → 工具调用（RAG）→ 回答生成 → 质量评估 → 路由输出
* **RAG（PDF 知识库）**：

  * 基于 PDF 文档构建向量数据库（Chroma）
  * 仅允许模型基于检索内容回答，避免编造
* **Answer Evaluator（答案质量评估器）**：

  * 使用独立 LLM 对回答进行二次评估
  * 判断是否：

    * 使用了检索内容（citation_used）
    * 回答核心问题（core_question_answered）
    * 出现幻觉（hallucination）
* **Evaluation Router（质量路由）**：

  * 不合格回答自动拒绝
  * 合格回答才对用户输出
* **自动样本沉淀（JSONL）**：

  * 每一次问答自动保存为结构化样本
  * 支持后续用于数据清洗、分析或模型微调
  * 样本分层

---

## 🧠 Agent 工作流说明

整体 Agent 执行流程如下：

```text
[User Question]
      ↓
    [LLM]
      ↓
是否需要工具？
      ↓
  ┌───────────┐
  │ Yes       │ No
  ▼           ▼
[Retriever] [Answer Evaluator]
  │               │
  └────→ [LLM]    │
          ↓       │
   [Answer Evaluator]
          ↓
   [Evaluation Router]
          ↓
        [End]
```

### 关键设计点

* **工具调用与回答生成解耦**
* **评估结果不直接作为回答，而是作为路由依据**
* **Evaluation 不污染对话 messages，仅更新状态**

---

## 📁 项目结构

```text
LangGraph-RAG-Agent-with-Evaluation/
│
├── run.py                     # 项目入口
├── .env                       # 环境变量（API Key / 模型配置）
├── data/
│   └── Stock_Market_Performance_2024.pdf
│
├── samples/
│   └── agent_samples.jsonl    # 自动生成的样本数据
│
├── src/
│   ├── config.py              # 项目配置（路径 / 模型 / 样本位置）
│   ├── vectorstore.py         # 向量库构建与加载
│   ├── tools.py               # RAG 检索工具
│   ├── nodes.py               # LangGraph 各节点逻辑（LLM / Evaluator）
│   └── graph.py               # Agent 图结构定义
│
└── README.md
```

---

## ⚙️ 环境配置

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 配置 `.env`

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

如需自定义路径：

```env
PDF_PATH=/absolute/path/to/pdf
SAMPLE_DIR=/absolute/path/to/samples
```

---

## 🚀 运行项目

```bash
python run.py
```

示例交互：

```text
请输入你的问题：2024 年涨势最好的股票有哪些？

=== 回答 ===
2024 年股市中，科技板块表现最为突出……
```

若问题与知识库无关（如天气）：

```text
⚠️ 本次回答未通过质量校验，原因如下：
- 未基于检索内容回答问题
```

---

## 🧪 样本数据格式（JSONL）

每一次交互都会生成一条样本：

```json
{
  "user_question": "最值得买的股票",
  "retrieved_docs": ["📄 文档 1: ..."],
  "final_answer": "2024 年科技股表现突出...",
  "evaluation": {
    "citation_used": true,
    "core_question_answered": "yes",
    "hallucination": false,
    "explanation": "答案基于文档内容"
  },
  "passed": true,
  "timestamp": "2025-12-17T16:57:25"
}
```

该数据可用于：

* Prompt 优化
* Agent 行为分析
* 构建训练 / 验证数据集

---

## 🎯 适用场景

* AI Agent / LangGraph 学习与实践
* RAG 系统质量控制
* 大模型数据标注与评估流程模拟
* 面试项目展示（Agent / 数据 / Prompt 工程方向）

---

## 🔮 后续可扩展方向

* 多维度评分（Score-based Evaluator）
* Gold / Silver / Reject 样本分级
* Prompt A/B 测试
* 多 Agent 协作（Planner / Critic）
* 样本统计与可视化分析

---

## 📌 作者说明

本项目为个人学习与工程实践项目，重点关注 **Agent 工作流设计与数据质量控制**，而非单一模型调用示例。

欢迎交流与讨论。
