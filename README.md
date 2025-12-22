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
├── src/
│ ├── run.py # 项目入口
│ ├── agent/ # LangGraph Agent定义&  Answer Evaluator & Router逻辑
│ ├── rag/ # 向量库 & 检索逻辑
│ ├── evaluation/ # 样本
│ ├── state.py # 结构化状态
│ └── config.py # 极简配置（路径、参数）
├── data/
│ └── Stock_Market_2024.pdf # 示例文档
├── samples/
│ └── agent_samples.jsonl # 自动沉淀样本
├── .env # API Key / 路径配置
└── README.md

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

🧪 示例行为说明
用户问题	系统行为
今天天气如何	合理拒答（gold_refusal）
你叫什么名字	合理拒答
根据2024股票走势推断2025	保守拒答
那个股涨了	若文档支持 → 正确回答
🎯 适合哪些人阅读这个项目？

想深入理解 LangGraph Agent 工作流 的开发者

不满足于“RAG 能跑就行”的工程实践者

准备在面试中展示 系统设计 & 工程取舍能力 的候选人

📌 后续可扩展方向

引入「谨慎推断模式（Inference-on-Document）」

不同拒答类型细分（Policy / Scope / Safety）

基于样本数据进行 Reward Model / 微调

---

## 📌 作者说明

本项目为个人学习与工程实践项目，重点关注 **Agent 工作流设计与数据质量控制**，而非单一模型调用示例。

欢迎交流与讨论。
