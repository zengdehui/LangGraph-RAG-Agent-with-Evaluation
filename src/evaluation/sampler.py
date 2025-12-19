import json
from datetime import datetime
from pathlib import Path

from src.config import SAMPLE_PATH

def classify_sample(evaluation: dict) -> str:
    """给样本定义「业务级分层」 gold 样本可直接用于训练，silver 用于清洗优化，reject 用于分析失败模式。"""
    if (
        evaluation.get("citation_used") is True
        and evaluation.get("core_question_answered") == "yes"
        and evaluation.get("hallucination") is False
    ):
        return "gold"

    if (
        evaluation.get("citation_used") is True
        and evaluation.get("core_question_answered") in ["yes", "partial"]
        and evaluation.get("hallucination") is False
    ):
        return "silver"

    return "reject"


def save_sample(
    user_question: str,
    retrieved_docs: list[str],
    final_answer: str,
    evaluation: dict,
    grade :str,
    passed: bool,
    file_path: str = SAMPLE_PATH
):
    """
    保存RAG Agent的一次交互样本到JSONL文件
    """
    sample = {
        "user_question": user_question,
        "retrieved_docs": retrieved_docs,
        "final_answer": final_answer,
        "evaluation": evaluation,
        "grade": grade,
        "passed": passed,
        "timestamp": datetime.now().isoformat()
    }

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")