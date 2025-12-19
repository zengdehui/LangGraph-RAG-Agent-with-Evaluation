import json
from datetime import datetime
from pathlib import Path

from src.config import SAMPLE_PATH


def save_sample(
    user_question: str,
    retrieved_docs: list[str],
    final_answer: str,
    evaluation: dict,
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
        "passed": passed,
        "timestamp": datetime.now().isoformat()
    }

    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")