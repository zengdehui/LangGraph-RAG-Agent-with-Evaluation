from pathlib import Path
import os

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据文件路径
PDF_PATH = os.getenv(
    "PDF_PATH",
    str(PROJECT_ROOT / "data" / "Stock_Market_Performance_2024.pdf")
)

# 向量库持久化目录
PERSIST_DIR = os.getenv(
    "PERSIST_DIR",
    str(PROJECT_ROOT / "chroma_db")
)

# 样本数据
SAMPLE_DIR = os.getenv(
    "SAMPLE_DIR",
    str(PROJECT_ROOT / "samples")
)

SAMPLE_FILE = os.getenv(
    "SAMPLE_FILE",
    "agent_samples.jsonl"
)

SAMPLE_PATH = str(Path(SAMPLE_DIR) / SAMPLE_FILE)

# 从.env读取配置（没有则用默认值）
CONFIG = {
    "PDF_PATH": os.getenv("PDF_PATH", "Stock_Market_Performance_2024.pdf"),
    "EMBEDDING_MODEL": os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "CHUNK_SIZE": int(os.getenv("CHUNK_SIZE", 1000)),
    "CHUNK_OVERLAP": int(os.getenv("CHUNK_OVERLAP", 200)),
    "TOP_K": int(os.getenv("TOP_K", 5)),
    "COLLECTION_NAME": os.getenv("COLLECTION_NAME", "stock_market_2024"),
    "PERSIST_DIR": os.getenv("PERSIST_DIR", "./chroma_db"),
    "LLM_TEMPERATURE": float(os.getenv("LLM_TEMPERATURE", 0.0))
}
