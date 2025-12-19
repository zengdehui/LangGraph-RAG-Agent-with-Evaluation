import logging
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def load_and_split_pdf(pdf_path: str) -> list:
    """加载PDF并智能分块（带异常处理）"""
    # 校验文件存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在：{pdf_path}")

    # 加载PDF
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        logger.info(f"成功加载PDF，共{len(pages)}页")
    except Exception as e:
        raise RuntimeError(f"加载PDF失败：{str(e)}")

    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200))
    )
    split_docs = text_splitter.split_documents(pages)
    logger.info(f"成功分块，共生成{len(split_docs)}个文档块")
    return split_docs