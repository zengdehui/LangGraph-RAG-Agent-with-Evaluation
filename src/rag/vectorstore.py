import logging
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

from src.config import PDF_PATH, PERSIST_DIR, CONFIG
from src.rag.loader import load_and_split_pdf

logger = logging.getLogger(__name__)

def build_or_load_vectorstore() -> Chroma:
    """构建向量库（如果已存在则直接加载，避免重复生成）"""
    embeddings = OpenAIEmbeddings(model=CONFIG["EMBEDDING_MODEL"])
    persist_dir = PERSIST_DIR
    collection_name = CONFIG["COLLECTION_NAME"]

    # 判断向量库是否已存在
    if os.path.exists(persist_dir) and len(os.listdir(persist_dir)) > 0:
        logger.info(f"向量库已存在，直接加载：{persist_dir}")
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    else:
        logger.info("向量库不存在，开始构建...")
        split_docs = load_and_split_pdf(PDF_PATH)
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        vectorstore.persist()  # 显式持久化，更稳妥
        logger.info("向量库构建并持久化完成")
    return vectorstore
