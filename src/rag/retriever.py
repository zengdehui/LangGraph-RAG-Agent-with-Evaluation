import logging
import os
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

from src.config import CONFIG
from src.rag.vectorstore import build_or_load_vectorstore

logger = logging.getLogger(__name__)

@tool
def retriever_tool(query: Optional[str]) -> str:
    """
    从2024年股市表现PDF中检索相关信息。
    参数:
        query: 检索的问题（不能为空）
    返回:
        格式化后的相关文档内容，未找到则返回提示语
    """
    # 校验输入
    if not query or query.strip() == "":
        return "检索失败：查询内容不能为空，请输入有效问题！"

    retriever = init_retriever(build_or_load_vectorstore())
    try:
        docs = retriever.invoke(query.strip())
    except Exception as e:
        logger.error(f"检索工具调用失败：{str(e)}")
        return f"检索失败：{str(e)}"

    if not docs:
        return "未找到与该问题相关的信息。"

    # 优化排版，更易读
    result = []
    for i, doc in enumerate(docs, 1):  # 直接指定起始序号1，不用i+1
        result.append(f"文档 {i}:\n{doc.page_content}")
    return "\n\n".join(result)  # 空两行分隔，更清晰

def init_retriever(vectorstore: Chroma) -> BaseRetriever:
    """初始化检索器"""
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": CONFIG["TOP_K"]}
    )
    return retriever
