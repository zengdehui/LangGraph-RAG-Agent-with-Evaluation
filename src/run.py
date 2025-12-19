import logging

from langchain_core.messages import HumanMessage
from src.agent.graph import build_rag_agent

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
def main():
    """启动RAG Agent交互"""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("=== 2024股市问答助手启动 ===")
    rag_agent = build_rag_agent()

    while True:
        user_input = input("\n请输入你的问题（输入exit/quit退出）：").strip()
        # 退出逻辑
        if user_input.lower() in ["exit", "quit"]:
            logger.info("用户退出，助手关闭")
            break
        # 空输入处理
        if not user_input:
            logger.info("输入的问题为空！")
            continue

        # 调用Agent
        try:
            messages = [HumanMessage(content=user_input)]
            result = rag_agent.invoke({"messages": messages})
            # 输出回答
            print("\n=== 回答 ===")
            print(result["messages"][-1].content)
        except Exception as e:
            logger.error(f"Agent调用异常：{str(e)}")

if __name__ == "__main__":
    main()