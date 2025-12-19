import logging
from src.agent.nodes import model_call_node, take_action_node, answer_evaluator_node
from src.agent.router import evaluation_router, should_continue_router
from src.state import AgentState
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)

def build_rag_agent() -> StateGraph:
    """构建并编译RAG Agent"""
    graph = StateGraph(AgentState)
    # 添加节点
    graph.add_node("llm", model_call_node)
    graph.add_node("retriever_agent", take_action_node)
    graph.add_node("answer_evaluator_agent", answer_evaluator_node)
    graph.add_node("evaluation_router", evaluation_router)
    # 添加条件边
    graph.add_conditional_edges(
        "llm",
        should_continue_router,
        {True: "retriever_agent", False: "answer_evaluator_agent"}
    )
    # 添加工具执行后回到LLM的边
    graph.add_edge("retriever_agent", "llm")
    # 添加评估器到LLM的边
    graph.add_edge("answer_evaluator_agent","evaluation_router")
    # 添加评估路由到最终结束点
    graph.add_edge("evaluation_router",END)
    # 设置入口点
    graph.set_entry_point("llm")
    # 编译
    rag_agent = graph.compile()
    logger.info("RAG Agent构建完成")
    return rag_agent
