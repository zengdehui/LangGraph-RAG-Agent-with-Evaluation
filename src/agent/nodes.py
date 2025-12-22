import json
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI

from src.config import CONFIG
from src.rag.retriever import retriever_tool
from src.state import AgentState
from src.agent.prompts import system_prompt,evaluator_prompt

logger = logging.getLogger(__name__)

def model_call_node(state: AgentState) -> dict:
    """LLM调用函数"""
    messages = state["messages"]
    # 拼接系统提示词（只拼一次，避免重复）
    messages_with_sys = [SystemMessage(content=system_prompt)] + list(messages)

    # 初始化LLM（指定模型名，更稳定）
    llm = ChatOpenAI(
        model=CONFIG["OPENAI_MODEL"],
        temperature=CONFIG["LLM_TEMPERATURE"]
    ).bind_tools([retriever_tool])

    try:
        response = llm.invoke(messages_with_sys)
        logger.info(f"LLM生成响应完成")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"LLM调用失败：{str(e)}")
        return {"messages": [HumanMessage(content=f"模型调用失败：{str(e)}")]}

def take_action_node(state: AgentState) -> AgentState:
    """工具执行函数"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls
    results = []

    tools_dict = {retriever_tool.name: retriever_tool}

    for t in tool_calls:
        tool_name = t["name"]
        tool_query = t["args"].get("query", "").strip()
        logger.info(f"调用工具：{tool_name}，查询内容：{tool_query}")

        # 工具存在性校验
        if tool_name not in tools_dict:
            err_msg = f"无效工具：{tool_name}，仅支持工具：{list(tools_dict.keys())}"
            logger.error(err_msg)
            result = err_msg
        else:
            # 调用工具并捕获异常
            try:
                result = tools_dict[tool_name].invoke(tool_query)
                logger.info(f"工具{tool_name}返回结果长度：{len(result)}字符")
            except Exception as e:
                result = f"工具调用失败：{str(e)}"
                logger.error(f"工具{tool_name}调用异常：{str(e)}")

        # 封装工具返回消息
        results.append(ToolMessage(
            tool_call_id=t["id"],
            name=tool_name,
            content=result
        ))

    logger.info("所有工具调用执行完成")
    return {"messages": results}

def answer_evaluator_node(state: AgentState):
    """根据 LLM 生成的答案进行质量评估，并将结构化结果存入 state.evaluation_result。
    不修改 messages，仅更新 evaluation_result 字段。"""
    messages = state["messages"]
    # 提取必要信息
    user_question = ""
    final_answer = ""
    retrieved_contents = []

    for msg in messages:
        if isinstance(msg, HumanMessage) and not user_question:
            user_question = msg.content
        elif isinstance(msg, AIMessage):
            final_answer = msg.content
        elif isinstance(msg, ToolMessage):
            retrieved_contents.append(msg.content)
    eval_messages = [
        SystemMessage(content=evaluator_prompt),
        HumanMessage(content=f"用户问题：{user_question}"),
        HumanMessage(content=f"模型答案：{final_answer}"),
        HumanMessage(content=f"检索内容：{' || '.join(retrieved_contents)}")
    ]
    eval_llm = ChatOpenAI(
        model=CONFIG["OPENAI_MODEL"],
        temperature=CONFIG["LLM_TEMPERATURE"]
    )
    try:
        response = eval_llm.invoke(eval_messages)
        # 强制解析为 JSON
        eval_dict = json.loads(response.content.strip())
        logger.info(f"质量评估LLM生成响应完成：{eval_dict}")
        return {"evaluation_result": eval_dict}
    except Exception as e:
        # 评估失败时，返回一个“不合格”的默认结果（或 None）
        logger.warning(f"评估失败，使用默认拒绝策略: {e}")
        return {
            "evaluation_result": {
                "citation_used": False,
                "core_question_answered": "no",
                "hallucination": True,
                "explanation": "评估系统异常"
            }
        }