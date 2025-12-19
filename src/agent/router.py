from langchain_core.messages import AIMessage

from src.evaluation.sampler import save_sample
from src.state import AgentState


def should_continue_router(state: AgentState) -> bool:
    """判断是否需要调用工具"""
    last_msg = state["messages"][-1]
    return hasattr(last_msg, "tool_calls") and len(last_msg.tool_calls) > 0

def evaluation_router(state: AgentState) -> dict:
    """ 根据 evaluation_result 决定是否放行原始回答。"""
    eval_result = state.get("evaluation_result")
    user_question = state["messages"][0].content
    retrieved_docs = [
        m.content for m in state["messages"]
        if m.__class__.__name__ == "ToolMessage"
    ]

    # 情况1: 没有评估结果 → 拒绝
    if eval_result is None:
        return {
            "messages": [AIMessage(content="系统无法验证答案可靠性，本次不予输出。")]
        }
    # 情况2: 评估不合格 → 拒绝并说明原因
    failed = (
            not eval_result.get("citation_used", False)
            or eval_result.get("core_question_answered") == "no"
            or eval_result.get("hallucination", False)
    )
    # 保存样本
    save_sample(
        user_question=user_question,
        retrieved_docs=retrieved_docs,
        final_answer=state["messages"][-1].content,
        evaluation=eval_result,
        passed=not failed # 评估不合格时为False failed 是“违规判断”，passed 是“业务可用性判断”passed 永远等于 not failed
    )
    if failed:
        explanation = eval_result.get("explanation", "未提供具体原因")
        return {
            "messages": [
                AIMessage(
                    content=(
                        "本次回答未通过质量校验，原因如下：\n"
                        f"- {explanation}\n\n"
                        "请尝试更具体的问题，或提供相关背景。"
                    )
                )
            ]
        }

    # 情况3: 评估合格 → 返回原始 AI 回答
    # 从 messages 中找最近的 AIMessage（现在更安全，因为知道它存在）
    original_answer = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            original_answer = msg
            break

    return {
        "messages": [
            AIMessage(content=original_answer.content if original_answer else "（无可用答案）")
        ]
    }