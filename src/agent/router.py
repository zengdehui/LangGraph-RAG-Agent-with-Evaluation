from langchain_core.messages import AIMessage
from src.evaluation.sampler import save_sample, classify_sample
from src.state import AgentState

REFUSAL_MESSAGE = "【非文档问题】未在资料范围内"

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
    final_answer = state["messages"][-1].content.strip()
    is_refusal = final_answer == REFUSAL_MESSAGE
    if is_refusal:
        grade = "gold_refusal" #样本等级分层
        passed = True  # 是否通过（用于下游 / 统计）
    else:
        grade = classify_sample(eval_result)
        passed = grade in ["gold"]
    # 保存样本
    save_sample(
        user_question=user_question,
        retrieved_docs=retrieved_docs,
        final_answer=final_answer,
        evaluation=eval_result,
        grade = grade ,# 值分为：gold 样本可直接用于训练，silver 用于清洗优化，reject 用于分析失败模式。
        passed=passed
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
