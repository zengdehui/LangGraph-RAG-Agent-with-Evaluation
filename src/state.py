from typing import TypedDict, Annotated, Sequence, Optional
from operator import add as add_messages
from langchain_core.messages import BaseMessage



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # 新增字段：存储结构化的评估结果（None 表示未评估）
    evaluation_result: Optional[dict]  # 例如: {"citation_used": true, "core_question_answered": "yes", ...}
