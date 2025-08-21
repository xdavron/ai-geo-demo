from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages
from langgraph.prebuilt.chat_agent_executor import RemainingSteps
from app.llm.main_agent.schemas import MainAgentResponse


class State(TypedDict):
    messages: Annotated[list, add_messages]
    lead_id: int
    active_agent: Optional[str]
    remaining_steps: RemainingSteps
    structured_response: Optional[MainAgentResponse]
