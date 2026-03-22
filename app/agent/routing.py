"""Routing functions for the agent graph conditional edges."""

from app.agent.state import AgentState


def route_by_category(state: AgentState) -> str:
    """Route after classification based on ticket category.

    Returns the category string which maps to graph nodes via
    add_conditional_edges:
        "question"  -> rag_search
        "complaint" -> empathy_response
        "unclear"   -> clarify
    """
    return state["category"]


def route_after_rag(state: AgentState) -> str:
    """Route after RAG search based on escalation flag (AR-4).

    If needs_escalation is True (no relevant results or low similarity),
    skip generate + quality_check and go straight to escalation.

    Returns:
        "escalate" if escalation is needed, "generate_response" otherwise.
    """
    if state.get("needs_escalation", False):
        return "escalate"
    return "generate_response"
