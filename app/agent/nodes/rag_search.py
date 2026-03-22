"""RAG search node — retrieves relevant knowledge base context for the ticket.

Implements AR-4 early escalation: if no relevant results are found
(best similarity score < 0.3 or no results at all), the node sets
needs_escalation=True to skip generate + quality_check and avoid hallucinations.
"""

from app.agent.state import AgentState
from app.rag.query_engine import get_query_engine

RAG_RELEVANCE_THRESHOLD = 0.3


def rag_search_node(state: AgentState) -> dict:
    """Query the knowledge base and return retrieved contexts and sources.

    Returns:
        dict with keys: retrieved_contexts, sources, iteration_count,
        needs_escalation, escalation_reason.
    """
    iteration_count = state.get("iteration_count", 0) + 1

    try:
        query_engine = get_query_engine()
        response = query_engine.query(state["ticket_text"])
    except Exception as exc:
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": f"RAG search failed: {exc}",
        }

    source_nodes = response.source_nodes

    if not source_nodes:
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": "No knowledge base results found for this query",
        }

    best_score = max(
        (node.score for node in source_nodes if node.score is not None),
        default=0.0,
    )

    if best_score < RAG_RELEVANCE_THRESHOLD:
        return {
            "retrieved_contexts": [],
            "sources": [],
            "iteration_count": iteration_count,
            "needs_escalation": True,
            "escalation_reason": (
                f"No relevant knowledge base articles found "
                f"(best score: {best_score:.2f})"
            ),
        }

    retrieved_contexts = [node.text for node in source_nodes]
    sources = [
        {
            "text": node.text,
            "score": node.score,
            "metadata": node.metadata,
        }
        for node in source_nodes
    ]

    return {
        "retrieved_contexts": retrieved_contexts,
        "sources": sources,
        "iteration_count": iteration_count,
        "needs_escalation": False,
        "escalation_reason": "",
    }
