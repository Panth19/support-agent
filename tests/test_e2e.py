"""End-to-end tests for the compiled LangGraph agent (Block 4.1).

All LLM-calling nodes are mocked.  Tests verify that the graph routes
correctly through all 5 scenarios: question, complaint, unclear/clarify,
retry cycle, and early escalation (AR-4).
"""

from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.nodes.clarify import ClarifyResult
from app.agent.nodes.classify import ClassificationResult
from app.agent.nodes.empathy import EmpathyResult
from app.agent.nodes.quality_check import QualityCheckResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    """Create a minimal AgentState dict for testing."""
    base = {
        "ticket_text": "How do I return an item?",
        "messages": [],
        "category": "",
        "priority": "low",
        "language": "en",
        "retrieved_contexts": [],
        "sources": [],
        "draft_response": "",
        "quality_score": 0.0,
        "needs_escalation": False,
        "escalation_reason": "",
        "iteration_count": 0,
        "tone_instructions": "",
        "empathy_preamble": "",
    }
    base.update(overrides)
    return base


def _mock_source_node(text="Context from KB", score=0.9, metadata=None):
    """Create a mock LlamaIndex NodeWithScore."""
    node = MagicMock()
    node.text = text
    node.score = score
    node.metadata = metadata or {"file_name": "faq.md"}
    return node


def _setup_classify_mock(mock_cls, category="question", priority="low", language="en"):
    """Configure classify LLM mock to return a fixed classification."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = ClassificationResult(
        category=category,
        priority=priority,
        language=language,
        summary="Test summary",
    )
    mock_cls.return_value.with_structured_output.return_value = mock_structured


def _setup_empathy_mock(mock_cls):
    """Configure empathy LLM mock."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = EmpathyResult(
        tone_instructions="Use a warm, apologetic tone.",
        empathy_preamble="We understand your frustration and sincerely apologize.",
    )
    mock_cls.return_value.with_structured_output.return_value = mock_structured


def _setup_generate_mock(mock_cls, content="Here is how to return your item..."):
    """Configure generate LLM mock."""
    mock_response = MagicMock()
    mock_response.content = content
    mock_cls.return_value.invoke.return_value = mock_response


def _setup_quality_mock(mock_cls, score=0.85):
    """Configure quality_check LLM mock."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = QualityCheckResult(
        relevance=score,
        completeness=score,
        tone=score,
        groundedness=score,
        overall_score=score,
        reason="Test quality assessment.",
    )
    mock_cls.return_value.with_structured_output.return_value = mock_structured


def _setup_rag_mock(mock_get_qe, nodes=None):
    """Configure RAG query engine mock with good results."""
    if nodes is None:
        nodes = [_mock_source_node()]
    mock_response = MagicMock()
    mock_response.source_nodes = nodes
    mock_get_qe.return_value.query.return_value = mock_response


# ---------------------------------------------------------------------------
# Scenario 1: Question -> rag_search -> generate -> quality_check -> send_reply
# ---------------------------------------------------------------------------


class TestQuestionFlow:
    """Happy-path question: classify -> rag_search -> generate -> quality_check -> send_reply."""

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.rag_search.get_query_engine")
    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_question_reaches_send_reply(
        self, mock_classify, mock_rag, mock_generate, mock_quality
    ):
        _setup_classify_mock(mock_classify, category="question", priority="low")
        _setup_rag_mock(mock_rag)
        _setup_generate_mock(mock_generate, content="You can return within 14 days.")
        _setup_quality_mock(mock_quality, score=0.85)

        graph = build_graph()
        result = graph.invoke(_make_state(ticket_text="How do I return an item?"))

        assert result["category"] == "question"
        assert "Support Team" in result["draft_response"]
        assert result["needs_escalation"] is False
        assert result["quality_score"] >= 0.7
        assert result["iteration_count"] == 1
        assert result["tone_instructions"] == ""
        assert result["empathy_preamble"] == ""


# ---------------------------------------------------------------------------
# Scenario 2: Complaint -> empathy -> rag_search -> ... -> send_reply
# ---------------------------------------------------------------------------


class TestComplaintFlow:
    """Complaint path: classify -> empathy -> rag_search -> generate -> quality_check -> send_reply."""

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.rag_search.get_query_engine")
    @patch("app.agent.nodes.empathy.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_complaint_includes_empathy(
        self, mock_classify, mock_empathy, mock_rag, mock_generate, mock_quality
    ):
        _setup_classify_mock(mock_classify, category="complaint", priority="high")
        _setup_empathy_mock(mock_empathy)
        _setup_rag_mock(mock_rag)
        _setup_generate_mock(mock_generate, content="We will process your refund...")
        _setup_quality_mock(mock_quality, score=0.8)

        graph = build_graph()
        result = graph.invoke(
            _make_state(ticket_text="My order never arrived! I want a refund!")
        )

        assert result["category"] == "complaint"
        assert result["tone_instructions"] != ""
        assert result["empathy_preamble"] != ""
        assert "Support Team" in result["draft_response"]
        assert result["needs_escalation"] is False


# ---------------------------------------------------------------------------
# Scenario 3: Unclear -> clarify -> interrupt -> resume -> question path
# ---------------------------------------------------------------------------


class TestClarifyFlow:
    """Unclear ticket: classify(unclear) -> clarify -> interrupt -> resume -> classify(question) -> ... -> send_reply."""

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.rag_search.get_query_engine")
    @patch("app.agent.nodes.clarify.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_clarify_interrupt_and_resume(
        self, mock_classify, mock_clarify, mock_rag, mock_generate, mock_quality
    ):
        # Classify: first call "unclear", second call "question"
        classify_results = iter([
            ClassificationResult(
                category="unclear", priority="low", language="ru", summary="Unclear",
            ),
            ClassificationResult(
                category="question", priority="medium", language="ru", summary="Payment page",
            ),
        ])
        mock_classify_structured = MagicMock()
        mock_classify_structured.invoke.side_effect = lambda _: next(classify_results)
        mock_classify.return_value.with_structured_output.return_value = (
            mock_classify_structured
        )

        # Clarify mock
        mock_clarify_structured = MagicMock()
        mock_clarify_structured.invoke.return_value = ClarifyResult(
            clarifying_question="What exactly is not working?",
        )
        mock_clarify.return_value.with_structured_output.return_value = (
            mock_clarify_structured
        )

        _setup_rag_mock(mock_rag)
        _setup_generate_mock(mock_generate)
        _setup_quality_mock(mock_quality, score=0.8)

        checkpointer = MemorySaver()
        graph = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "test-e2e-clarify"}}

        # First invoke — should pause at interrupt
        graph.invoke(_make_state(ticket_text="Не работает!!!"), config)

        # Verify interrupted
        graph_state = graph.get_state(config)
        assert graph_state.tasks, "Graph should have pending tasks (interrupted)"
        assert len(graph_state.tasks[0].interrupts) > 0

        # Resume with clarification
        result = graph.invoke(
            Command(resume="The payment page does not load"),
            config,
        )

        assert result["category"] == "question"
        assert "Не работает!!!" in result["ticket_text"]
        assert "The payment page does not load" in result["ticket_text"]
        assert "Support Team" in result["draft_response"]


# ---------------------------------------------------------------------------
# Scenario 4: Retry cycle -> escalation after 2 attempts
# ---------------------------------------------------------------------------


class TestRetryCycle:
    """Low quality score twice: classify -> rag(1) -> gen -> qc(low) -> rag(2) -> gen -> qc(low) -> escalate."""

    @patch("app.agent.nodes.quality_check.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.generate.ChatGoogleGenerativeAI")
    @patch("app.agent.nodes.rag_search.get_query_engine")
    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_retry_then_escalate(
        self, mock_classify, mock_rag, mock_generate, mock_quality
    ):
        _setup_classify_mock(mock_classify, category="question")
        _setup_rag_mock(mock_rag)
        _setup_generate_mock(mock_generate)
        _setup_quality_mock(mock_quality, score=0.4)

        graph = build_graph()
        result = graph.invoke(_make_state(ticket_text="Some complex question"))

        assert result["needs_escalation"] is True
        assert "ESCALATION PACKAGE" in result["draft_response"]
        assert result["iteration_count"] == 2

        # quality_check should have been called twice
        quality_structured = (
            mock_quality.return_value.with_structured_output.return_value
        )
        assert quality_structured.invoke.call_count == 2


# ---------------------------------------------------------------------------
# Scenario 5: AR-4 early escalation (low RAG score)
# ---------------------------------------------------------------------------


class TestEarlyEscalation:
    """AR-4: low RAG similarity -> skip generate/quality -> escalate immediately."""

    @patch("app.agent.nodes.rag_search.get_query_engine")
    @patch("app.agent.nodes.classify.ChatGoogleGenerativeAI")
    def test_low_rag_score_escalates_immediately(self, mock_classify, mock_rag):
        _setup_classify_mock(mock_classify, category="question")

        # RAG returns low-score result -> triggers AR-4 escalation
        _setup_rag_mock(mock_rag, nodes=[_mock_source_node(score=0.1)])

        graph = build_graph()
        result = graph.invoke(_make_state(ticket_text="How to fly a spaceship?"))

        assert result["needs_escalation"] is True
        assert "ESCALATION PACKAGE" in result["draft_response"]
        # generate and quality_check were NOT mocked — if the graph
        # incorrectly routed through them, this test would fail with
        # a missing API key error.
