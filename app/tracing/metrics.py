"""Langfuse metrics — trace retrieval and aggregate statistics.

Queries the Langfuse API for individual traces and computes aggregate
metrics (latency, cost). Domain data (category, escalation) is stored
as Langfuse scores and available per-trace via get_trace_by_id().
"""

import logging
from datetime import datetime

from langfuse.api.core.request_options import RequestOptions

from app.tracing.langfuse_setup import get_langfuse_client

logger = logging.getLogger(__name__)

_REQ_OPTS = RequestOptions(timeout_in_seconds=15)


def _list_trace_records(client, list_kwargs: dict) -> list:
    """Return a list of normalized trace records across Langfuse SDK versions.

    v3/v4 expose ``client.api.trace.list``. v5 removed it and traces must be
    derived from ``client.api.observations.get_many``.
    """
    # --- Langfuse v3/v4 path ---
    trace_api = getattr(client.api, "trace", None)
    if trace_api is not None:
        for method_name in ("list", "list_traces"):
            method = getattr(trace_api, method_name, None)
            if callable(method):
                response = method(**list_kwargs, request_options=_REQ_OPTS)
                return [_normalize_record(t) for t in (getattr(response, "data", None) or [])]

    # --- Langfuse v5 path (observations) ---
    obs_api = getattr(client.api, "observations", None)
    if obs_api is not None:
        method = getattr(obs_api, "get_many", None)
        if callable(method):
            obs_kwargs = {"limit": list_kwargs.get("limit", 100)}
            if list_kwargs.get("from_timestamp"):
                obs_kwargs["from_start_time"] = list_kwargs["from_timestamp"]
            if list_kwargs.get("to_timestamp"):
                obs_kwargs["to_start_time"] = list_kwargs["to_timestamp"]
            response = method(**obs_kwargs, request_options=_REQ_OPTS)
            return [_normalize_record(o) for o in (getattr(response, "data", None) or [])]

    raise AttributeError("No supported trace listing method found in Langfuse SDK")


def _normalize_record(obj) -> dict:
    """Extract latency/cost from a trace (v4) or observation (v5) object."""
    latency = getattr(obj, "latency", None)
    if latency is None:
        latency = getattr(obj, "calculated_latency", None) or 0
    cost = None
    for attr in ("total_cost", "calculated_total_cost", "cost"):
        cost = getattr(obj, attr, None)
        if cost is not None:
            break
    return {"latency": latency or 0, "cost": cost or 0}


def _scores_data(response) -> list:
    """Extract the list of score objects from a scores response (v3/v4/v5)."""
    return getattr(response, "data", None) or []


def _fetch_scores(client, score_kwargs: dict):
    """Fetch scores across Langfuse SDK versions (v3/v4/v5)."""
    scores_api = getattr(client.api, "scores", None)
    if scores_api is not None:
        for method_name in ("get_many", "list", "list_scores"):
            method = getattr(scores_api, method_name, None)
            if callable(method):
                return method(**score_kwargs, request_options=_REQ_OPTS)
    raise AttributeError("No supported scores listing method found in Langfuse SDK")


def get_trace_by_id(trace_id: str) -> dict | None:
    """Fetch a single trace by ID and return a structured dict.

    Returns None if the trace is not found or Langfuse is unreachable.
    """
    try:
        client = get_langfuse_client()
        if client is None:
            return None

        # --- Langfuse v3/v4 path ---
        trace_api = getattr(client.api, "trace", None)
        get_method = getattr(trace_api, "get", None) if trace_api is not None else None
        if callable(get_method):
            trace = get_method(trace_id, request_options=_REQ_OPTS)
            observations = []
            for obs in getattr(trace, "observations", []) or []:
                observations.append({
                    "name": obs.name,
                    "latency_ms": obs.latency,
                    "model": getattr(obs, "model", None),
                    "cost": getattr(obs, "total_cost", None),
                })
            return {
                "trace_id": trace.id,
                "name": trace.name,
                "duration_ms": trace.latency,
                "total_cost": trace.total_cost,
                "observations": observations,
                "metadata": trace.metadata or {},
                "scores": getattr(trace, "scores", []) or [],
            }

        # --- Langfuse v5 path (derive trace from observations) ---
        obs_api = getattr(client.api, "observations", None)
        get_many = getattr(obs_api, "get_many", None) if obs_api is not None else None
        if callable(get_many):
            response = get_many(trace_id=trace_id, limit=200, request_options=_REQ_OPTS)
            obs_list = getattr(response, "data", None) or []
            if not obs_list:
                return None
            observations = []
            total_cost = 0.0
            max_latency = 0.0
            for obs in obs_list:
                latency = getattr(obs, "latency", None) or getattr(obs, "calculated_latency", None) or 0
                cost = None
                for attr in ("total_cost", "calculated_total_cost", "cost"):
                    cost = getattr(obs, attr, None)
                    if cost is not None:
                        break
                cost = cost or 0
                total_cost += cost
                max_latency = max(max_latency, latency or 0)
                observations.append({
                    "name": getattr(obs, "name", None) or getattr(obs, "type", None) or "span",
                    "latency_ms": latency,
                    "model": getattr(obs, "model", None),
                    "cost": cost,
                })
            return {
                "trace_id": trace_id,
                "name": "support-ticket",
                "duration_ms": max_latency,
                "total_cost": total_cost,
                "observations": observations,
                "metadata": {},
                "scores": [],
            }

        return None
    except Exception:
        logger.warning("Failed to fetch trace %s", trace_id, exc_info=True)
        return None


def get_aggregate_metrics(
    limit: int = 100,
    from_timestamp: datetime | None = None,
    to_timestamp: datetime | None = None,
) -> dict:
    """Compute aggregate metrics from recent Langfuse traces.

    Returns a dict with total_traces, avg_latency_ms, avg_cost,
    automation_pct, and category_distribution.
    On error, returns a dict with zeros and an 'error' field.

    Args:
        limit: Maximum number of traces to fetch.
        from_timestamp: Only include traces after this datetime (UTC).
        to_timestamp: Only include traces before this datetime (UTC).
    """
    empty = {
        "total_traces": 0,
        "avg_latency_ms": 0.0,
        "avg_cost": 0.0,
        "automation_pct": 0.0,
        "category_distribution": {},
    }

    try:
        client = get_langfuse_client()
        if client is None:
            return {**empty, "error": "Langfuse disabled or unavailable"}
        list_kwargs: dict = {"limit": limit}
        if from_timestamp:
            list_kwargs["from_timestamp"] = from_timestamp
        if to_timestamp:
            list_kwargs["to_timestamp"] = to_timestamp
        records = _list_trace_records(client, list_kwargs)
    except Exception as exc:
        logger.warning("Failed to fetch traces for metrics", exc_info=True)
        return {**empty, "error": str(exc)}

    if not records:
        return empty

    total = len(records)
    total_latency = sum(r.get("latency") or 0 for r in records)
    total_cost = sum(r.get("cost") or 0 for r in records)

    # Escalation rate
    try:
        score_kwargs: dict = {"name": "escalated", "limit": limit}
        if from_timestamp:
            score_kwargs["from_timestamp"] = from_timestamp
        if to_timestamp:
            score_kwargs["to_timestamp"] = to_timestamp
        escalation_scores = _scores_data(_fetch_scores(client, score_kwargs))
        escalated_count = sum(
            1 for s in escalation_scores
            if str(getattr(s, "value", "")).lower() == "true"
        )
        total_scored = len(escalation_scores)
        automation_pct = (
            round((1 - escalated_count / total_scored) * 100, 1)
            if total_scored > 0
            else 0.0
        )
    except Exception:
        logger.warning("Failed to fetch escalation scores", exc_info=True)
        automation_pct = 0.0

    # Category distribution
    category_distribution: dict[str, int] = {}
    try:
        cat_kwargs: dict = {"name": "category", "limit": limit}
        if from_timestamp:
            cat_kwargs["from_timestamp"] = from_timestamp
        if to_timestamp:
            cat_kwargs["to_timestamp"] = to_timestamp
        for s in _scores_data(_fetch_scores(client, cat_kwargs)):
            val = str(getattr(s, "string_value", None) or getattr(s, "value", "unknown"))
            category_distribution[val] = category_distribution.get(val, 0) + 1
    except Exception:
        logger.warning("Failed to fetch category scores", exc_info=True)

    return {
        "total_traces": total,
        "avg_latency_ms": round(total_latency / total, 2),
        "avg_cost": round(total_cost / total, 6),
        "automation_pct": automation_pct,
        "category_distribution": category_distribution,
    }
