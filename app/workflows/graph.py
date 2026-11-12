"""
LangGraph workflow builder.

Topology + compiled singleton. Node implementations live under
app.workflows.nodes.*; routing helpers live in app.workflows.routing.
"""

import asyncio
import logging
from typing import Any, Callable

from langgraph.graph import END, StateGraph

from app.workflows.state import ConversationState
from app.workflows.routing import (
    route_after_gate,
    route_after_payment,
    route_after_quote,
    route_after_search,
    route_by_intent,
    route_by_validation,
)

# Re-export for existing imports: `from app.workflows.graph import route_by_intent`, etc.
from app.workflows.nodes.save_state import save_state
from app.workflows.nodes.feature_gate_node import feature_gate

from app.workflows.nodes.context_loader import load_context
from app.workflows.nodes.understand import understand
from app.workflows.nodes.validator import validate
from app.workflows.nodes.room_searcher import search_rooms
from app.workflows.nodes.quote_generator import generate_quote
from app.workflows.nodes.payment_handler import handle_payment
from app.workflows.nodes.booking_confirmer import confirm_booking
from app.workflows.nodes.faq_handler import handle_faq
from app.workflows.nodes.response_generator import generate_response
from app.workflows.nodes.cancellation_handler import handle_cancellation
from app.workflows.nodes.room_query_handler import handle_room_query
from app.workflows.nodes.menu_handler import handle_menu
from app.workflows.nodes.escalation_handler import handle_escalation
from app.workflows.nodes.task_handler import handle_task
from app.workflows.nodes.price_match_handler import handle_price_match
from app.workflows.nodes.silent_upload_handler import handle_silent_upload
from app.workflows.nodes.room_service_handler import handle_room_service
from app.workflows.nodes.amenity_handler import handle_amenity_node
from app.workflows.nodes.modification_handler import handle_modification
from app.workflows.nodes.lifecycle_handler import handle_lifecycle
from app.workflows.nodes.bank_details_handler import handle_bank_details
from app.workflows.nodes.stage_handlers.stage_payment_check import handle_stage_payment_check
from app.workflows.nodes.stage_handlers.stage_confirmed import handle_stage_confirmed
from app.workflows.nodes.stage_handlers.stage_in_house import handle_stage_in_house
from app.workflows.nodes.stage_handlers.stage_checked_out import handle_stage_checked_out
from app.workflows.nodes.name_after_payment import handle_name_after_payment

logger = logging.getLogger(__name__)

__all__ = [
    "build_graph",
    "get_compiled_graph",
    "save_state",
    "feature_gate",
    "route_after_gate",
    "route_by_intent",
    "route_by_validation",
    "route_after_search",
    "route_after_payment",
]


def _instrument_node(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a node to record latency and error counters."""

    def _labels(state: Any) -> dict:
        hotel_id = state.get("hotel_id") if isinstance(state, dict) else None
        return {"node": name, "hotel_id": hotel_id}

    if asyncio.iscoroutinefunction(fn):
        async def async_wrapper(state: ConversationState) -> ConversationState:
            from app.observability.metrics import metrics, record_node_error

            labels = _labels(state)
            try:
                with metrics.timer("workflow.node_latency_ms", **labels):
                    return await fn(state)
            except Exception as e:
                from app.observability.failure_store import record_failure
                from app.observability.tracing import bind_context

                sid = state.get("session_id") if isinstance(state, dict) else None
                if sid:
                    bind_context(session_id=sid, hotel_id=labels.get("hotel_id"))
                record_node_error(name, hotel_id=labels.get("hotel_id"))
                record_failure(
                    component=f"workflow.{name}",
                    error=e,
                    hotel_id=labels.get("hotel_id"),
                    session_id=sid,
                    details={"node": name},
                    exc=e,
                )
                logger.error("NODE_FAILED | node=%s error=%s", name, e, exc_info=True)
                raise

        async_wrapper.__name__ = getattr(fn, "__name__", name)
        return async_wrapper

    def sync_wrapper(state: ConversationState) -> ConversationState:
        from app.observability.metrics import metrics, record_node_error

        labels = _labels(state)
        try:
            with metrics.timer("workflow.node_latency_ms", **labels):
                return fn(state)
        except Exception as e:
            from app.observability.failure_store import record_failure

            sid = state.get("session_id") if isinstance(state, dict) else None
            record_node_error(name, hotel_id=labels.get("hotel_id"))
            record_failure(
                component=f"workflow.{name}",
                error=e,
                hotel_id=labels.get("hotel_id"),
                session_id=sid,
                details={"node": name},
                exc=e,
            )
            logger.error("NODE_FAILED | node=%s error=%s", name, e, exc_info=True)
            raise

    sync_wrapper.__name__ = getattr(fn, "__name__", name)
    return sync_wrapper


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph workflow.

    Called once at startup. The compiled graph is reused for all conversations.
    """
    graph = StateGraph(ConversationState)

    graph.add_node("load_context", _instrument_node("load_context", load_context))
    graph.add_node("understand", _instrument_node("understand", understand))
    graph.add_node("feature_gate", _instrument_node("feature_gate", feature_gate))
    graph.add_node("intent_router", _instrument_node("intent_router", lambda state: state))
    graph.add_node("validate", _instrument_node("validate", validate))
    graph.add_node("search_rooms", _instrument_node("search_rooms", search_rooms))
    graph.add_node("generate_quote", _instrument_node("generate_quote", generate_quote))
    graph.add_node("handle_payment", _instrument_node("handle_payment", handle_payment))
    graph.add_node("confirm_booking", _instrument_node("confirm_booking", confirm_booking))
    graph.add_node("handle_faq", _instrument_node("handle_faq", handle_faq))
    graph.add_node("generate_response", _instrument_node("generate_response", generate_response))
    graph.add_node("handle_cancellation", _instrument_node("handle_cancellation", handle_cancellation))
    graph.add_node("handle_room_query", _instrument_node("handle_room_query", handle_room_query))
    graph.add_node("handle_menu", _instrument_node("handle_menu", handle_menu))
    graph.add_node("handle_escalation", _instrument_node("handle_escalation", handle_escalation))
    graph.add_node("hotel_task", _instrument_node("hotel_task", handle_task))
    graph.add_node("room_service", _instrument_node("room_service", handle_room_service))
    graph.add_node("price_match", _instrument_node("price_match", handle_price_match))
    graph.add_node("silent_upload", _instrument_node("silent_upload", handle_silent_upload))
    graph.add_node("amenity_query", _instrument_node("amenity_query", handle_amenity_node))
    graph.add_node("handle_modification", _instrument_node("handle_modification", handle_modification))
    graph.add_node("lifecycle_handler", _instrument_node("lifecycle_handler", handle_lifecycle))
    graph.add_node("bank_details", _instrument_node("bank_details", handle_bank_details))
    graph.add_node("stage_payment_check", _instrument_node("stage_payment_check", handle_stage_payment_check))
    graph.add_node("stage_confirmed", _instrument_node("stage_confirmed", handle_stage_confirmed))
    graph.add_node("stage_in_house", _instrument_node("stage_in_house", handle_stage_in_house))
    graph.add_node("stage_checked_out", _instrument_node("stage_checked_out", handle_stage_checked_out))
    graph.add_node("name_after_payment", _instrument_node("name_after_payment", handle_name_after_payment))
    graph.add_node("save_state", _instrument_node("save_state", save_state))

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "understand")
    graph.add_edge("understand", "feature_gate")

    graph.add_conditional_edges("feature_gate", route_after_gate, {
        "blocked": "save_state",
        "allowed": "intent_router",
    })

    graph.add_conditional_edges("intent_router", route_by_intent, {
        "greeting": "generate_response",
        "thank_you": "generate_response",
        "faq": "handle_faq",
        "booking_flow": "validate",
        "payment": "handle_payment",
        "bank_details": "bank_details",
        "cancel_booking": "handle_cancellation",
        "room_query": "handle_room_query",
        "menu": "handle_menu",
        "escalation": "handle_escalation",
        "hotel_task": "hotel_task",
        "room_service": "room_service",
        "price_match": "price_match",
        "silent_upload": "silent_upload",
        "amenity_query": "amenity_query",
        "handle_modification": "handle_modification",
        "lifecycle": "lifecycle_handler",
        "stage_payment_check": "stage_payment_check",
        "stage_confirmed": "stage_confirmed",
        "stage_in_house": "stage_in_house",
        "stage_checked_out": "stage_checked_out",
        "name_after_payment": "name_after_payment",
        "unknown": "generate_response",
    })

    graph.add_conditional_edges("validate", route_by_validation, {
        "need_info": "generate_response",
        "search": "search_rooms",
        "quote": "generate_quote",
        "payment": "handle_payment",
    })

    graph.add_conditional_edges("search_rooms", route_after_search, {
        "no_rooms": "generate_response",
        "done": "save_state",
        "quote": "generate_quote",
    })

    graph.add_conditional_edges("handle_payment", route_after_payment, {
        "confirm": "confirm_booking",
        "done": "save_state",
    })

    graph.add_conditional_edges("generate_quote", route_after_quote, {
        "search": "search_rooms",
        "done": "save_state",
    })

    graph.add_edge("generate_response", "save_state")
    graph.add_edge("confirm_booking", "save_state")
    graph.add_edge("handle_faq", "save_state")
    graph.add_edge("handle_cancellation", "save_state")
    graph.add_edge("handle_room_query", "save_state")
    graph.add_edge("handle_menu", "save_state")
    graph.add_edge("handle_escalation", "save_state")
    graph.add_edge("hotel_task", "save_state")
    graph.add_edge("room_service", "save_state")
    graph.add_edge("price_match", "save_state")
    graph.add_edge("silent_upload", "save_state")
    graph.add_edge("amenity_query", "save_state")
    graph.add_edge("handle_modification", "save_state")
    graph.add_edge("lifecycle_handler", "save_state")
    graph.add_edge("bank_details", "save_state")
    graph.add_edge("stage_payment_check", "save_state")
    graph.add_edge("stage_confirmed", "save_state")
    graph.add_edge("stage_in_house", "save_state")
    graph.add_edge("stage_checked_out", "save_state")
    graph.add_edge("name_after_payment", "save_state")
    graph.add_edge("save_state", END)

    return graph


_compiled_graph = None


def get_compiled_graph():
    """Get or build the compiled graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph
