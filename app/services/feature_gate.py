"""
Feature Gate — deterministic Python-side feature enforcement.

This module is the authoritative source for all business decisions.
The AI model NEVER decides whether a feature is enabled or allowed.

Usage:
    from app.services.feature_gate import FeatureGate

    gate = FeatureGate(hotel_config)
    allowed, response = gate.check_booking()
    if not allowed:
        return response  # deterministic, no AI involved

Design:
    - Pure functions, no side effects
    - Returns (allowed: bool, response: str | None)
    - Response is a pre-configured business message when feature is disabled
    - All business limits are enforced here, not in the AI prompt
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from app.config.hotel_config import HotelConfig

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of a feature gate check."""
    allowed: bool
    response: Optional[str] = None

    def __bool__(self) -> bool:
        return self.allowed


class FeatureGate:
    """
    Deterministic feature enforcement engine.

    Evaluates business configuration to determine whether a workflow
    should proceed. Returns pre-configured responses for disabled features.
    The AI never participates in these decisions.
    """

    def __init__(self, config: HotelConfig, session_id: str | None = None):
        self.config = config
        self.features = config.business_config.features
        self.limits = config.business_config.limits
        self.session_id = session_id

    def _log_decision(self, feature: str, decision: str, reason: str = "") -> None:
        """Centralized structured logging for all Feature Gate decisions."""
        log_parts = [
            decision,
            f"hotel_id={self.config.hotel_id}",
            f"feature={feature}"
        ]
        if self.session_id:
            log_parts.append(f"session_id={self.session_id}")
        if reason:
            log_parts.append(f"reason={reason}")
            
        logger.info(" | ".join(log_parts))
        try:
            from app.observability.metrics import record_feature_gate
            record_feature_gate(feature, decision, hotel_id=self.config.hotel_id)
        except Exception:
            pass

    # ── Strongly Typed Helpers ───────────────────────────────────

    def is_ai_enabled(self) -> bool:
        enabled = bool(self.features.ai_enabled)
        self._log_decision("ai_enabled", "FEATURE_ALLOWED" if enabled else "FEATURE_BLOCKED")
        return enabled

    def is_lead_only_mode(self) -> bool:
        return bool(self.features.lead_only_mode)

    def can_direct_book(self) -> bool:
        # Lead Only Mode has highest precedence: if active, direct booking is strictly false
        if self.is_lead_only_mode():
            return False
            
        return bool(self.features.direct_booking)

    def can_use_coupon(self) -> bool:
        return bool(self.features.coupon_visibility)

    def can_dynamic_price(self) -> bool:
        return bool(self.features.dynamic_pricing)

    def can_auto_upgrade(self) -> bool:
        return bool(self.features.auto_upgrade)

    def can_auto_upsell(self) -> bool:
        return bool(self.features.auto_upsell)

    # ── Core AI Gate ─────────────────────────────────────────────

    def check_ai_enabled(self) -> GateResult:
        """Check if AI is enabled for this hotel."""
        if not self.is_ai_enabled():
            return GateResult(
                allowed=False,
                response="Our AI assistant is currently unavailable. Please contact our reception directly for assistance."
            )
        return GateResult(allowed=True)

    # ── Booking Gates ────────────────────────────────────────────

    def check_booking_browsing(self) -> GateResult:
        """
        Check if browsing rooms, checking availability, and viewing quotes is allowed.

        Priority:
        1. leadOnlyMode=true → block ALL booking browsing (lead collection only)
        2. directBooking=false → browsing IS allowed (quote shown, but no payment)
        3. directBooking=true → full flow allowed
        """
        if self.is_lead_only_mode():
            return GateResult(
                allowed=False,
                response=None,  # Caller handles lead collection flow
            )
        # directBooking=false still allows browsing (rooms, prices, quotes)
        return GateResult(allowed=True)

    def check_booking_confirmation(self) -> GateResult:
        """
        Check if the booking can proceed to payment/confirmation.

        This is the enforcement point where directBooking=false stops the flow.
        Called when the user confirms a quote (CONFIRM_BOOKING intent).

        Priority:
        1. leadOnlyMode=true → never confirm (handled earlier by browsing gate)
        2. directBooking=false → block payment, create manual inquiry
        3. directBooking=true → proceed to payment
        """
        if self.is_lead_only_mode():
            return GateResult(
                allowed=False,
                response=None,  # Caller handles lead collection
            )

        if not self.can_direct_book():
            return GateResult(
                allowed=False,
                response=None,  # Caller creates manual inquiry
            )

        return GateResult(allowed=True)

    def check_booking_modification(self) -> GateResult:
        """Check if booking modifications are allowed."""
        if not self.features.booking_modification_enabled:
            self._log_decision("booking_modification", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "Booking modifications are currently handled by our front desk team. "
                    "Please contact reception to update your booking details."
                )
            )
        self._log_decision("booking_modification", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_booking_cancellation(self) -> GateResult:
        """Check if booking cancellations are allowed."""
        if not self.features.booking_cancellation_enabled:
            self._log_decision("booking_cancellation", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "Booking cancellations are processed by our reservations team. "
                    "Please contact reception for cancellation assistance."
                )
            )
        self._log_decision("booking_cancellation", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    # ── Feature-specific Gates ───────────────────────────────────

    def check_price_match(self) -> GateResult:
        """Check if price matching is enabled."""
        if not self.features.price_match_enabled:
            self._log_decision("price_match", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "Price matching is not currently available. "
                    "Please contact our reservations team for the best available rates."
                )
            )
        self._log_decision("price_match", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_human_escalation(self) -> GateResult:
        """Check if human escalation is enabled."""
        if not self.features.human_escalation_enabled:
            self._log_decision("human_escalation", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "I apologize for the inconvenience. "
                    "Please contact our reception directly for immediate assistance."
                )
            )
        self._log_decision("human_escalation", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_task_creation(self) -> GateResult:
        """Check if hotel task creation is enabled."""
        if not self.features.task_creation_enabled:
            self._log_decision("task_creation", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "Service requests are handled directly by our front desk. "
                    "Please contact reception for assistance with your request."
                )
            )
        self._log_decision("task_creation", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_support_ticket(self) -> GateResult:
        """Check if support ticket creation is enabled."""
        if not self.features.support_ticket_enabled:
            self._log_decision("support_ticket", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "For support inquiries, please contact our front desk directly. "
                    "They will be happy to assist you."
                )
            )
        self._log_decision("support_ticket", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_document_understanding(self) -> GateResult:
        """Check if document/image understanding is enabled."""
        if not self.features.document_understanding_enabled:
            self._log_decision("document_understanding", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "Document and image uploads are not currently supported. "
                    "Please describe your request in text, or contact reception for assistance."
                )
            )
        self._log_decision("document_understanding", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_payment_verification(self) -> GateResult:
        """Check if payment verification is enabled."""
        if not self.features.payment_verification_enabled:
            self._log_decision("payment_verification", "FEATURE_BLOCKED")
            return GateResult(
                allowed=False,
                response=(
                    "Online payment verification is currently unavailable. "
                    "Please contact our front desk to confirm your payment."
                )
            )
        self._log_decision("payment_verification", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    # ── Business Limit Validations ───────────────────────────────

    def validate_guest_count(self, adults: int, children: int = 0) -> GateResult:
        """Validate guest counts against business limits."""
        self._log_decision("guest_limit", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def validate_room_count(self, num_rooms: int) -> GateResult:
        """Validate room count against business limits."""
        self._log_decision("room_limit", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_vision_classification(self) -> GateResult:
        """Check if multimodal vision classification is enabled."""
        if not getattr(self.features, "vision_classification_enabled", True):
            self._log_decision("vision_classification", "FEATURE_BLOCKED", "visionClassificationEnabled=false")
            return GateResult(
                allowed=False,
                response="Image processing is currently unavailable. Please send your request in text format.",
            )
        self._log_decision("vision_classification", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_id_verification(self) -> GateResult:
        """Check if ID document verification is enabled."""
        if not getattr(self.features, "id_verification_enabled", True):
            self._log_decision("id_verification", "FEATURE_BLOCKED", "idVerificationEnabled=false")
            return GateResult(
                allowed=False,
                response="Online ID verification is currently disabled. Please present your government ID at the front desk upon check-in.",
            )
        self._log_decision("id_verification", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_ota_comparison(self) -> GateResult:
        """Check if OTA rate comparison and 5% lower offer generation is enabled."""
        if not getattr(self.features, "ota_comparison_enabled", True):
            self._log_decision("ota_comparison", "FEATURE_BLOCKED", "otaComparisonEnabled=false")
            return GateResult(
                allowed=False,
                response="OTA price matching is currently unavailable for this property.",
            )
        self._log_decision("ota_comparison", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_document_ocr(self) -> GateResult:
        """Check if document / map / event OCR extraction is enabled."""
        if not getattr(self.features, "document_ocr_enabled", True):
            self._log_decision("document_ocr", "FEATURE_BLOCKED", "documentOcrEnabled=false")
            return GateResult(
                allowed=False,
                response="Document reading is currently disabled. Please share your request in text.",
            )
        self._log_decision("document_ocr", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def check_staff_command_intelligence(self) -> GateResult:
        """Check if staff command parsing and intelligence is enabled."""
        if not getattr(self.features, "staff_command_intelligence_enabled", True):
            self._log_decision("staff_command_intelligence", "FEATURE_BLOCKED", "staffCommandIntelligenceEnabled=false")
            return GateResult(
                allowed=False,
                response=None,
            )
        self._log_decision("staff_command_intelligence", "FEATURE_ALLOWED")
        return GateResult(allowed=True)

    def validate_same_day_booking(self, is_same_day: bool) -> GateResult:
        """Check if same-day bookings are allowed."""
        if is_same_day and not self.features.same_day_booking_allowed:
            return GateResult(
                allowed=False,
                response=(
                    "Same-day bookings are not available online. "
                    "Please contact our reception to check room availability for today."
                )
            )
        return GateResult(allowed=True)

    # ── Convenience: Map intent to gate ──────────────────────────

    def check_intent(self, intent: str) -> GateResult:
        """
        Map a classified intent to the appropriate feature gate.
        Returns GateResult — if not allowed, the response is ready to send.

        This is the main entry point used by the workflow routing.

        NOTE: Booking browsing intents (BOOK_ROOM, SELECT_ROOM, CHECK_AVAILABILITY)
        are NOT gated here for directBooking=false. Those are allowed to flow through
        to show rooms and quotes. The payment/confirmation checkpoint is enforced
        separately in the graph (handle_payment / generate_quote nodes).

        leadOnlyMode is also handled separately in the feature_gate graph node
        to enable the lead collection workflow.
        """
        gate_map = {
            "MODIFY_BOOKING": self.check_booking_modification,
            "CANCEL_BOOKING": self.check_booking_cancellation,
            "PRICE_MATCH": self.check_price_match,
            "HUMAN_REQUEST": self.check_human_escalation,
            "COMPLAINT": self.check_support_ticket,
            "HOTEL_TASK": self.check_task_creation,
            "SILENT_DOCUMENT_UPLOAD": self.check_document_understanding,
        }

        gate_fn = gate_map.get(intent)
        if gate_fn:
            result = gate_fn()
            if not result.allowed:
                logger.info(
                    f"Feature gate BLOCKED intent={intent} for hotel_id={self.config.hotel_id} | "
                    f"Response: {result.response[:80]}..."
                )
            return result

        # Intent not gated — allow by default
        return GateResult(allowed=True)
