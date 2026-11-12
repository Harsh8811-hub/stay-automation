"""
Payment Handler Node — generates payment link and checks status.

Implementation lives in app.workflows.nodes.payment.*; this module
keeps the historical import path stable.
"""

from app.workflows.nodes.payment.handler import handle_payment

__all__ = ["handle_payment"]
