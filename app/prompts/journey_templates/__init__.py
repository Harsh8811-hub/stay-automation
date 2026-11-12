from typing import Callable, Any

from app.prompts.journey_templates.inquiry_templates import inq_greet
from app.prompts.journey_templates.lead_templates import lq_quote, lq_urgency, lq_objection
from app.prompts.journey_templates.payment_templates import payment_link, payment_failed
from app.prompts.journey_templates.confirmation_templates import booking_confirmed, cfm_addon
from app.prompts.journey_templates.instay_templates import stay_wifi
from app.prompts.journey_templates.escalation_templates import stay_neg

TEMPLATE_REGISTRY = {
    "INQ_GREET": inq_greet,
    "LQ_QUOTE": lq_quote,
    "LQ_URGENCY": lq_urgency,
    "LQ_OBJECTION": lq_objection,
    "PAYMENT_LINK": payment_link,
    "PAYMENT_FAILED": payment_failed,
    "BOOKING_CONFIRMED": booking_confirmed,
    "CFM_ADDON": cfm_addon,
    "STAY_WIFI": stay_wifi,
    "STAY_NEG": stay_neg,
}

def get_template(template_id: str, lang: str = "en", **kwargs) -> list[dict]:
    """
    Get a formatted template response based on the template ID.
    If language preference is provided and supported, templates could be adapted here.
    """
    template_func = TEMPLATE_REGISTRY.get(template_id)
    if not template_func:
        # Fallback to a plain text if missing
        return [{"type": "text", "text": f"[{template_id}] Content pending..."}]
    
    return template_func(**kwargs)
