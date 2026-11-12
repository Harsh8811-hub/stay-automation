"""Does the guest's own message name an online travel agency, or ask us to beat one?

One pattern, two readers. `understand` uses it to decide that a *general* discount
question is really a price-match request; `price_match_handler` uses it to decide
whether it may compare our rate to a portal's. Those two answers have to agree —
if the handler thinks a comparison is unasked-for while `understand` thinks the
guest raised MakeMyTrip, the guest gets routed to a price match and then told
nothing about the price they raised.

It lived inline in `understand.py` until 2026-10-14. It is a module now because
the second reader arrived, and this codebase has paid for duplicated matchers
before: three copies of "which room did the guest mean" disagreed and looped a
guest forever (`stayinn-split-line-not-a-room`).

The pattern is moved here verbatim — not rewritten — so that `understand`'s
behaviour is unchanged by the extraction.
"""

import re

# Each alternation is a way a guest raises a competitor:
#   * a direct comparison ("cheaper online", "lower rate online")
#   * a portal by name, including the `mmt` abbreviation
#   * a Hinglish form ("online sasta", "online dekha")
#   * the vocabulary of matching itself ("competitor", "price match")
#
# `\b` at both ends, and `re.IGNORECASE` even though callers pass lowercased
# text, so the helper is safe on whatever casing it is handed.
COMPETITOR_PATTERN = re.compile(
    r"\b(?:cheaper\s+online|lower\s+(?:price|rate)\s+online|getting\s+cheaper"
    r"|makemytrip|mmt|agoda|goibibo|booking\.com|expedia|tripadvisor|yatra|easemytrip|cleartrip"
    r"|online\s+(?:sasta|kam|cheaper|kamti|portal|app)|online\s+dekha|competitor"
    r"|match\s+(?:the\s+)?price|price\s+match)\b",
    re.IGNORECASE,
)


def mentions_competitor(text: str | None) -> bool:
    """True when the guest's own words raise a portal or a price match.

    `None` and empty string are both False, so a caller reading a session field
    that was never written gets the conservative answer — no comparison offered.
    """
    if not text:
        return False
    return bool(COMPETITOR_PATTERN.search(str(text)))
