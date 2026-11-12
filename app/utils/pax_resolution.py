"""
Who reads the party size — the model or the regex — and in what order.

Why this is a module and not four lines inside ``understand()``
--------------------------------------------------------------
It was four lines inside ``understand()``, and they were an either/or with the
wrong side first. The regex branch came first and an ``elif`` gave the model its
turn only when the regex had found nothing. On 2026-09-24 that produced:

    "Check in: 12/11/26 check out 15/11/26 person 5+1"

The regex matched the ``26`` of the checkout year — sitting exactly where a
headcount belongs — and wrote ``adults=26, children=0``. The model read the same
sentence as ``adults=5, children=1``, which is what the guest asked for, and was
never consulted. A guest who wanted six people was told the bot had 26, then
that no room could seat them.

The inversion is one word of structure, which is precisely why it needs a test:
nothing about reading the function tells you which side wins unless you trace
the ``if``/``elif``, and the losing side fails silently. Inline in a 2100-line
function no test could reach it. Here it is a pure function over six values, and
``tests/test_pax_resolution.py`` pins the ordering directly.

The rule, stated once
---------------------
**The model is the reader; the regex is the fallback.** The model handles "5+1",
Hinglish, and spellings no pattern anticipates. The regex stays because it is
instant and exact on the shapes it does know, and because a model that returns
nothing (an outage, or a field it did not see) must not leave the field
unfilled. Preferring the reader over the fallback — and never letting the
fallback overrule a value the reader actually produced — is the whole rule.

One consequence worth knowing before you edit this. In the regex branch
``children`` is forced to ``0``, because the regex has no way to see a child
mixed into a pax phrase and a consistent reading is better than a stale one. In
the model branch ``children`` is written **only when the model saw one** — it is
never zeroed. That asymmetry is deliberate: the ``+1`` above is the child the
old zeroing used to discard.
"""

from __future__ import annotations

from typing import Any
import re


def resolve_pax_counts(
    *,
    model_adults: Any = None,
    model_children: Any = None,
    regex_pax: Any = None,
    has_child_kw: bool = False,
    is_option_number_selection: bool = False,
    expected_action: Any = None,
) -> tuple[dict[str, Any], bool]:
    """
    The count fields to write, and whether the regex answered.

    Returns ``(updates, used_regex)``. The caller merges ``updates`` into its own
    dict; ``used_regex`` is returned separately because one regex-only field
    (``num_rooms``, read beside the pax phrase) belongs to the caller's scope and
    must not be written when the model answered.

    Pure: no I/O, no model, no state. Every branch below is reachable from a
    test with plain arguments.
    """
    updates: dict[str, Any] = {}

    # The model first. `expected_action == "need_child_ages"` is preserved from
    # the pre-2026-09-24 code: while the bot is collecting ages, a headcount the
    # model volunteered does not overwrite the party it is collecting for.
    if (
        model_adults is not None
        and not is_option_number_selection
        and expected_action != "need_child_ages"
    ):
        updates["adults"] = model_adults
        updates["effective_adults"] = model_adults
        if model_children is not None and not is_option_number_selection:
            updates["children"] = model_children
            updates["effective_children"] = model_children
            if model_children == 0:
                updates["child_ages"] = []
                updates["child_age_ask_count"] = 0
        return updates, False

    # The regex, only when the model returned no adults at all.
    if regex_pax is not None and not has_child_kw and not is_option_number_selection:
        updates["adults"] = regex_pax
        updates["effective_adults"] = regex_pax
        # Forced, and only in this branch — see the module docstring.
        updates["children"] = 0
        updates["effective_children"] = 0
        updates["child_ages"] = []
        updates["child_age_ask_count"] = 0
        return updates, True

    return updates, False


def extract_structured_heading_pax(text: str) -> tuple[int | None, int | None, list[int]]:
    """
    Extract adults and children counts (and child ages) from structured heading blocks,
    such as 'Adults:- ... Children:- 1. Aarav (7) 2. Anaya (2.5)'.

    Handles decimal ages (e.g. 2.5 yrs -> age 2) and line-by-line guest lists.
    """
    if not text or not text.strip():
        return None, None, []

    adults_count: int | None = None
    children_count: int | None = None
    child_ages: list[int] = []

    # 1. Look for Children block: e.g. "Children:-", "Children:", "Kids:-", "Bachche:-"
    m_child_heading = re.search(r'(?:^|\n)\s*(?:children|kids|bachche?)\s*[:\-]+[ \t]*(.*)', text, re.IGNORECASE)
    if m_child_heading:
        rest_of_heading = m_child_heading.group(1).strip()
        start_pos = m_child_heading.end()
        remaining_text = text[start_pos:]

        m_next = re.search(r'\n\s*(?:adults|rooms?|dates?|check[- ]?in|payment|advance|notes?)\s*[:\-]+', remaining_text, re.IGNORECASE)
        child_block = remaining_text[:m_next.start()] if m_next else remaining_text

        lines = [line.strip() for line in child_block.splitlines() if line.strip()]

        m_count_inline = re.match(r'^(\d+)\b', rest_of_heading)
        if m_count_inline:
            children_count = int(m_count_inline.group(1))

        child_entries = []
        for line in lines:
            if re.match(r'^(?:\d+[\.\)]|[-*•])\s*', line) or re.search(r'\d+(?:\.\d+)?\s*(?:yrs?|years?|yo|saal|m|months?)|\(\s*\d+(?:\.\d+)?\s*\)', line, re.I):
                child_entries.append(line)

        all_child_text = rest_of_heading + " " + " ".join(child_entries if child_entries else lines)

        age_matches = re.findall(r'(?:\(\s*(\d+(?:\.\d+)?)\s*(?:yrs?|years?|yo|saal|m|months?)?\s*\)|\b(\d+(?:\.\d+)?)\s*(?:yrs?|years?|yo|saal)\b)', all_child_text, re.I)
        for m1, m2 in age_matches:
            raw_age = m1 or m2
            try:
                age_val = float(raw_age)
                child_ages.append(int(age_val))
            except Exception:
                pass

        total_children = max(
            children_count or 0,
            len(child_entries),
            len(child_ages),
        )
        if total_children > 0:
            children_count = total_children

    # 2. Look for Adults block: e.g. "Adults:-", "Adults:", "Grown-ups:-"
    m_adult_heading = re.search(r'(?:^|\n)\s*(?:adults|guests?|bade)\s*[:\-]+[ \t]*(.*)', text, re.IGNORECASE)
    if m_adult_heading:
        rest_of_heading = m_adult_heading.group(1).strip()
        start_pos = m_adult_heading.end()
        remaining_text = text[start_pos:]
        m_next = re.search(r'\n\s*(?:children|kids|bachche?|rooms?|dates?|check[- ]?in|payment)\s*[:\-]+', remaining_text, re.IGNORECASE)
        adult_block = remaining_text[:m_next.start()] if m_next else remaining_text

        m_count_inline = re.match(r'^(\d+)\b', rest_of_heading)
        if m_count_inline:
            adults_count = int(m_count_inline.group(1))

        lines = [line.strip() for line in adult_block.splitlines() if line.strip()]
        adult_entries = [line for line in lines if re.match(r'^(?:\d+[\.\)]|[-*•])\s*', line) or re.search(r'\b\w+\b', line)]
        if len(adult_entries) > 0:
            adults_count = max(adults_count or 0, len(adult_entries))

    return adults_count, children_count, child_ages
