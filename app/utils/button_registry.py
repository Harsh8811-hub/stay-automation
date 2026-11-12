"""
Every reply button the bot can send, and what a tap on one means.

WHY THIS MODULE EXISTS
----------------------
A WhatsApp reply button is sent as an id and a title, but **only the title comes back**.
The CRM unwraps a tap and forwards it as an ordinary text message, dropping
`button_reply.id` (`combot_crm_v2/webhook_receiver/views.py:115-128`). So the router has to
recognise the buttons by the words we wrote on them.

Before this module those words lived in two unrelated places: the templates that *send* a
button, and a set of hand-typed literal lists in `understand.py` that *read* it. Nothing
tied them together, and anything that drifted became a control that looks wired and does
nothing. Five had already drifted by 2026-10-15 — see `tests/test_button_registry.py`, which
now fails if a sixth does.

WHAT THIS MODULE IS
-------------------
The closed set of buttons, each with the action it means and **every** title it may be sent
with. It is:

  * the routing table — `resolve_button()` is the single answer to "what did this tap mean";
  * the contract — `template_renderers.button_card` refuses a title that is not declared
    here, so a template cannot invent a button nobody can answer;
  * the guard — a test walks every authored button in `app/` and fails on any id or title
    this file does not know.

WHAT IT IS NOT
--------------
It does not send anything, and it holds no per-conversation state. A tap is resolved from
its own text, with no memory of what was offered — deliberately, because a remembered
"last menu" is a stale prompt that hijacks a later typed message.

Title normalisation (`normalize_reply_text`) is shared by the declaring side and the reading
side, so a title matches regardless of case, padding, or punctuation the channel may add.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "Button",
    "BUTTONS",
    "BUTTONS_BY_ID",
    "TAP_ROUTED_ACTIONS",
    "MEAL_PLAN_BY_ACTION",
    "normalize_reply_text",
    "resolve_button",
    "collect_titles",
]


# ============================================================================
# Normalisation
# ============================================================================

_PUNCTUATION = re.compile(r"[^\w\s]", re.UNICODE)


def normalize_reply_text(text: str | None) -> str:
    """
    The one normalisation used on both sides of a button.

    Case, surrounding space, runs of whitespace, and punctuation are all removed, so
    `"Room types & rates"`, `"room types and rates"`'s ampersand cousin, and a title that
    arrives with a stray trailing period all land on the same key.

    Punctuation is *dropped* rather than kept, which is what separates `"No"` (the
    extra-bed answer) from `"No, keep booking"` (the cancellation abort). A substring or
    prefix comparison would let the first swallow the second.
    """
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    return _PUNCTUATION.sub("", collapsed).casefold()


# ============================================================================
# The buttons
# ============================================================================


@dataclass(frozen=True)
class Button:
    """One reply button: a stable id, the action it means, and every title it may carry."""

    id: str
    action: str
    titles: tuple[str, ...]

    def matches(self, normalized_text: str) -> bool:
        """True when `normalized_text` is one of this button's declared titles."""
        return normalized_text in {normalize_reply_text(t) for t in self.titles}


#: Every button the bot can send. Ids are stable and are what arrives if the CRM ever
#: starts forwarding `button_reply.id`; titles are what arrives today.
BUTTONS: tuple[Button, ...] = (
    # ---- choosing a room combination ---------------------------------------
    # The room card prints its combinations as 1️⃣ / 2️⃣ / 3️⃣ and asks the guest
    # to type one. These are the same three choices as taps.
    #
    # Deliberately NOT in `TAP_ROUTED_ACTIONS` (see the note under that set): an
    # "Option 2" tap only means something while the room card is the active
    # prompt, and the reader that knows that is `room_card_reply` — which
    # already resolves typed "Option 2", "2" and the keycaps by position. A tap
    # sends the title back as plain text, so it lands on that same reader
    # instead of needing a second, faster path that could disagree with it.
    Button("room_option_1", "ROOM_OPTION_1", ("Option 1",)),
    Button("room_option_2", "ROOM_OPTION_2", ("Option 2",)),
    Button("room_option_3", "ROOM_OPTION_3", ("Option 3",)),
    # ---- the main menu and its relatives -----------------------------------
    # Five titles, one action. The greeting, the returning-guest greeting, the fallback and
    # the "something else" reply all offer "look at rooms", and each words it differently.
    Button(
        "menu_rooms",
        "SHOW_ROOMS",
        (
            "Room types & rates",
            "Kamre aur rates",
            # Pinned as SHOW_ROOMS by tests/test_button_registry.py: a returning
            # guest's "Book again" browses before it commits, which is the same
            # door the greeting's "Book a room" opens directly.
            "Book again",
            "Fir se book karein",
            "Room options",
        ),
    ),
    # The greeting used to offer three buttons, none of which *started* a booking:
    # "Room types & rates" browses, "Check booking" looks backwards. A guest who
    # wants to book had to type it. This is that missing door, and it opens the
    # collecting flow directly rather than the price list.
    Button(
        "menu_book",
        "BOOK_NOW",
        ("Book a room", "Room shuru karein", "Book now"),
    ),
    Button("menu_booking", "CHECK_EXISTING_BOOKING", ("Check booking", "Booking check karein")),
    Button(
        "menu_reception",
        "HUMAN_REQUEST",
        (
            "Talk to reception",
            "Reception se baat",
            "Talk to manager",
            "Request callback",
        ),
    ),
    # ---- payment -----------------------------------------------------------
    Button("payment_paid", "PAYMENT_PAID", ("I've paid",)),
    Button(
        "quote_confirm",
        "CONFIRM_BOOKING",
        ("Confirm Booking", "Proceed at best rate", "New payment link"),
    ),
    # ---- cancellation ------------------------------------------------------
    Button("cancel_confirm", "CANCEL_BOOKING", ("Yes, cancel",)),
    # "No, keep booking" was authored in `cancellation_handler.py` while the router only knew
    # "Keep booking". Both are declared: the same button is worded two ways at two sites, and
    # the longer one is the one a guest taps to *stop* a cancellation.
    Button("cancel_abort", "CANCEL_ABORT", ("Keep booking", "No, keep booking")),
    # ---- idle nudge --------------------------------------------------------
    Button("nudge_continue", "NUDGE_CONTINUE", ("Continue booking",)),
    Button("nudge_cancel", "NUDGE_CANCEL", ("Not now",)),
    # ---- meal plan ---------------------------------------------------------
    Button("meal_ep", "MEAL_EP", ("Room Only (EP)",)),
    Button("meal_cp", "MEAL_CP", ("Breakfast (CP)",)),
    Button("meal_map", "MEAL_MAP", ("MAP (Bkfast+Din)",)),
    # ---- AC preference -----------------------------------------------------
    Button("ac_yes", "AC_YES", ("AC",)),
    Button("ac_no", "AC_NO", ("Non-AC",)),
    Button("ac_both", "AC_BOTH", ("Show both", "Dono dikhaiye")),
    # ---- extra bed ---------------------------------------------------------
    Button("bed_yes", "BED_YES", ("Yes",)),
    Button("bed_no", "BED_NO", ("No",)),
    # ---- the change menu ---------------------------------------------------
    # Four options, one past the three-button cap, so these ship as list rows rather than
    # reply buttons (see `templates.change_options_menu`). A row's title comes back exactly
    # as a button's does, so they are declared here for the same reason.
    #
    # The titles are deliberately plain — no emoji, no "📅". The title *is* the token the
    # router matches, so decorating it would also decorate the thing a guest has to type,
    # and a row reading "📅 Stay Dates" only resolves if the guest reproduces the emoji.
    Button("change_dates", "CHANGE_DATES", ("Stay Dates",)),
    Button("change_guests", "CHANGE_GUESTS", ("Guests / Room count",)),
    Button("change_ac", "CHANGE_AC", ("AC / Non-AC",)),
    Button("change_meal", "CHANGE_MEAL", ("Meal Plan",)),
    # ---- the composition question (5+ guests) ------------------------------
    # Two rows, not the four a text menu used to print. The reader
    # (`understand.py`, `need_composition_preference`) stopped answering options 3 and 4 on
    # 2026-09-19 and says so in a comment, but the template went on printing them: a guest
    # tapping "3" was answering a question the bot had already stopped asking.
    Button("compose_max", "COMPOSE_MAX", ("Best price", "Kam kharcha")),
    Button("compose_base", "COMPOSE_BASE", ("Most comfortable", "Aaram se")),
)


BUTTONS_BY_ID: dict[str, Button] = {b.id: b for b in BUTTONS}

#: The actions `understand.py` dispatches directly from a tap.
#:
#: Declaring a button here is not the same as declaring it above, and the difference is
#: deliberate. Membership in this set means "the fast path handles it by itself, from the
#: text alone". A button is *declared* above and kept out of this set when something else
#: has to decide what its tap means — either because the answer is only meaningful while a
#: particular question is the active prompt, or because the meaning depends on state the
#: tap cannot carry:
#:
#:   * AC and extra bed — read by `ac_text` / `extra_bed_intent`; the AC answer only counts
#:     while the AC question is the active prompt, and the bed answer depends on the room row.
#:   * the change menu — read under `next_action == "change_menu_shown"`. Honour it anywhere
#:     else and a guest typing "Meal Plan" out of context jumps into a question they never
#:     asked for.
#:   * the composition rows — read under `need_composition_preference`, whose matcher also
#:     accepts free text ("kam kharcha", "budget 6000") that no title could enumerate.
#:
#: In all three cases the title is still validated by `button_card` / `list_card`, and a tap
#: is still *recognisable* — it just resolves at its own site rather than here.
TAP_ROUTED_ACTIONS: frozenset[str] = frozenset(
    {
        "SHOW_ROOMS",
        "BOOK_NOW",
        "CHECK_EXISTING_BOOKING",
        "HUMAN_REQUEST",
        "PAYMENT_PAID",
        "CONFIRM_BOOKING",
        "CANCEL_BOOKING",
        "CANCEL_ABORT",
        "NUDGE_CONTINUE",
        "NUDGE_CANCEL",
        "MEAL_EP",
        "MEAL_CP",
        "MEAL_MAP",
    }
)

#: Which meal plan each meal button selects. Declared here rather than derived from the
#: action with `split("_")`, because the plan codes are what the pricing engine keys on and
#: a derived value would fail silently if an action were ever renamed.
MEAL_PLAN_BY_ACTION: dict[str, str] = {
    "MEAL_EP": "EP",
    "MEAL_CP": "CP",
    "MEAL_MAP": "MAP",
}


# ============================================================================
# Lookup
# ============================================================================

_TITLE_INDEX: dict[str, Button] = {}


def _build_index() -> None:
    """
    Map every declared title to its button, refusing ambiguous ones.

    Two buttons claiming one title is not a style problem — it makes a tap have no single
    meaning, and whichever won would be decided by tuple order. Building the index loudly is
    the only place that can catch it, since the conflict is invisible at both call sites.
    """
    for button in BUTTONS:
        for title in button.titles:
            key = normalize_reply_text(title)
            existing = _TITLE_INDEX.get(key)
            if existing is not None and existing.id != button.id:
                raise ValueError(
                    f"Button title {title!r} is claimed by both {existing.id!r} and "
                    f"{button.id!r}; a tap on it would have no single meaning."
                )
            _TITLE_INDEX[key] = button


_build_index()


def resolve_button(text: str | None) -> Button | None:
    """
    The button a message is a tap on, or None if it is ordinary typed text.

    Titles are tried first because the title is what actually arrives. The id is tried
    second so the routing is already correct for the day the CRM stops dropping
    `button_reply.id` — and so a button keeps working if its wording is ever changed here
    without the sending side being updated.

    Returning None is a normal outcome, not an error: most messages are people typing.
    """
    normalized = normalize_reply_text(text)
    if not normalized:
        return None

    by_title = _TITLE_INDEX.get(normalized)
    if by_title is not None:
        return by_title

    return BUTTONS_BY_ID.get(normalized)


def collect_titles() -> dict[str, tuple[str, ...]]:
    """Every button id mapped to its declared titles. Read by the guard test."""
    return {b.id: b.titles for b in BUTTONS}
