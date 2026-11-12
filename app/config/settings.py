"""
Application settings loaded from environment variables.
Uses pydantic-settings for validation and type coercion.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration — all values come from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── MongoDB ──────────────────────────────────────────────────
    MONGODB_URI: str
    MONGODB_DATABASE: str = "stayinn_web"

    # ── Hotel ────────────────────────────────────────────────────
    HOTEL_ID: str = "111111"
    HOTEL_NAME: str = "our hotel"

    # ── LLM provider ─────────────────────────────────────────────
    #
    # Which model answers every text and image call. "gemini" is the long-standing
    # path; "deepseek" routes the same calls to DeepSeek's OpenAI-compatible API
    # (`deepseek-flash`, which carries vision natively).
    #
    # This exists because the Gemini key was leaked on 2026-10-22 and the project
    # hit its monthly spend cap on 2026-10-21, which silently killed every
    # model-backed feature for ~28 hours. A provider that can be swapped with one
    # environment variable is the difference between a one-minute fix and a
    # redeploy during an outage.
    #
    # Default stays "gemini" so the switch is a console change, not a code change,
    # and so rolling back is flipping the same variable.
    LLM_PROVIDER: Literal["gemini", "deepseek"] = "gemini"

    # ── Gemini ───────────────────────────────────────────────────
    #
    # Optional since the leak: a process running on DeepSeek must still boot when
    # no Gemini key is present. Selecting "gemini" without a key fails at the
    # first call with a message that names the missing variable, rather than at
    # import time with a pydantic validation error nobody can act on.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # ── DeepSeek ─────────────────────────────────────────────────
    #
    # Docs: api-docs.deepseek.com. `deepseek-flash` IS the vision model — there is
    # no separate multimodal id — so images, receipts and ID cards all go through
    # the same model as text.
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-flash"

    # Thinking mode is ON by default at effort "high" on DeepSeek, and while it is
    # on the API ignores `temperature` entirely. Every call in this bot is a short,
    # tightly-scoped extraction or reply where hidden reasoning is pure cost and
    # latency — and temperature=0.0 is load-bearing for the JSON extractors. So it
    # is off unless someone turns it on deliberately to compare quality.
    DEEPSEEK_THINKING: Literal["enabled", "disabled"] = "disabled"
    DEEPSEEK_REASONING_EFFORT: str = "low"

    # A turn that waits longer than this is a turn the guest has left. The
    # provider default is far longer than a WhatsApp conversation can tolerate.
    LLM_TIMEOUT_SECONDS: float = 45.0

    # ── ChromaDB ─────────────────────────────────────────────────
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    KNOWLEDGE_SIMILARITY_THRESHOLD: float = 1.2  # Max cosine distance for RAG confidence

    # ── App ──────────────────────────────────────────────────────
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # ── Rate Limiting ────────────────────────────────────────────
    MAX_MESSAGES_PER_MINUTE: int = 30
    # A guest's message is truncated to this before `understand` reads it.
    #
    # 500 was cutting real bookings in half. 2026-10-29, hotel Sample Property: a
    # guest pasted a complete brief — dates, room type, fare, and eight named
    # guests with ages — 600-odd characters. `sanitize_input` kept the first
    # 500, so the last line of the children section ("8) TRISHIKA SHRIVAS
    # Age:- 2.5 Year") never reached the graph. The bot counted 6 guests + 1
    # child for a party of eight and printed "Child (5 yrs)"; the missing child
    # was not an extraction failure, the text was simply gone.
    #
    # WhatsApp allows 65,536 characters. 4,000 covers any brief a hotel could
    # paste while keeping the prompt bounded — and the cap still exists, so a
    # pathological message cannot be pushed into the model whole.
    MAX_MESSAGE_LENGTH: int = 4000

    # How long a turn waits for the guest to finish typing, in seconds.
    #
    # Guests on WhatsApp send one thought as several messages ("28 october",
    # "Check in", "10 person with kids"). Each one used to be a full turn, so a
    # thought arrived as three replies — measurably the same question asked
    # three times, or two contradictory answers in the same second. This window
    # merges them: every message joins a per-conversation buffer, the turn
    # sleeps, and only the newest message runs the graph, over all of them
    # joined. Superseded turns return an empty response, which is the same
    # "say nothing" signal the takeover path already uses.
    #
    # Two windows, because the two kinds of message want opposite things:
    #
    # `MESSAGE_DEBOUNCE_SECONDS` applies to a message that reads as complete —
    # a full sentence, a button label, anything long enough to stand alone.
    # Delaying those makes the bot feel dead for no benefit.
    #
    # `MESSAGE_DEBOUNCE_FRAGMENT_SECONDS` applies to a short message, which is
    # what a guest sends mid-thought ("Non AC", "28 october", "Check in").
    # Measured on 137 consecutive guest messages (2026-10-18/19): only ~6% land
    # within 3 s of the previous one, but ~30% land within 8 s and almost all of
    # those are these short fragments. The long window is what makes the merge
    # actually fire; the short one is what keeps normal replies quick.
    #
    # Set both to 0 to disable the merge entirely (no delay, one turn per message).
    MESSAGE_DEBOUNCE_SECONDS: float = 3.0
    MESSAGE_DEBOUNCE_FRAGMENT_SECONDS: float = 8.0

    # ── Reply guard ──────────────────────────────────────────────
    # Strip a generated question when the state it would ask about is already
    # filled — measured on guest 919051141676 (2026-10-18): "What is your
    # check-in date?" was sent four times after check-in had been recorded as
    # 28/10. See app/services/reply_guard.py for the exact rule and why it is
    # a deterministic filter rather than a prompt instruction.
    #
    # This runs on every turn (`save_state`), so it is the integration point
    # most worth being able to turn off alone: set False and redeploy.
    REPLY_KNOWN_DATA_GUARD_ENABLED: bool = True

    # ── Session ──────────────────────────────────────────────────
    SESSION_TTL_HOURS: int = 24

    # How many previous messages of the conversation are handed to the model —
    # both as `RECENT CONVERSATION` in the understand prompt and as
    # `messages_history` in state.
    #
    # The single place to change it. It used to be a literal `10` in two
    # callers, which is half of what `save_state` keeps (`session.messages[-20:]`),
    # so the bot was reading half of what it had saved. Raised to 20 on
    # 2026-09-23 to match the store rather than leave the two numbers to drift.
    HISTORY_WINDOW_MESSAGES: int = 20

    # How many of a stale session's turns to carry into the fresh one it is reset to.
    # 0 restores the old behaviour, which was to carry none.
    #
    # The reset itself is deliberate — an abandoned draft must not be resurrected by
    # a guest returning days later — but it was also throwing away the *conversation*,
    # so the waking bot had no idea what had already been discussed and asked for the
    # dates and the party all over again. Measured 2026-11-10: `STALE_SESSION_RESET`
    # fired for guests in COLLECTING, ROOMS_SHOWN, AWAITING_PAYMENT and NEW, several
    # of them inside 48 hours. Only the transcript is carried; every booking field
    # still resets, which is what keeps a dead quote from being answered as if live.
    STALE_SESSION_KEEP_RECENT_MESSAGES: int = 12

    # ── History labels ───────────────────────────────────────────
    # The conversation reaches the model as plain text, and every line is
    # prefixed with who said it. The prefix used to be `[GUEST]` for
    # everything: `sender_type` was added to `MessageRecord` after `role` with
    # a default of "guest", and every writer left it there — so all 7,632
    # stored bot replies and 94 staff notes read as the guest (measured
    # 2026-10-20 against the live conversation store). The model could not
    # tell a question from its answer, which is what made a second WhatsApp
    # message ("28 october") unreadable and left the bot re-asking.
    #
    # `MessageRecord` now derives the sender from `role` when the row does not
    # already say otherwise, so the stored rows repair themselves on load. This
    # flag gates everything the model sees: the label on every line and the
    # prompt rules that resolve a short message against the bot's last
    # question. Set False and the transcript is every-line-`[GUEST]` again,
    # exactly as before 2026-10-20. The stored `sender_type` repair is not
    # gated — it is inert on its own, and reverting it would need a migration.
    #
    # The flag is a default, not a live switch: like TAKEOVER_RESUME_ENABLED
    # below, flipping it on Cloud Run needs `iam.serviceaccounts.actAs`, which
    # the operator account does not hold. The real undo is a revert commit.
    UNDERSTAND_HISTORY_LABELS_ENABLED: bool = True

    # ── Human Takeover ───────────────────────────────────────────
    # Inactivity window after the latest staff message before AI resumes.
    #
    # 300 (5 min), lowered from 600 on 2026-10-08 to match the CRM. The CRM's
    # `ManualOverride.expires_at` is `now + timedelta(minutes=5)` at both of its
    # creation sites (`api_gateway/views/messages.py`, `webhook_receiver/views.py`),
    # so it stops withholding guest messages at 5 minutes. While this was 600 the
    # two disagreed: a guest writing at minute 6 got past the CRM and was then
    # paused *again* by the bot, and heard nothing for another four minutes.
    # One number, on both sides.
    TAKEOVER_TIMEOUT_SECONDS: int = 900  # 15 minutes; keeps the CRM and the bot agreed

    # ── Staff price interpretation ───────────────────────────────
    # Whether the staff-decision extractor is asked to report each quoted number
    # as a *unit* ("1600 per room per night") and let code do the arithmetic,
    # instead of trusting the model's own `payable_total`.
    #
    # The complaint this exists for, twice reported: a room quoted at ₹X, and the
    # bot wakes up quoting the whole stay at ₹X. Two guards have already been
    # bolted on after the fact — `_corroborate_payable_total` and
    # `_number_is_framed_as_a_rate` — and both are regexes deciding a semantic
    # question. Asking the model what the number *is* and multiplying in Python
    # makes the bad outcome unrepresentable rather than merely unlikely.
    #
    # ON by default: the operator account cannot set Cloud Run environment
    # variables (no `iam.serviceaccounts.actAs`), so a default-OFF flag would be
    # a flag nobody can turn on. Turning it off restores the old behaviour, in
    # which the model's `payable_total` is used as-is and the regex guards are
    # the only protection.
    STAFF_PRICE_INTERPRETATION_ENABLED: bool = True

    # ── Context Refresh ──────────────────────────────────────────
    # Whether `POST /api/admin/context-refresh` extracts *typed* staff decisions
    # (pricing, payment, amenities, booking) and applies them as conversation
    # overrides, instead of writing a free-text summary note.
    #
    # It changes what a guest can be charged: with the flag on, a price staff
    # agreed in the takeover becomes the number the quote and payment paths read.
    # A note never did — `prompt_summary()` does not render `notes`, so a summary
    # saying "₹3,000" left the engine quoting its own figure, and the bot
    # contradicted itself between prose and the payment link.
    #
    # ON as of 2026-09-28. It is a *default*, not a constant: pydantic-settings
    # gives the environment precedence, so setting
    # CONTEXT_REFRESH_STRUCTURED_DECISIONS=false on the Cloud Run service turns it
    # back off without a deploy. That route needs `iam.serviceaccounts.actAs`,
    # which this project's operator account does not hold — so in practice the
    # working undo is a revert commit, which Cloud Build deploys under its own
    # service account. Flipping the default here is how it was turned on.
    CONTEXT_REFRESH_STRUCTURED_DECISIONS: bool = True

    # ── Direct booking discount ──────────────────────────────────
    # Whether the bot may answer "any discount?" by granting one itself — the
    # "Exclusive Direct Booking Privilege" ladder in `price_match_handler`:
    # 5% off now, another 5% on the second ask, capped by the season's
    # negotiation limit.
    #
    # OFF as of 2026-10-18 by the client's decision — there is no direct
    # booking discount. With it off the handler still validates a competitor
    # screenshot (a real price match), but it never reduces the price on its
    # own turn and never promises a percentage.
    #
    # It rewrote `total_price`, `amount_due` and `discount_amount` directly,
    # so it moved the number a guest is charged. That is why the code stays
    # behind a flag instead of being deleted: if the policy comes back, the
    # undo is one environment variable rather than a revert commit.
    DIRECT_BOOKING_DISCOUNT_ENABLED: bool = False

    # ── Mixed room requests ─────────────────────────────────────
    # Whether "and 1 ac room", written after the guest already picked a
    # non-AC room, is read as the second half of a split instead of a change
    # of mind. On, the custom-room reader gets both halves and prices them
    # together; off, the turn behaves as it did before — the earlier pick is
    # wiped and the card is redrawn, which is what guest 919088198242 saw on
    # 2026-10-19.
    AUTO_CUSTOM_ROOMS_FOR_ADDITIVE_ROOM: bool = True

    # ── Start Fresh ──────────────────────────────────────────────
    # Whether an explicit "start fresh" wipes the draft even for a guest whose
    # *stay* is already confirmed.
    #
    # Why it needs a switch at all: the reset lives in `apply_intent_overrides`,
    # and `understand` answers a confirmed guest from a branch that returns
    # before that function is ever called. So the reset was unreachable for
    # exactly the guests most likely to want one — measured 2026-10-17, a
    # "/start fresh" logged `STAGE_RESOLVED_FROM_CRM | confirmed stay, no booking
    # row of ours` then `STAGE_2_CONFIRMED_DISPATCH`, no reset, no override line.
    #
    # The reservation is never touched either way — only the session state is
    # dropped, so the guest can start a new enquiry while their booking stands.
    # Turning this off restores the old behaviour, where such a guest keeps the
    # confirmed-stage replies.
    #
    # ON as of 2026-10-17. As with the flag above, the environment wins over this
    # default, but this project's operator account cannot mutate the Cloud Run
    # service, so the working undo is a revert commit.
    START_FRESH_RESETS_CONFIRMED_STAY: bool = True

    # ── Post-stay feedback window ────────────────────────────────
    # How long a completed stay may keep the guest in `checked_out` — the stage
    # that asks for a 1–5 rating and answers everything else with a thank-you.
    #
    # The resolver used to return `checked_out` for every booking whose check-out
    # had passed, with no upper bound. Because a completed stay outranks a new
    # enquiry, any guest who had ever stayed was pinned to the feedback stage for
    # the life of the property. Measured on prod 2026-10-22 for phone 9958263793:
    # `/start fresh` dismissed one past booking, the guest's *older* past booking
    # took over the stage on the next turn, and every message since — `hi`,
    # `/start`, `menu_rooms` — was answered with the same rating prompt.
    #
    # Past the window the stay no longer sets the stage: the guest resolves to
    # `lead`, so a returning guest is greeted as a lead rather than asked to rate
    # a stay that ended a week ago. Set this very high (e.g. 876000) to restore
    # the old "a past stay owns the guest forever" behaviour; the operator account
    # cannot mutate Cloud Run, so the working undo is a revert commit.
    CHECKED_OUT_FEEDBACK_WINDOW_HOURS: int = 48

    # ── Start fresh: a booking row of ours ───────────────────────
    # Whether "start fresh" also dismisses a booking the bot itself created.
    #
    # The flag above covers a stay staff confirmed in the CRM, which never lands in
    # our `bookings` collection. A guest who booked *through the bot* has a row, and
    # `resolve_guest_stage` reads that row on every turn — so without this their
    # reset dropped the draft and nothing else, and the next message came back
    # confirmed. Measured on prod 2026-10-17: `/start fresh` saved `Stage: new`, then
    # `book room , tomorrow 3 people` resolved `stage=confirmed` again off
    # BK-20261016-E7B4FA89.
    #
    # Scoped to the booking id the guest had at the moment they asked, so a later
    # booking is a different id and is honoured normally. The reservation itself is
    # never touched — this changes what the bot treats as the active stay, not the
    # booking. Turning it off restores the old behaviour.
    START_FRESH_DISMISSES_BOOKING_ROW: bool = True

    # ── Start fresh: the whole phone number ──────────────────────
    # Whether "/start" also forgets every booking the number had at that moment,
    # not just the one row that set the stage this turn.
    #
    # The two flags above name a stay (a CRM check-out date, or one booking id).
    # That misses two cases that both happened on the bossman's own account on
    # 2026-10-23: a phone can hold more than one row, and the row that owned the
    # stage can be one the guest did not mean (his was BK-6E14274A — created
    # 2026-06-06, no dates, no rooms, a Rs 50,000 total). Either way the reset
    # lasted one message and the next "hi" came back `confirmed`.
    #
    # This records a timestamp instead of an id: `resolve_guest_stage` drops every
    # row created at or before it. A booking made *after* the reset has a later
    # `created_at` and is honoured normally, so the guest is never left blind to
    # a genuine new stay. The reservation is still never touched — nothing is
    # deleted, this only changes which rows name the guest.
    #
    # ON as of 2026-10-23. The operator account cannot mutate Cloud Run, so the
    # working undo is this flag in the environment or a revert commit.
    START_FRESH_FORGETS_ALL_BOOKINGS: bool = True

    # ── Confirmation slip with nothing to show ───────────────────
    # Whether the confirmation slip drops the lines it has no data for.
    #
    # A `confirmed` row can carry no stay at all. BK-6E14274A did, and the slip
    # printed "Dates: None se None", "Guests: 0 Guest(s)" and "Balance Due at
    # Check-in: Rs 50,000" — a total with no nights behind it, which is what made
    # a real guest ask "how 50,000?". With this on, missing dates also suppress
    # the money lines: nothing is claimed that the row cannot support. The
    # booking id is still printed, because that part is true.
    CONFIRMATION_SLIP_HIDES_UNKNOWN_STAY_FIELDS: bool = True

    # ── Confirmation slip payment figures ────────────────────────
    # Whether the slip reads the amount actually paid from the payments ledger
    # before printing it, and refuses to claim "fully paid" when it knows nothing.
    #
    # `amount_paid` defaulted to `total_price` in the builder, so a row that never
    # recorded a payment printed "Amount Paid: <full total>" and "Balance Due: 0
    # (Fully Paid)". Guest 917738662309 (2026-11-01) paid Rs 1,110 against a
    # Rs 3,700 booking and got a slip saying it was fully paid; 15 of the 19 slips
    # sent on the clone did this. Off restores the old default for one release.
    CONFIRMATION_SLIP_VERIFIES_PAID_AMOUNT: bool = True

    # ── Group bookings (5+ rooms / 15+ adults) ───────────────────
    # Whether the bot sells a large party itself instead of handing it to the
    # front desk.
    #
    # T6 (2026-09) capped self-serve at four rooms so that anything bigger became
    # a "our reservations team will assist you directly" card. Two things went
    # wrong with that in production:
    #
    #   1. The card repeated on *every* turn, because the guard that sends it is
    #      the same guard that blocks the search — nothing in the flow ever set
    #      `group_lead_submitted`, so there was no way out of the loop.
    #   2. It was a dead end, not a handoff. The guest was asked for contact
    #      details they had already given (their WhatsApp number is the session
    #      id) and, on 2026-10-17, a real 7-room / 17-adult enquiry on 9064956832
    #      received the same paragraph four times and no quote at all.
    #
    # ON: the party is searched and priced like any other, with T6's four-room
    # ceiling lifted in `allocation_engine`. OFF: the old handoff, but sent once
    # per conversation rather than every turn.
    GROUP_BOOKING_SELF_SERVE: bool = True

    # ── Takeover Resume ──────────────────────────────────────────
    # Whether the bot *speaks* when a human takeover ends, instead of only
    # remembering it.
    #
    # Before this, `POST /api/admin/context-refresh` summarised the staff
    # conversation and wrote it into the session, then returned — it never
    # composed or sent anything, so a guest sitting in a paused chat heard
    # silence until they happened to write again. The CRM does call the endpoint
    # when its override expires (`run_override_scheduler`); nothing came of it.
    #
    # ON as of 2026-10-08, and the default is the only working switch: flipping
    # the Cloud Run env var needs `iam.serviceaccounts.actAs`, which the operator
    # account does not hold. The real undo is a revert commit. This sends a real
    # message to a real guest, so treat it accordingly.
    #
    # OFF as of 2026-10-27, by product instruction: the CRM's override
    # scheduler calls `/api/admin/context-refresh` roughly every five minutes,
    # and each call ended in an unprompted message to the guest. The team does
    # not want that follow-up. Context updates — summary, staff decisions,
    # `pending_context` — still happen; only the "speak" half is gated here.
    #
    # ON again 2026-11-10, by product instruction, with the cause fixed rather
    # than tolerated. The repeat sends were not the composer being chatty: the
    # CRM rebuilds its window on every refresh, so `_bot_spoke_since` — which
    # compares against the window's own start — lapsed as soon as a newer start
    # time arrived, and the same unanswered guest turn was answered again. The
    # hand-back now carries `takeover_resume_fingerprint` on the session and
    # refuses to answer the same run of guest messages twice, so the five-minute
    # loop is idempotent. It also still refuses when staff had the last word, so
    # a guest who has already been answered is not talked over. Set False to
    # silence the hand-back again; the context write is unaffected.
    TAKEOVER_RESUME_ENABLED: bool = True

    # ── GST ──────────────────────────────────────────────────────
    # A single rate applied to every property, overriding both the per-hotel
    # `taxes` slabs and the built-in Rs 7,500 / 5% / 18% fallback.
    #
    # This exists because per-property GST is not representable in the database
    # today: all four `hotel_settings` docs carry `tax = {"gst": "", "pan": ""}`
    # (an empty GSTIN) and `tax_config.slab = "include"`, which is inclusive-vs-
    # exclusive *wording*, not a rate. Nothing anywhere stores a percentage, so
    # every property was priced by the same hardcoded slab table.
    #
    # It sat at 5.0 from 2026-09-29 to 2026-09-30, which flattened *every* room
    # to the low band — including the ones at Rs 7,500 a night and above, which
    # have been taxed at 18% since the September 2025 reform. product ruling on
    # 2026-09-30: Sample Hotel has no room at Rs 7,500 or more, so the low band
    # covers all of what it sells, and the high band has to be free to bite for
    # the properties that do sell above the threshold.
    #
    # Pricing is GST-**inclusive**, so the rate does not change what a guest
    # pays — it changes only how the invoice splits between taxable value and
    # tax. A too-low rate under-remits GST and overstates revenue; it never
    # overcharges the guest.
    #
    # `None` = per-hotel slabs and the built-in Rs 7,500 / 5% / 18% fallback,
    # which is the correct behaviour and the state this is in now. Set a number
    # only to pin every property to one rate again. The real fix is a
    # per-property rate in `hotel_settings` that `booking_engine`'s `taxes`
    # reader can pick up.
    #
    # Whatever this is set to, `price_match_service` reads it too — check its
    # fallback before changing it, because that one is only reached when this is
    # `None`.
    GST_RATE_OVERRIDE_PERCENT: float | None = None

    # ── Extra-person charge ──────────────────────────────────────
    # Whether an adult above a room's base occupancy is charged the extra person
    # rate on a CP / MAP / AP plan, or only ever when a mattress was asked for.
    #
    # Product, 2026-10-06: *"On EP Plan we can allow extra person to stay in room
    # without any extra bed ... EP can have NO EXTRA BED (no extra charges) but
    # that SHOULD NOT be allowed on MAP and CP."*
    #
    # The old rule required the guest to have asked for a bed: both charge sites
    # read `if takes_bed and wants_bed` (`allocation_engine.py:578`) and
    # `if wants_bed and bed_allowed` (`:817`). So a guest on a **CP** plan who
    # said "no extra bed" was charged nothing at all for the third adult who ate
    # breakfast in that room — the plan they bought covers that guest, and the
    # bot gave it away. On **EP** the same silence is correct: the room-only rate
    # buys the room, and a guest who takes no bed takes nothing extra.
    #
    # With this on, `app/utils/meal_plan.charges_an_extra_person` gates the
    # charge on the plan's own type — read from `rate_type_id` where the
    # catalogue states it, falling back to the plan name. The money is still the
    # room type's `extra_adult_rate`: the *permission* is a plan property, the
    # *price* a room-type one.
    #
    # A plan we cannot classify charges nothing above base, so an unknown plan
    # can never start billing guests. The charge also lands on its own field
    # (`extra_bed_count` stays 0) so the quote can say "Extra person" rather than
    # claiming a bed the guest was never offered.
    #
    # `False` restores the bed-only charge exactly, for a one-push rollback.
    EXTRA_PERSON_CHARGED_ON_CP_MAP: bool = True

    # ── Paid early check-in / late check-out ─────────────────────
    # Whether the bot quotes the hotel's configured fee when a guest states an
    # arrival or departure time outside the standard ones.
    #
    # The hotels already sell this. The Control Centre page exists, operators use
    # it, and hotel 111111 has a live `early_late_checkout` row (`27% / 42% /
    # Rs 1502`) in production — but nothing in this repo read it, so a guest who
    # said "I'll reach by 7am, can I check in early?" got no price and no policy.
    # `app/services/early_late_checkout_service.resolve_early_late_fees` now
    # reads it and `booking_engine.format_price_quote` prints the fee as its own
    # line, with the disclaimer attached to the number.
    #
    # This flag is **not** the per-hotel switch and must not be treated as one.
    # That switch is `is_active` inside each hotel's own document, which the
    # Control Centre writes; a hotel with no row is off. This boolean exists only
    # so a defect in the fee path can be shut off in one push without editing
    # Mongo. Note it therefore cannot be flipped from a developer machine — Cloud
    # Run env changes need `iam.serviceaccounts.actAs` — so its real rollback is
    # the revert commit that carries it.
    #
    # The fee is charged ONCE for the stay, as a percentage of one night's room
    # rate (a rate that already includes GST, so the fee is never taxed again),
    # and never on a quote the guest did not ask a time for.
    #
    # `False` restores the old behaviour exactly, for a one-push rollback: no
    # line, no number, no disclaimer.
    EARLY_LATE_CHECKOUT_FEE_ENABLED: bool = True

    # Removed 2026-10-10: `EARLY_LATE_ASK_ENABLED` and the post-selection
    # question it gated. The bot does not raise the subject — product rule,
    # same day. The guest raises it, and `intent_overrides` answers it: a stated
    # clock is priced, and a mention with no clock gets the hotel's own policy
    # line and a request for the time.

    # ── Rate threshold protection ────────────────────────────────
    # Whether the bot bounds a computed nightly rate by the hotel's own floor and
    # ceiling, read from `hotel_settings.default_rate_threshold` (the Control
    # Centre's "Rate Threshold Protection" page).
    #
    # The page has existed for a while and one hotel has it configured in
    # production (`min 0.73 / max 5.57`), but nothing in this repo read it — the
    # number a guest was charged was unbounded by it. The one modifier on a
    # nightly rate that has no ceiling of its own is the seasonal multiplier
    # (`season_repository` floors it at 0.50 and caps it at nothing), so this is
    # the leash on the season. `app/utils/rate_threshold.py` holds the rule and
    # the two pricing paths both apply it.
    #
    # This flag is **not** the per-hotel switch. That switch is the hotel having
    # a usable window at all; a hotel with no row is not clamped, so the blast
    # radius of this feature today is the one hotel that has one. This boolean
    # exists only so a defect can be shut off in one push without editing Mongo.
    # Like every flag here it **cannot be flipped from a developer machine** —
    # Cloud Run env changes need `iam.serviceaccounts.actAs` — so its real
    # rollback is the revert commit that carries it.
    #
    # `False` restores the old behaviour exactly: every rate as the modifiers
    # left it.
    RATE_THRESHOLD_PROTECTION_ENABLED: bool = True

    # ── Room card ────────────────────────────────────────────────
    # Whether a child the hotel's own child policy makes FREE is absent from the
    # berths the room card packs for, while still counting toward the heads a
    # room must hold.
    #
    # A room has two ceilings and they are different numbers: `max_adults`, how
    # many adults it sleeps, and `max_occupancy`, how many people the hotel will
    # put in it. A free child shares the parents' bed, so it needs no berth of
    # its own and is still a person in the room.
    #
    # On 2026-09-29 this was off, so the card packed a party of five adults and a
    # free 7-year-old as six berths and quoted two King rooms at Rs 3,600 —
    # when a single Deluxe Room Four Bed, which seats 5 and holds 6, cost
    # Rs 3,400. The search had already found that room and offered it as its
    # cheapest option; only the card could not see it.
    #
    # `False` restores the old packing exactly, for a one-push rollback.
    FREE_CHILD_NEEDS_NO_BERTH: bool = True

    # ── Selection round-trip ─────────────────────────────────────
    # Whether a chosen set of rooms records the catalogue's own ceilings on
    # every `SelectedRoom`.
    #
    # `validator` has always *read* `max_adults` and `max_occupancy` off a
    # selected room (`validator.py:282-283`), but `SelectedRoom` never carried
    # either field, so the read always fell through to
    # `assigned_adults + assigned_children`. For the combination allocator that
    # fallback is the distribution it just built and it is correct; for the
    # requirements allocator it is the room's *base* occupancy, which is a
    # different number — a Deluxe King Bed is recorded there as 2 seats when the
    # room seats 3.
    #
    # Measured 2026-09-29, hotel 111111, six adults: the guest asked for
    # "2 Delux room king bed", the search offered `2 x King` at Rs 3,600, and
    # replying "1" drew
    # `VALIDATOR_SELECTION_MISMATCH: capacity=4/6 ... Triggering fresh search`
    # — forever, because two Kings recorded 4 seats against a party of 6 and the
    # validator threw the selection away and redrew the identical card.
    #
    # `False` restores the fallback exactly, for a one-push rollback.
    SELECTION_RECORDS_ROOM_CEILINGS: bool = True

    # Whether the AC / Non-AC question is suppressed when the hotel's own room
    # configuration says there is no non-AC room left to choose.
    #
    # The question is asked only when the search can show an AC option and a
    # Non-AC option. "Non-AC" was decided by the room's NAME, and a name does
    # not update itself when the hotel fits AC to a room: Sample Hotel's
    # "STANDARD ROOM (NON-AC)" has Air Conditioning configured on its row, so
    # the bot asked a question whose second answer the hotel no longer sells
    # (measured 2026-11-03, hotel 111111).
    #
    # With this on, a non-AC candidate that is *explicitly configured* as AC is
    # no longer evidence of a choice. A room whose amenities are silent keeps
    # its name as the only evidence, so hotels with an empty amenity
    # configuration ask exactly as before.
    #
    # `False` restores the ask-always-on-name behaviour, for a one-push rollback.
    AC_QUESTION_HONOURS_ROOM_CONFIG: bool = True

    # Whether the bot asks "AC or Non-AC?" at all.
    #
    # OFF for now (Product, 2026-11-03). At Sample Hotel the guest was being asked a
    # question whose two answers lead to the same rooms, and the cheapest rooms
    # never reached an assumed-AC quote — so the question cost a turn and bought
    # nothing. With it off the bot assumes AC, says so in the footer
    # ("Assuming: AC, Room Only"), and the guest can still correct it with
    # "change" or by saying "non-AC" themselves; an explicit non-AC request is
    # honoured exactly as before, because that path reads the guest's words and
    # never goes through this gate.
    #
    # `True` turns the question back on (subject to
    # `AC_QUESTION_HONOURS_ROOM_CONFIG` above, which still suppresses it at a
    # hotel with no non-AC inventory).
    ASK_AC_PREFERENCE: bool = False

    # Whether a selection made from the room card leaves `room_type_preference`
    # alone.
    #
    # The SELECT_ROOM branches write the matched option's `display_label` —
    # "2x DELUXE ROOM KING BED" — into `room_type_preference`, which is the field
    # the search reads as a room *type*: `room_searcher.py:73` →
    # `room_service.py:143` → `filter_rooms_by_reference`. So choosing a row
    # silently pinned every later search to the rooms that one set happened to
    # contain, and overwrote whatever the guest had actually asked for.
    #
    # Measured 2026-09-29 on the same transcript: after "child is 15 years old"
    # moved the party to six adults, the candidate pool held `DELUXE ROOM FOUR
    # BED` and nothing else, so `2 x King` at Rs 3,600 could not be offered
    # against `2 x Four Bed` at Rs 6,800. The selection is already recorded in
    # `selected_rooms`; the preference belongs to the guest's own words.
    #
    # `False` restores the write exactly, for a one-push rollback.
    SELECTION_DOES_NOT_PIN_ROOM_TYPE: bool = True

    # Whether a change to the *party* drops a `room_type_preference` the guest
    # stated for the old one.
    #
    # A preference is pinned against a party size. "A deluxe room" means a
    # different room for two people than for six, and one the guest named for
    # five says nothing at all about six — yet it survived the change and went on
    # narrowing every search that followed, which is how a party that grew by one
    # child kept being quoted the one room type it had already outgrown.
    #
    # Measured 2026-09-29 (hotel 111111): "child is 15 years old" moved the party
    # from five adults and a child to six adults. The preference carried over as
    # `Deluxe Room Four Bed`, so the candidate pool for the next search held that
    # room and nothing else, and `2 x Deluxe Room King Bed` at Rs 3,600 could not
    # be offered against `2 x Deluxe Room Four Bed` at Rs 6,800.
    #
    # **Only a *stale* preference.** A guest who restates one in the same message
    # as the new party — "6 adults, we want the king" — has that preference in
    # `changed_fields` too, and it is theirs and current, so it is kept. This
    # clears only what was carried over from an earlier turn.
    #
    # `False` restores the carry-over exactly, for a one-push rollback.
    PARTY_CHANGE_DROPS_ROOM_PREFERENCE: bool = True

    # ── Several bookings in one conversation ─────────────────────
    # One message can name two stays ("2 deluxe 20-22 Dec, 1 family 23-25
    # Dec"). The queue has held both since 2026-11 — detection, per-stay
    # allocation and per-stay payment all existed — but only stay one was ever
    # *collected*: the guest picked a room, got a quote and a payment link, and
    # stay two was forgotten until after that payment cleared.
    #
    # With this ON, a selection for a non-last stay is stashed on its queue
    # entry and the conversation moves to the next stay's card; the last one
    # closes with a single quote naming every stay and a grand total. Payment
    # stays one link per booking, as it already was.
    #
    # ON from 2026-11-09 at product instruction. Set it to False in the
    # environment for a one-push rollback: with it off `multi_stay` is never
    # entered and the one-stay-at-a-time flow is byte-for-byte what it was.
    MULTI_STAY_COMBINED_ENABLED: bool = True

    # ── Confirmed bookings ───────────────────────────────────────
    # Whether a session whose booking is already confirmed stops trying to
    # sell after that.
    #
    # Product, 2026-09-30: *"When booking confirmed, stop the bot on that chat
    # (it keeps saying book room etc, make it faq based i guess but if thats a
    # big change in code dont do it, jsut stop the bot)"*.
    #
    # The confirmed router in `understand.py` handled a new-booking request and
    # a greeting, and let everything else fall through to the general flow —
    # which re-runs the room search and the upsell on a guest who has already
    # paid.
    #
    # This is the smaller of the two options he offered: a message naming a
    # side topic (amenities, directions, check-in, invoice, or simply ending in
    # "?") still falls through to the amenity / location / FAQ branches, and
    # anything else gets one closing reply instead of a room card.
    #
    # `False` restores the fall-through exactly, for a one-push rollback.
    CONFIRMED_BOOKING_STOPS_THE_BOT: bool = True

    # ── Confirmation slip on a screenshot payment ────────────────
    # Whether verifying a UPI screenshot in chat also sends the guest the full
    # booking confirmation slip.
    #
    # Three routes already confirm a booking and send the slip: the Razorpay
    # webhook, staff confirmation in the Control Centre, and the nightly
    # reconciliation sweep. A fourth — the guest photographing a UPI receipt,
    # which is how Sample Hotel's bookings are actually paid — confirmed the
    # booking and sent nothing. Measured 2026-10-11 on `BK-20261010-94330B97`
    # (Sample Hotel, 11-12 Oct, ₹460 of ₹1,550): the guest's entire confirmation
    # was "Payment of ₹460 verified successfully! 🎉", with no dates, no room, no
    # address; the last slip of any kind reached any guest on 2026-09-01, and
    # only 5 of 412 bookings have `confirmation_sent`.
    #
    # This does NOT replace the acknowledgement line the guest gets today — the
    # slip is sent on top of it, so a slip failure costs them nothing they had.
    #
    # `False` stops the slip being sent from the screenshot path and leaves the
    # acknowledgement line exactly as it is, for a one-push rollback. Honest
    # caveat, same as the other flags here: this is a Cloud Run env var, so it
    # cannot actually be flipped from a machine without
    # `iam.serviceaccounts.actAs` — the real rollback is a revert commit.
    CONFIRMATION_SLIP_ON_SCREENSHOT_PAYMENT: bool = True

    # ── UPI screenshot auto-verification ─────────────────────────
    # Whether a payment screenshot alone can confirm a booking.
    #
    # Product, 2026-10-16: *"turn off UPI payment auto verification."*
    #
    # With this on, `validate_payment` (`app/services/vision/validator.py`)
    # reads the amount out of the guest's screenshot, decides it matches the
    # balance, writes `payment_status: paid` and calls
    # `confirm_pending_booking` — a booking confirmed by a picture, with no
    # human anywhere in the loop.
    #
    # With it off, arms A and B of the financial evaluation are skipped
    # entirely: the screenshot is **queued as a front-desk task**, the guest is
    # told the receipt has been received and the team will confirm shortly, and
    # **nothing is recorded as paid** — no `amount_paid`, no `payment_status`,
    # no confirmation. Only staff moving the task to COMPLETED confirms.
    #
    # The duplicate-UTR, tampering, FAILED/DECLINED and unreadable-amount
    # rejections all still run *before* this gate, so a screenshot the old code
    # would have rejected is still rejected, with the same wording.
    #
    # `True` restores auto-confirmation from the screenshot path exactly as it
    # behaved before. Same Cloud Run caveat as the flags above: pushing is the
    # only thing that moves this, so the rollback is a revert commit.
    UPI_SCREENSHOT_AUTO_VERIFICATION: bool = False

    # ── Confirmation writes an honest ledger ─────────────────────
    # Whether staff completing a payment task is taken as proof that the whole
    # booking total arrived.
    #
    # The bug, measured on `BK-20261011-6C18E226` (guest 917017478117) on
    # 2026-10-16: a ₹4,000 advance against a ₹15,300 booking. The bot correctly
    # told the guest "Total Paid: ₹4,000, Balance Due at check-in: ₹11,300" at
    # 13:03:29. Staff completed the payment task at 13:03:59, and at 13:06:57
    # the same bot said the ₹15,300 was fully paid — then, after a further
    # ₹2,000 screenshot, told the guest they had *overpaid* by ₹2,000 when in
    # fact they still owed about ₹5,000.
    #
    # Cause: `app/api/routes/tasks.py` substituted the booking's `total_price`
    # for the amount whenever staff did not state one, and passed it as
    # `amount_paid`. Confirming a booking is not evidence that money arrived.
    #
    # `False` records an amount only when the staff request actually carried one
    # — `UpdateTaskStatusRequest` has no such field, so the generic status route
    # records none, and `confirm-payment` records whatever `amount_paid` it was
    # given. `True` restores the substitution exactly.
    STAFF_CONFIRM_ASSERTS_FULL_PAYMENT: bool = False

    # Whether `set_booking_confirmed` derives `payment_status` from the money on
    # the record, instead of hardcoding `"paid"`.
    #
    # Same booking: the row ended up with `payment_status: "paid"` and
    # `amount_due: 11300` at the same time — a self-contradicting record, and
    # three different answers for the same fact across `bookings`,
    # `workflow_state` and the confirmation slip.
    #
    # `True` writes `"paid"` only when the recorded money actually covers the
    # total, and `"partial"` otherwise — which is what the sibling
    # `confirm_pending_booking` (used by the payment-screenshot path) already
    # does. It also writes the matching `amount_due` when a total is known, so
    # the two fields cannot disagree.
    #
    # `False` restores the hardcoded `"paid"`. Note the reach: this function is
    # also the webhook and reconciliation confirm path, and
    # `app/workflows/nodes/payment/link.py:160` branches on
    # `payment_status != "paid"` when deciding whether to reuse an existing
    # payment link — so a partial payment that used to read `"paid"` now reads
    # `"partial"` and may be issued a fresh link. That is the intended reading,
    # but it is the part to watch if a guest reports a duplicate link.
    CONFIRM_WRITES_HONEST_PAYMENT_STATUS: bool = True

    # ── Dashboard compatibility fields ───────────────────────────
    # Whether a booking write also carries the PascalCase field names the
    # StayInn dashboard's aggregate still sums (`$Guests`, `$Adults`,
    # `$Children`, `$Pricing.total`, `$BookingState`, `$RoomType`).
    #
    # The dashboard reads the same `stayinn_web.bookings` collection the bot
    # writes — measured 2026-10-27, it found all 319 of hotel 111111's bookings
    # but reported 0 guests, 0 adults, ₹0 revenue and empty breakdowns, because
    # it sums keys we spell in lowercase. See
    # `app/repositories/booking_panel_fields.py` for the six mappings.
    #
    # Our lowercase fields stay the source of truth; these are derived at write
    # time and nothing in the bot reads them. `False` stops the writes, at
    # which point the dashboard goes back to reporting zeros — the state before
    # this change. Flip this off first if a dashboard number ever looks wrong.
    PANEL_BOOKING_FIELD_ALIASES_ENABLED: bool = True

    # ── Payment-turn understanding ───────────────────────────────
    # Whether the model gets a say in whether a guest's message is a decision
    # to pay, or a request to change something first.
    #
    # The rule it replaces is `_NEGOTIATION_OR_CHANGE` in
    # `app/utils/intent_overrides.py` — an eleven-word blocklist. It exists
    # because the payment step once fired on "Thoda kam karo", "Actually show me
    # non AC rooms instead" and "STOP". It now catches those three, and it
    # catches them because `actually\s+show` was added to the regex by hand
    # after the fact. The next change request phrased with different words
    # ("make it two rooms instead") goes to payment again.
    #
    # With this on, `classify_payment_turn` (gemini_service) answers
    # CONFIRM / CHANGE / UNCLEAR for the turn, and both payment gates
    # (`routing.route_by_intent` and `validator`) prefer that answer.
    #
    # **The classifier can only agree or abstain.** UNCLEAR, a timeout, a
    # missing key, a malformed reply, or the flag being off all fall back to
    # the regex — so a Gemini outage cannot open the payment step to a message
    # the regex would have held back, and cannot route a real confirmation into
    # the booking flow. Every disagreement between the two is logged.
    #
    # `False` restores the regex-only behaviour exactly, for a one-push
    # rollback, and skips the extra model call entirely.
    PAYMENT_TURN_LLM_VERDICT: bool = True

    # ── Stay dates: who parses them ──────────────────────────────
    #
    # `True` (default): the model is the only date parser on a healthy turn. The
    # regex extractor in `app/services/date_extractor.py` stays behind two
    # narrow doors — it answers a date reply when the model is *unavailable*,
    # and it runs before the model when this flag is `False`.
    #
    # `False` restores the old short-circuit (regex first, model skipped when
    # the regex matched) for a one-push rollback. It is not byte-identical to
    # the old code: the one-date slot-shuffling heuristic is gone, because that
    # is what put a check-out into the check-in slot (prod 2026-10-23 06:30:40).
    #
    # This machine cannot mutate Cloud Run env vars (`iam.serviceaccounts.actAs`
    # fails), so the practical rollback is `git revert`, not this flag.
    DATES_FROM_MODEL_ONLY: bool = True

    # ── Language detection ───────────────────────────────────────
    # Whether the lexical language cues are weighted, or simply counted.
    #
    # Product, 2026-09-30: *"detection of Hindi, Hinglish (Roman Hindi) is not
    # consistent, it still replies in english most of the time."*
    #
    # The old rule was a straight head-count — `h_count >= e_count` for
    # Hinglish, `e_count > h_count` for English — and the two sides are not
    # comparable. Hinglish borrows English *content* words ("room", "book",
    # "price", "family"); English borrows no Hindi. So one borrowed noun could
    # outvote real Hindi:
    #
    #   "2 adults aur 1 child"      -> English   ("aur" was not even a cue)
    #   "room ki photo bhejo"       -> no cues at all -> English default
    #   "thank you" (mid-Hinglish)  -> English, and it stuck for the session
    #
    # Three changes, all in `understand.py`:
    #
    #   1. The cue set gains the function words Roman Hindi is actually made
    #      of — `kya`, `mujhe`, `ka`, `ki`, `ke`, `aur`, `wala`, `bhejo`. Their
    #      absence is most of why messages scored zero Hindi cues.
    #   2. Hindi wins ties and near-ties: English must score more than *twice*
    #      the Hindi count to take the turn. An English word inside a Hindi
    #      sentence is not evidence of English.
    #   3. An acknowledgement — "ok", "thanks", "done", "yes" — changes
    #      nothing, from the cue counts or from the model's read of them. It
    #      answers a question; it does not choose a language.
    #
    # Measured on 34 real strings against a human read of the same 34: 3
    # wrong under the head-count, 0 under the weighted rule. Exactly three
    # strings change: "room ki photo bhejo" and "2 adults aur 1 child" become
    # Hinglish, and "thank you" stops choosing a language.
    #
    # `False` restores the head-count and the legacy cue set exactly, for a
    # one-push rollback.
    LANGUAGE_DETECTION_WEIGHTS_HINDI: bool = True

    # ── Guest name capture ───────────────────────────────────────
    # Whether the name prompt reads a *name* out of the guest's reply, or takes
    # the reply whole.
    #
    # Product, 2026-09-30: *"better name parsing when asking for guests name,
    # right now when name is asked it takes the whole thing no matter what the
    # customer sent"*.
    #
    # The old read took `state["user_message"]` verbatim, stripped one known
    # prefix, and stored whatever survived if it was at most four words, held no
    # digit, and matched a letters-only regex. So `"hi i am rahul"` passed every
    # check and the guest was saved as **"Hi I Am Rahul"**, and `"Rahul English"`
    # swallowed the language answer into the name. The checks looked for reasons
    # to *reject* a whole message; none of them looked for a name.
    #
    # `app/utils/person_name.py` scans instead: greetings and self-introduction
    # filler are dropped, the leading run of name-shaped tokens is kept, and the
    # scan stops at the first token that is not name-shaped. A reply with no
    # name in it now returns `None` — which means the bot asks again, the safe
    # direction, rather than calling the guest "Looking".
    #
    # `False` restores the verbatim read exactly, for a one-push rollback.
    NAME_CAPTURE_READS_ONLY_THE_NAME: bool = True

    # ── Payment Backend ──────────────────────────────────────────
    STAYINN_BACKEND_URL: str = ""
    STAYINN_PAYMENT_ORDER_KEY: str = ""

    # ── Razorpay locked-amount UPI QR ────────────────────────────
    # ON from 2026-11-09 at product instruction: a Razorpay QR replaces the
    # locally-built `upi://` QR image, because the QR is the only payment route
    # that can confirm itself. A hand-made `upi://` QR is a picture of a payment
    # address — nothing watches it, so a guest who scans it still has to send a
    # screenshot and wait for the front desk. A Razorpay QR is an order, so the
    # webhook names it when it is paid.
    #
    # Feature gate: `False` restores the old static UPI QR image and the old
    # verify-payment flow, in one environment change. All new reads are behind
    # this flag; the payment-link path itself is untouched either way.
    RAZORPAY_QR_ENABLED: bool = True

    # QR lifetime. The guest-facing hold is 30 minutes (`hold_expires_at` in
    # payment/link.py), and a QR that outlives its hold would take money for a
    # room already released. Keep the two equal; do not raise this alone.
    RAZORPAY_QR_EXPIRY_MINUTES: int = 30

    # ── Task / escalation completion notices ─────────────────────
    # Staff complete tasks in the CRM panel, which writes `STATUS_CHANGED`
    # directly into `hotel_tasks` — it does not call `/api/tasks/{id}/...`.
    # Measured: one call to that route since 2026-10-01. So a guest whose
    # payment was just verified in the panel is told nothing. This sweep reads
    # the same collection the panel writes and sends that message.
    #
    # Feature gate: `False` stops the loop dead in one environment change.
    TASK_NOTIFICATION_SWEEP_ENABLED: bool = True

    # How often the sweep looks for completed-but-unannounced tasks. Cloud Run
    # scales to zero, so this is best-effort: the loop only ticks while an
    # instance is warm, and a missed tick is picked up on the next request.
    TASK_NOTIFICATION_SWEEP_INTERVAL_SECONDS: int = 120

    # Nothing completed before this instant is ever announced. This is a fixed
    # cut-over date, not "now at boot", so the tasks already completed before
    # the feature existed are never messaged. Set to the day *after* this ships:
    # the clone already holds real completions from 2026-11-09, and replaying
    # them to their guests would be a message nobody asked for. Anything staff
    # finish from 10 November onward is announced.
    TASK_NOTIFICATION_MIN_COMPLETED_AT: str = "2026-11-10T00:00:00+00:00"

    # The same silence exists for escalations, and for the same reason: the API
    # routes notify the guest, but staff resolve tickets in the CRM panel, which
    # writes `support_tickets` directly. This sweep reads that collection and
    # sends the resolution message. `False` stops the loop dead in one
    # environment change.
    ESCALATION_NOTIFICATION_SWEEP_ENABLED: bool = True

    # Nothing resolved before this instant is ever announced. A fixed cut-over
    # date rather than "now at boot", so tickets already resolved before the
    # feature existed are never replayed to their guests. Anything staff resolve
    # from 10 November onward is announced.
    ESCALATION_NOTIFICATION_MIN_RESOLVED_AT: str = "2026-11-10T00:00:00+00:00"

    # Declared so pydantic-settings will actually read them: this model sets
    # extra="ignore", so an env var with no matching field is discarded and
    # getattr(settings, ...) returns None. Without these two lines the Razorpay
    # webhook computed an empty secret and rejected every request with 400.
    #
    # Both default to empty, and payment_webhook fails closed on an empty
    # secret. Do NOT give these a hardcoded fallback: anyone who can read this
    # repository could then forge a "payment captured" webhook and confirm
    # bookings without paying. Set the real value in the deploy environment.
    RAZORPAY_WEBHOOK_SECRET: str = ""
    RAZORPAY_KEY_SECRET: str = ""

    # ── Media Processing ─────────────────────────────────────────
    GOOGLE_APPLICATION_CREDENTIALS: str | None = None
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_IMAGE_TYPES: list[str] = ["jpg", "jpeg", "png"]
    ALLOWED_PDF_TYPES: list[str] = ["pdf"]
    ALLOWED_AUDIO_TYPES: list[str] = ["mp3", "wav", "m4a", "ogg", "webm"]

    # ── CRM & Staff Authentication (GCP/VM Environment Ready) ────
    STAFF_PHONE_NUMBERS: str = ""  # Comma-separated staff phone numbers for VM/GCP env fallback
    CRM_BACKEND_URL: str = ""
    CRM_CHAT_HISTORY_URL: str = ""
    CRM_MONGO_URI: str = "mongodb+srv://USER:PASSWORD@CLUSTER.mongodb.net/"
    BACKEND_API_SECRET_KEY: str = "CHANGE_ME_BACKEND_SECRET"

    # ── CRM confirmation lookup ──────────────────────────────────
    # Whether the bot may ask the CRM whether a guest's stay is already confirmed by
    # staff, and suppress its own payment demand if so.
    #
    # The problem this addresses (2026-10-12): a booking confirmed by staff in the
    # Control Centre is written to the CRM's database and stops there. Nothing carries
    # it into the bot's `bookings`, so the stage resolver still reads `pending_payment`
    # and re-enters the payment flow — a guest who has already paid is asked for the
    # advance again. Measured on hotel 111111: paid 2026-10-11 13:26, staff confirmed
    # 13:34, and at 06:37 the next morning the bot minted a new booking and demanded
    # ₹8,040.
    #
    # This is a READ of another team's cluster, in the guest's turn. Switched ON
    # 2026-10-13 after a live replay on hotel 111111: same real confirmed guest, same
    # dates, flag off → the bot minted a booking and demanded a ₹2,300 advance via UPI;
    # flag on → "our team has already confirmed this booking", no booking, no demand.
    #
    # `False` restores the old behaviour exactly, but NOT from a machine: it is a Cloud
    # Run env var and this account lacks `iam.serviceaccounts.actAs`, so the real
    # rollback is a revert commit. Set it here, not in the deploy config, for that reason.
    CRM_CONFIRMATION_LOOKUP_ENABLED: bool = True

    # The confidence below which a CRM `CONFIRMED_BOOKING` is not believed. Measured
    # 2026-10-12 across the CRM's 167 confirmed contacts: 155 sit at ≥0.95, and the 3
    # below 0.9 all carry no dates and no amount, so this floor drops exactly the ones
    # that could not be matched to a stay anyway.
    CRM_CONFIRMATION_MIN_CONFIDENCE: float = 0.9

    # Which CRM database holds `contact`. There is no env var default to inherit by
    # accident: the field below is the single source of truth.
    #
    # The CRM's real collections live in a database named `test`: its own `MONGO_URL`
    # carries no database in the path, so MongoEngine falls back to its default name.
    # The name is an accident rather than a decision, which is exactly why it is a
    # setting: if the CRM ever names the database properly, that is a config change and
    # not a code change. Open question for the CRM team.
    CRM_CONFIRMATION_DATABASE: str = "test"

    # Whether a CRM-confirmed stay whose check-out date has NOT passed puts the
    # guest into the confirmed stage — the FAQ/concierge path, zero selling — even
    # though the bot owns no booking row for them. product rule, 2026-10-13: "if
    # the end date of the stay has not passed act like an faq bot, and if it has
    # passed act normally."
    #
    # No check-out date on the CRM contact → we cannot say the stay has not ended →
    # the guest is treated normally. Fails closed, like everything else here.
    #
    # Deliberately separate from `CRM_CONFIRMATION_LOOKUP_ENABLED`: that flag also
    # drives the payment-demand suppression, so turning the stage promotion off must
    # not silently switch the payment fix back on.
    CRM_CONFIRMED_STAGE_ENABLED: bool = True

    # How long a CRM confirmation that names NO dates keeps a guest in the confirmed
    # stage. 0 disables the window, restoring the old "no dates ⇒ treat as a lead".
    #
    # Why this exists: measured 2026-11-10 on the CRM's 219 `CONFIRMED_BOOKING`
    # contacts, 105 carry no `checkOutDate` at all — the CRM's classifier could not
    # read one from the conversation even though the guest had paid and staff had
    # confirmed. Under the old rule every one of them fell through to the selling
    # branches, and `handle_stage_confirmed` answered them with "please tell me your
    # check-in dates and number of guests" — asking a paid guest to send in the
    # details of the booking they already held. That is the reported complaint.
    #
    # 30 days is a heuristic and is meant to be: with no dates there is nothing to
    # compare the clock against, so the only question we can answer is "did the CRM
    # judge this recently?". The anchor is `classification.classified_at` (an ISO
    # string on all 219 contacts, measured), falling back to `last_interaction` when
    # that is missing. Both move when the guest writes again, so a guest from a past
    # stay who chats today reads as current — see the risk note in the PR: the window
    # only ever REMOVES selling, and 0 turns it off in one env change.
    CRM_CONFIRMATION_NO_DATE_WINDOW_DAYS: int = 30

    # Whether a guest we ALREADY know is confirmed can be shown a room card or asked
    # for their stay dates, even if this turn's stage resolution did not say so.
    #
    # `CRM_CONFIRMED_STAGE_ENABLED` above promotes the guest and `route_by_intent`
    # sends every confirmed turn to `stage_confirmed` (zero selling). That chain has
    # one weak link: it trusts this turn's resolved stage. When resolution misses —
    # a lookup that timed out, a revision older than the CRM fix, a session stage
    # that was written before the guest was confirmed — the guest falls through to
    # the ordinary selling branches, and the card goes out anyway. Measured on the
    # prod clone, 2026-10-21: 10 guests the CRM called confirmed had a room card
    # emitted AFTER the CRM's own `classified_at`, and 18 more in the window after
    # the CRM fix was deployed.
    #
    # The guest 919433543471 (2026-10-19, 2026-10-20) is the shape of it: they wrote
    # "Kindly confirm our booking from 21-23 Oct 2026", and the reply was the full
    # ROOMS card twice, while the CRM held exactly that stay at confidence 0.95.
    #
    # This flag adds the same predicate the four ghost guards already share —
    # `guest_is_confirmed`, our own booking id or the CRM's staff signal — at the
    # three places a card or a date-ask is produced, so the guarantee no longer
    # depends on stage resolution succeeding first. It only ever REMOVES selling.
    #
    # ON by default, and the rollback is a revert: this account cannot flip a Cloud
    # Run env var (`iam.serviceaccounts.actAs`), the same reason
    # `BOOKING_ID_ON_PAYMENT_PROOF_ENABLED` is written here.
    CONFIRMED_GUEST_NEVER_SOLD_ENABLED: bool = True

    # Translate the buttons and lists the templates author into the `interactive` payload
    # Meta accepts, instead of flattening them into the message body.
    #
    # The templates have authored `button_group` cards since 2026-10-14, but nothing
    # converted them and `format_cards_for_whatsapp` has no branch for the type, so every
    # button was dropped and the guest received the card's body text alone — the main menu
    # arrived as "Here is what I can help you with at Sample Hotel:" with no options under it.
    # The CRM passes an `{"type": "interactive", "data": {...}}` card straight through to
    # Meta (`combot_crm_v2/webhook_receiver/views.py:305`), so only the bot side changes.
    #
    # ON is the default **because production is already broken without it** — the gate
    # cannot be flipped from this account (`iam.serviceaccounts.actAs`), so a default of
    # False would ship the dead menu and need a second deploy to fix. Turning it off in the
    # Cloud Run console is the kill switch: every prompt falls back to the numbered text it
    # carried before, which is the older, known-good behaviour.
    WHATSAPP_BUTTONS_ENABLED: bool = True

    # Say which half of a room request could not be met, and what IS free.
    #
    # AC / Non-AC is a *hard* inventory filter: when no room of the requested
    # class is free, the search returns empty and the guest is told "we don't
    # have rooms available that match your requirements" — even though other
    # rooms are sitting open for the same dates. A live guest (918017904770,
    # 2026-10-17) asked for one non-AC room for 18–21 Oct, when both non-AC
    # room types were closed on the 20th but two AC room types were free on
    # every night; the bot said the same sentence three times and only a staff
    # member quoting a different property moved the booking.
    #
    # ON by default for the same reason as the flags above: the gate cannot be
    # flipped from this account, so a default of False would ship the dead-end
    # message and need a second deploy to fix. Turning it off in the Cloud Run
    # console restores the plain "no rooms available" card exactly as it was.
    ROOM_CLASS_GAP_HINT_ENABLED: bool = True

    # Draw three rooms-card rows — one per AC class, then a step up — when the
    # guest explicitly asked to see both classes.
    #
    # The rollback lever for the 2026-10-28 "show both" fix. Before it, an
    # explicit **Show both** was collapsed to "unstated", the card drew from one
    # class only, and the allocator's ten-cheapest cut meant the AC sets never
    # reached the card on a large party — so the guest got a Non-AC card while
    # the ROOMS list above it advertised the AC rooms. Three changes ride on
    # this flag: the extra rows (`room_packing.both_class_rows`), the
    # class-diverse allocator truncation
    # (`allocation_engine._keep_both_classes`), and the wider ROOMS list that
    # follows the rows.
    #
    # ON by default for the same reason as the flags above — it cannot be
    # flipped from the deploying account, so False would ship the defect. Set it
    # False in the Cloud Run console to restore the one-class card exactly,
    # including the old truncation.
    ROOM_CARD_SHOW_BOTH_ENABLED: bool = True

    # Only the turn whose search actually failed may say "we don't have rooms".
    #
    # `room_type_not_available` is a stored latch: `room_searcher` sets it when a
    # search falls back, and no node clears it afterwards. `generate_response`
    # read it without asking whether *this* turn searched, so every later turn
    # that produced no reply of its own — a thank-you, a greeting, a short
    # acknowledgement — was answered with the availability card. Live transcript,
    # 2026-10-20: the guest asked about hot water and got the answer, then wrote
    # "Ok... thank you sir." and the bot answered "I'm sorry, we don't have rooms
    # available that match your requirements" — twice, contradicting the answer
    # it had just given.
    #
    # ON by default, like the flags above, because the gate cannot be flipped
    # from this account. Turning it off in the Cloud Run console restores the
    # old unconditional branch without a revert.
    NO_ROOMS_REQUIRES_FRESH_SEARCH_ENABLED: bool = True

    # Hand a guest whose preferred room cannot be built to a senior staff member,
    # instead of leaving them at a dead end with a room list.
    #
    # The branch this replaces said "we couldn't build your stay from **X** rooms
    # for these dates" and then listed the other room types. That is a dead end
    # for a guest who asked for one specific room: the list answers a question
    # they did not ask, and no one is told a booking is about to be lost. With
    # this ON the guest is told to hold on, a ticket is raised in the same turn,
    # and the front desk can recover the booking.
    #
    # Narrow on purpose: a plain sold-out search, and a fallback where other
    # rooms *were* allocated, keep their alternative lists — the guest still has
    # a path, and escalating those would drown the front desk in requests the
    # bot already answered.
    #
    # ON by default like the flags above (the gate cannot be flipped from this
    # account); turning it off in the Cloud Run console restores the old text.
    PREFERRED_ROOM_HANDOFF_ENABLED: bool = True

    # Ask for the guest's name after payment is verified, not before it.
    #
    # Until 2026-10-21 the name was a gate in front of the payment link: the
    # guest was asked for it the moment they said "confirm", and could not reach
    # the payment step until they typed it. That is friction at the exact point
    # a guest has already decided to pay, and it asks for a confirmation detail
    # before there is anything to confirm.
    #
    # With this ON the gate is gone: "confirm" goes straight to the payment
    # link, the booking is created with whatever name we hold (usually the
    # WhatsApp profile name, or "Guest"), and the confirmation turn asks for the
    # name to print on the slip. The guest's reply updates the booking row.
    #
    # ON by default; turning it off restores the pre-payment gate without a revert.
    NAME_AFTER_PAYMENT_ENABLED: bool = True

    # No booking ID until the guest has sent proof of payment.
    #
    # Until 2026-10-21 the bot minted a `BK-…` id and wrote a `pending_payment`
    # row the moment the guest typed "confirm", then printed the id in the
    # payment message. The guest therefore held a booking number before paying
    # anything, which reads as a confirmed reservation — and staff in the CRM
    # saw a booking that did not exist yet.
    #
    # With this ON, that turn mints a `PAY-…` reference instead: it keys the
    # gateway order and the 30-minute inventory hold, but is never shown to the
    # guest and is not a booking. The `BK-…` id is minted when the payment
    # screenshot arrives (or, for a gateway-only payment, at confirmation),
    # and the hold row is renamed to it so the same room stays reserved.
    #
    # ON by default, like the flags above: the gate cannot be flipped from this
    # account. Turning it off in the Cloud Run console restores the old
    # "booking id at confirm" behaviour without a revert.
    BOOKING_ID_ON_PAYMENT_PROOF_ENABLED: bool = True

    # Answer what the guest asked while a payment is under verification.
    #
    # `handle_stage_payment_check` returns one fixed paragraph — "we have noted
    # your payment, the front desk is verifying it" — for every turn it serves.
    # That is the right reply to a payment claim and the wrong reply to
    # everything else. Live guest 919833515350 (2026-10-17) was in this stage
    # after saying "I have already paid 1500", then asked "Hi", "Can u send me
    # my booking confirmation" and "But aaj 3 baje mera checkin hai.. jaldi se
    # confirmation bhejiye" — and got the same paragraph verbatim three times,
    # with no acknowledgement that they had a check-in in a few hours. Staff
    # confirmed by hand minutes later.
    #
    # ON by default for the same reason as the flags above: the gate cannot be
    # flipped from this account. Turning it off restores the single paragraph.
    PAYMENT_CHECK_STATUS_REPLIES_ENABLED: bool = True

    # A guest waiting on payment verification can still ask for something else.
    #
    # Prit, 2026-10-27: *"payment_check triggers for unrelated requests — add
    # intent-based routing, reject invalid payment states."*
    #
    # `route_by_intent` holds every turn in `payment_check` unless the intent is
    # one of a short FAQ/social allowlist. That is fine for "what's the wifi
    # password", but `_DIGRESSION_FAQ_AND_SOCIAL_INTENTS` does not contain
    # CANCEL_BOOKING, MODIFY_BOOKING, COMPLAINT, HUMAN_REQUEST or
    # LIFECYCLE_RESET — so a guest asking to cancel, or asking for a person,
    # gets "our front desk is verifying your payment" instead. The escalation
    # branch at `route_by_intent` is never reached, because this check returns
    # first.
    #
    # The same handler also writes to the booking on every turn it serves:
    # `stage: payment_check`, `claimed_paid: True`, `booking_status:
    # pending_payment`. `stage_resolver` reads `claimed_paid` as proof of a
    # payment claim, so a guest who said "hi" is pinned to payment_check from
    # then on. Under this flag the write happens only when the message is
    # actually a claim or a proof was submitted.
    #
    # ON by default (the gate cannot be flipped from this account); turning it
    # off restores both the single allowlist and the unconditional write.
    PAYMENT_CHECK_INTENT_ROUTING_ENABLED: bool = True

    # A "send me the room rent" turn that carries its own dates and party is
    # priced, not catalogued.
    #
    # Measured 2026-10-28, guest 917980021370, hotel Sahu: the guest wrote
    # "Booking Date 25 December 26/27  Prson =9 Adult  One child  Please Hotel
    # Room Rent" — three times — and got the same static room list back each
    # time, headed "2 Adults, 2 Children". The state held check_in
    # 2026-12-25, check_out 2026-12-27, adults 9 and children 1. Nothing was
    # lost in parsing; `SHOW_PRICES` simply routes to `menu`, and the menu
    # prints the catalogue without ever reading the party.
    #
    # ON by default, like the flag above and for the same reason: the gate
    # cannot be flipped from this account, so a default of False would be a
    # change that never reaches production. Turning it off restores the
    # unconditional route to `menu`.
    SHOW_PRICES_PARTY_ROUTING_ENABLED: bool = True

    # A payment waiting for verification gets escalated to staff after 2 hours.
    #
    # Measured 2026-10-27, guest 919999000111 (the booking product flagged): the
    # guest paid Rs.4,400 at 06:32 on 26 Oct and the bot was still saying
    # "PENDING VERIFICATION" 28 hours later. Nothing in the bot ever noticed the
    # wait — each turn re-stated the same status, and the front-desk task sat
    # OPEN and unassigned. The guest cancelled and filed a complaint.
    #
    # Under this flag, once `payment_claimed_at` is older than
    # `PAYMENT_CHECK_ESCALATION_HOURS`, the bot raises the staff task to URGENT
    # once (latched by `payment_escalated_at` on the booking) and tells the guest
    # their case has gone to a manager. It still never says "confirmed".
    #
    # ON by default, like the flags above (the gate cannot be flipped from this
    # account). Turning it off restores the endless "pending" wording.
    PAYMENT_CHECK_ESCALATION_ENABLED: bool = True
    PAYMENT_CHECK_ESCALATION_HOURS: float = 2.0

    # When the guest's own total disagrees with ours, say so instead of
    # repeating ours.
    #
    # Same guest, same thread: staff quoted Rs.17,880 on 25 Oct; the guest
    # priced the stay himself and got Rs.17,501; over two days he asked "please
    # check" five times and every reply re-asserted Rs.17,880 without
    # acknowledging that a second number existed. He cancelled.
    #
    # Under this flag `save_state` drops the sentences that assert our figure,
    # puts "you calculated X, our records show Y" in front of the guest, and
    # raises one front-desk BILLING_INQUIRY task so a human settles it. It never
    # changes a price and never confirms anything.
    #
    # ON by default; turning it off restores the bare repetition.
    PRICE_DISPUTE_GUARD_ENABLED: bool = True

    # A message that is not a receipt may fill gaps in `payment_details`, but
    # must not overwrite what a receipt already told us.
    #
    # Guest 919999000111 typed "Total payable: 17500/-" into a complaint about
    # the price. The extractor tagged it as payment details and `understand`
    # replaced the whole object, so the real receipt's amount (Rs.4,400) became
    # his typed total and its transaction id became None. Every later reply then
    # quoted "your uploaded payment of Rs.17,500" and asked for the Rs.380
    # "balance" — a debt nobody owed.
    #
    # ON means `merge_payment_details` decides: a screenshot/payment-proof turn
    # overwrites, a prose turn only fills fields that are empty. OFF restores
    # last-writer-wins.
    PAYMENT_DETAILS_MERGE_GUARD_ENABLED: bool = True

    # While a payment claim is under verification, no reply may ask for money.
    #
    # Guest 919999000111 paid a Rs.4,400 advance on 26 Oct and was told
    # "please complete the *25% advance payment*" plus a razorpay.me link on
    # 27 Oct — twice — and again after his own arithmetic was read as money
    # received ("total due is Rs.17,880 ... a small balance of Rs.380"). The ask
    # came from a FAQ article reproduced by RAG and from handlers that never
    # consult the payment stage; the "verification in progress" line was glued
    # underneath by `append_booking_resume`, which inspects the stage and never
    # the body it appends to. He filed a complaint and cancelled.
    #
    # ON means `save_state` drops every payment-asking sentence, every payment
    # link / VPA / QR card, and states the pending line if it went missing. A
    # reply that mentions a price without demanding it is left alone.
    PENDING_PAYMENT_GUARD_ENABLED: bool = True

    # A payment claim that matches no booking row is reported, not swallowed.
    #
    # Guest 919999000111: the stay was booked in the hotel's own PMS and the
    # voucher sent as a PDF, so `stayinn_web.bookings` never had a row for him.
    # `stage_payment_check`'s `update_one` matched nothing, its result was never
    # inspected, and eight PAYMENT_CONFIRMATION tasks asked the front desk to
    # verify a payment against a booking that did not exist here.
    #
    # ON means a zero-match claim first checks whether *any* row exists for this
    # phone/reference; if none does, the dropped write is recorded via
    # `record_failure` and one URGENT task says the front desk's actual problem:
    # "this guest says he paid and we have no booking for him".
    PAYMENT_CLAIM_ORPHAN_TASK_ENABLED: bool = True

    # A UPI handle has to be configured, not guessed from the hotel's name.
    #
    # `get_hotel_upi_details` used to end in two fallbacks that *invented* an
    # address: `{property_code or hotel name}@ybl` from the `hotels` document,
    # and `{hotel_name}@ybl` when there was no document at all — both with
    # `connected = True`. So every hotel, configured or not, came back
    # "connected" with a usable-looking VPA and a qrserver.com QR pointing at it.
    # Measured 2026-10-27: hotel 999999 (exists nowhere) → connected=True,
    # sometestresort@ybl, QR yes.
    #
    # That was inert while the QR was only attached on the gateway-failure path.
    # `02aaf0a` (Prit, 2026-10-26) hoists the QR card so it also rides beside a
    # live Razorpay link, which makes an invented handle sit next to a real
    # payment link on *every* payment message. A guest who scans it pays an
    # address that may belong to nobody — and if it does belong to someone, it
    # is not us. Even a correct-looking `sometestresort@ybl` is a guess, and the
    # guess is what a guest's money follows.
    #
    # ON means: only a VPA stored in `payment_gateways` or `hotel_settings` is
    # returned, and an explicit `connected: false` on the gateway config is
    # honoured. Turning it off restores both invented fallbacks.
    #
    # Turning this ON does not configure anything. A hotel whose VPA lives only
    # in the old code special-case (`988273` / `108959` → hotelsahu@ybl) now
    # returns "no UPI" until its handle is stored. That is the safe direction:
    # the guest gets the gateway link, and on a gateway outage they get an
    # honest "couldn't generate a payment link" instead of an unverified address.
    PAYMENT_UPI_REQUIRES_CONFIGURED_VPA: bool = True

    # A settled guest stays settled for the turn.
    #
    # Product, 2026-10-27: *"Locks i mean Paid so paid / Confirm so confirm / Don't
    # confuse the guest."* The stage resolver decides the stage from the booking
    # record at the top of every turn — but nothing stopped a later node from
    # overwriting it. `understand` has three early returns that rebuild the stage
    # from scratch, and its ghost guards write `lead` whenever the guest has no
    # id this repo recognises. A guest who had paid and was confirmed was put
    # back to `lead` mid-conversation and sold a second booking (conversation
    # 919739843511, 2026-10-26).
    #
    # With this ON, `feature_gate` — the one node every turn passes through
    # between `understand` and routing — restores a settled stage that a later
    # node tried to move backwards. It can only hold a stage, never advance one:
    # a demotion the guest asked for (`/start`, "cancel this") is released, and
    # every turnaround the resolver itself makes (a stay that aged out of the
    # checked-out window, a second-booking draft) is computed before this runs,
    # so it is never second-guessed here. See `app/services/stage_lock.py`.
    #
    # ON by default for the same reason as the flags above: the gate cannot be
    # flipped from this account. Turning it off restores today's behaviour.
    STAGE_LOCK_ENFORCED: bool = True

    # Do not raise a payment-verification task that carries nothing to verify.
    #
    # Prit, 2026-10-27: *"Require a valid booking and validated payment evidence
    # before creating payment tasks."* Measured in the mirror: 133 of 157
    # payment-confirmation tasks have no booking reference. Most are legitimate
    # — a guest sends a screenshot before a `BK-…` id exists — so a booking ref
    # is not required. What is refused is the task with *no* booking ref, no
    # amount, no UTR and no receipt behind it: front desk cannot act on it and
    # it clutters the same queue as the real ones.
    #
    # ON by default, like the flags above. Turning it off restores the old
    # unconditional creation.
    PAYMENT_TASK_REQUIRES_EVIDENCE: bool = True

    # ── A confirmed guest asking what the rooms cost ──────────────────────────
    #
    # Product, 2026-10-25: *"bot should be able to answer room prices and normal
    # questions"*. Until this flag the stage refused. `stage_confirmed`'s rate
    # branch answered any price word with "you already have a booking, no new
    # quote needed", and `kb_answer`'s selling guard turned a room-list question
    # into the directions paragraph. Neither is an answer to the question asked.
    #
    # A price list sells nothing and creates no booking. The duplicate-booking
    # risk this stage's ZERO SELLING rule was written for lives in the draft,
    # and that stays gated behind S57's explicit "is this a separate booking?"
    # question — which this flag does not touch.
    #
    # ON by default for the same reason as every flag around it: it cannot be
    # flipped from this account, so a default-OFF would ship the broken
    # behaviour. Turning it off restores the S57 question for a room browse.
    CONFIRMED_STAGE_ANSWERS_ROOM_PRICES: bool = True

    # ── A confirmed guest asking for the room list ────────────────────────────
    #
    # "Room types & rates" and "Book a room" are reply buttons on the menu a
    # confirmed guest is still shown, and `routing.py` sends every confirmed turn
    # to `stage_confirmed` no matter which button was tapped. The tap used to
    # reach that handler's tail and come back with the directions/parking
    # paragraph — an answer to a different question. ZERO SELLING still forbids
    # showing the catalogue, so the reply is S57's: the current stay is safe, is
    # this a separate booking? product ruling, 2026-10-18.
    #
    # Narrowed 2026-10-25 by CONFIRMED_STAGE_ANSWERS_ROOM_PRICES above: while
    # that flag is on, a "Room types & rates" tap is answered with the price
    # list, and only "Book a room" (a booking intent, not a price question)
    # still reaches S57's question through this switch.
    #
    # ON by default for the same reason as the flags above. Turning it off
    # restores the canned paragraph for a rooms tap.
    CONFIRMED_ROOMS_ASK_SEPARATE_BOOKING: bool = True

    # ── A second booking alongside the confirmed one ──────────────────────────
    #
    # S57 asks a confirmed guest "is this a separate booking?", and on "yes" it
    # sets `booking_stage="lead"` — which lasts exactly one turn. `context_loader`
    # calls `resolve_guest_stage` on every load, that function re-reads the
    # guest's booking rows, and a confirmed row outranks a lead draft by
    # construction, so the guest is dragged back to `confirmed` and the new
    # booking cannot start. Measured on prod 2026-10-23 (the bossman's own
    # 90-message run, phone 919616422222): "start fresh new booking" and
    # "Yes, new booking" were sent three times and every following turn was
    # still `STAGE_2_CONFIRMED_DISPATCH`.
    #
    # With this on, `separate_booking_active` makes the resolver hold the
    # standing stay aside: the guest resolves to `lead`, the stay is carried as
    # `standing_booking` instead of `active_booking`, and both survive. The
    # latch is bounded twice, because an unbounded one hides a paid booking: it
    # expires after SECOND_BOOKING_LATCH_HOURS, and it clears itself the moment
    # the session reaches `confirmed` again — the new booking's own completion.
    # `/start fresh` clears it too.
    #
    # TWO gates guard this change, and both must be on:
    #
    #   1. this env flag — the kill switch, ON by default for the same reason as
    #      every flag above: it cannot be flipped from this account, so a
    #      default-OFF would ship the broken behaviour;
    #   2. `second_booking_alongside_enabled` in the hotel's own Hotel-Settings
    #      document — set `behavior.features.second_booking_alongside_enabled` in
    #      Mongo; `business_config.features` is the same flag after parsing. ON by
    #      default, and the OPERABLE off-switch. The DB team can set it false for
    #      one hotel and the latch stops within the 5-minute config cache, with no
    #      deploy and no Cloud Run access. That is the rollback the reviewer of the
    #      first cut asked for; "on by default with a redeploy as the only way
    #      back" was the objection.
    #
    # `second_booking_gate_enabled` (app/services/second_booking.py) is the one
    # reader of both, and both the latch's writer and its reader call it.
    SECOND_BOOKING_ALONGSIDE_ENABLED: bool = True

    # How long S57's "yes" keeps the standing booking aside.
    #
    # Without a bound, a guest who says "yes, new booking" and then walks away
    # resolves as a lead forever: `context_loader` stops passing their booking
    # id, so the in-house, checked-out, cancel and modify paths all become
    # unreachable while their confirmed stay still stands. A day is long enough
    # for a real second booking to be built and short enough that the guest is
    # back to normal on their next visit.
    SECOND_BOOKING_LATCH_HOURS: int = 24

    # ── Booking lookup by `guest_id` ──────────────────────────────────────────
    #
    # `get_recent_bookings` and the stage resolver used to match a guest on
    # `customer_phone` and `customer_identifier` only. 373 of the 413 rows in the
    # bookings collection carry neither, because they predate the writer that
    # populates them — so those guests resolved to `lead` and were quoted rates
    # for a stay that was already confirmed and paid for.
    #
    # The number is not missing, only unread: those rows carry
    # `guest_id = GUEST-<hotel_id>-<phone>`, and the `guests` collection holds the
    # same number under `phone` (they agreed on 56 of 59 sampled rows). This flag
    # adds a third `$or` branch matching that id, which the `guest_id_1` and
    # `hotel_guest_created` indexes already cover.
    #
    # ON by default for the same reason as the flag above: the gate cannot be
    # flipped from this account, so a default of False would ship the broken
    # behaviour and need a second deploy to fix. Turn it off in the Cloud Run
    # console to fall back to the phone-only lookup, which is byte-identical to
    # the query this code ran before.
    BOOKING_GUEST_ID_LOOKUP_ENABLED: bool = True

    # ── Gemini token usage logging ────────────────────────────────────────────
    #
    # Emits one `GEMINI_USAGE` log line per Gemini call, carrying prompt /
    # cached / output / total token counts. Nothing in this codebase read
    # `response.usage_metadata` before, so LLM spend could not be attributed to
    # a function and there was no way to tell whether Google's implicit context
    # caching was hitting the large static prefixes this bot resends each turn.
    #
    # ON by default because the flag cannot be flipped from this account — a
    # default of False would ship a build with no visibility and need a second
    # deploy to fix. Turn it off in the Cloud Run console if the log volume ever
    # becomes a problem; the wrapper then does nothing and behaviour is
    # byte-identical to the call it made before.
    GEMINI_USAGE_LOGGING: bool = True


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — parsed once on first call."""
    return Settings()
