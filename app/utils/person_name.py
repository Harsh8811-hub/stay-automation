"""
Read a person's name out of a guest's reply — and nothing else.

The reported defect, in product words:

    better name parsing when asking for guests name, right now when name is
    asked it takes the whole thing no matter what the customer sent

**Why it happened.** The capture at `understand.py` read `state["user_message"]`
verbatim, stripped one known prefix if the message happened to start with it,
and then accepted whatever was left so long as it was at most four words, held
no digit, and matched a letters-only regex. Every one of these passed:

    "hi i am rahul"      -> saved as  "Hi I Am Rahul"
    "Rahul English"      -> saved as  "Rahul English"   (the language swallowed)
    "rahul sharma here"  -> saved as  "Rahul Sharma Here"

The checks were looking for *reasons to reject* a whole message. None of them
was looking for a name. A greeting is short and has no digits, so it qualified.

**What this does instead: it scans.** Leading greetings, self-introduction
filler and honorifics are dropped; then the leading run of name-shaped tokens
is kept, and the scan stops at the first token that is not name-shaped — a
digit, a language name, a punctuation break, or a word no name contains. So
"hi i am rahul english" yields "Rahul" and drops the rest rather than failing.

**Deliberately not a model call.** This runs on the turn that captures the
guest's name, which is already a model turn; a second call to read two words
would cost a round trip and could still be wrong. The rules below are cheap,
deterministic, and testable, and a name they get wrong is one the guest will
see echoed back and can correct.

Pure function. No state, no I/O, no model.
"""

from __future__ import annotations

import re

#: Self-introduction phrases, stripped from the front before scanning. Longest
#: first, so "my name is" is not eaten by "name is" leaving "my".
_INTRO_PREFIXES = (
    "my good name is",
    "my name is",
    "mera naam hai",
    "mera naam",
    "naam hai mera",
    "naam hai",
    "name is",
    "name:",
    "name -",
    "this is",
    "here is",
    "myself",
    "i am",
    "i'm",
    "im",
    "my self",
)

#: Dropped wherever they appear at the *front* of the scan. Greetings carry no
#: name; the self-introduction words are what is left when a prefix strip left
#: a stray ("hi i am rahul" leaves "hi" before "i am" is reached).
_LEADING_FILLER = {
    "hi", "hii", "hiii", "hey", "hello", "helo", "namaste", "namaskar",
    "good", "morning", "afternoon", "evening", "night",
    "sir", "madam", "maam", "ma'am", "ji", "shri", "shree", "mr", "mrs",
    "ms", "dr", "and", "aur", "ok", "okay", "pls", "please", "kindly",
    "request", "requests", "thank", "thanks", "thankyou", "thx", "shukriya",
    "dhanyawad", "dhanyavad", "bhai", "bhaiya",
    "i", "am", "is", "are", "my", "me", "we", "you", "the", "a", "an",
    "here", "this", "that", "it", "name", "naam", "myself", "self",
}

#: Any of these anywhere in the scan ends it. They are not name-shaped: either
#: a language answer riding along with the name, or a function word that only
#: appears in a sentence. Without this "rahul english" keeps both words.
_STOP_WORDS = {
    # language answers — the commonest thing to arrive in the same message
    "english", "hindi", "hinglish", "hindi", "urdu", "kannada", "tamil",
    "telugu", "marathi", "bengali", "gujarati", "malayalam", "punjabi",
    "inglish", "angrezi",
    # Sentence furniture that is not a name. These are the words that separate
    # "Rahul Sharma" from "I am looking for job" — without them the intro strip
    # leaves a bare verb and the scan happily reads it as a name.
    "want", "need", "would", "like", "to", "for", "with", "from", "book",
    "booking", "room", "rooms", "hotel", "stay", "night", "nights", "adult",
    "adults", "child", "children", "people", "person", "guest", "checkin",
    "check", "out", "in", "price", "cost", "payment", "paid", "pay",
    "chahiye", "karna", "karo", "karde", "hai", "hain", "ho", "hoga", "hogi",
    "kitna", "kitne", "kya", "kaun", "mujhe", "mera", "meri", "hum", "aap",
    "batao", "bataye", "batayiye", "dena", "dedo", "lena", "milega",
    "contact", "detail", "details", "number", "phone", "mobile", "email",
    "vacancy", "job", "jobs", "hiring", "resume", "cv",
    # Courtesies and requests
    "request", "requests", "kindly", "thank", "thanks", "thankyou", "thx",
    "shukriya", "dhanyawad", "dhanyavad",
    # Yes/no and confirmations. A bare "yes" at the name prompt is an answer to
    # a different question, and was being saved as the name "Yes".
    "yes", "yeah", "yep", "yup", "no", "nope", "haan", "nahi", "nahin",
    "confirm", "confirmed", "sure", "definitely", "maybe", "correct", "right",
    # Verbs. The guest describing what they are doing is a sentence, not a name.
    "looking", "searching", "enquiring", "inquiring", "writing", "checking",
    "planning", "travelling", "traveling", "coming", "going", "staying",
    "visiting", "interested", "wondering", "hoping", "trying", "asking",
    "calling", "messaging", "contacting", "enquiry", "inquiry", "query",
    # Question and auxiliary words.
    "what", "when", "where", "which", "who", "why", "how", "do", "does",
    "did", "can", "could", "will", "should", "may", "might", "have", "has",
    "had", "was", "were", "be", "been",
}

#: A single name token: letters, and the punctuation real names carry
#: (O'Brien, D'Souza, Kumar-Singh, A. K. Sharma).
_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'’.\-]*$")

#: A bare initial — "A." in "A. K. Sharma". Tested *before* the filler list,
#: because stripping the period turns it into "a", which is a filler word, and
#: the initial was being eaten before the surname was.
_INITIAL_RE = re.compile(r"^[A-Za-z]\.$")

#: Clause breaks end the scan — "Rahul and Priya" is two people and the lead
#: guest is the first of them.
_BREAK_RE = re.compile(r"[,;:!?/\\|()\[\]{}\"“”]|\s+-\s+|\d")

#: The most words a name may be. Four covers "Mohammed Abdul Kalam Sheikh".
_MAX_NAME_WORDS = 4


def _strip_intro(text: str) -> str:
    """Remove one leading self-introduction phrase, if one is there."""
    low = text.lower()
    for prefix in _INTRO_PREFIXES:
        if low.startswith(prefix):
            return text[len(prefix):]
    return text


def extract_person_name(text: str) -> str | None:
    """The guest's name, or `None` if the message does not open with one.

    Returns a title-cased name of at most four words. `None` means "no name
    here", which callers must treat as "ask again", never as "use the message".
    """
    if not text or not text.strip():
        return None

    body = _strip_intro(str(text).strip())

    # Split on clause breaks and scan the FIRST clause only: a name is at the
    # front of the message or it is not a name answer at all. Scanning later
    # clauses is what let "hi i am rahul" keep the greeting — the greeting is
    # clause one and the name is clause two.
    #
    # But an intro strip or a dropped greeting can leave the name a clause
    # later ("hi, i am rahul" -> "hi, rahul" -> clause 2), so the first *two*
    # clauses are tried, in order, and the first that yields a name wins.
    clauses = [c for c in _BREAK_RE.split(body) if c.strip()]

    for clause in clauses[:2]:
        tokens = clause.split()
        # Drop leading filler and greeting words. An initial is never filler,
        # even though the bare letter after stripping its period would be.
        while (
            tokens
            and not _INITIAL_RE.match(tokens[0])
            and tokens[0].lower().strip(".,!") in _LEADING_FILLER
        ):
            tokens.pop(0)

        name: list[str] = []
        words = 0  # initials do not spend the word budget — see below
        for token in tokens:
            if _INITIAL_RE.match(token):
                # An initial is part of a name, not a word of it. Counting
                # "A. P. J." as three words truncated "A. P. J. Abdul Kalam"
                # to "A. P. J. Abdul" and threw the surname away.
                name.append(token)
                continue
            bare = token.strip(".,!-'’")
            if not bare:
                break
            low = bare.lower()
            if low in _STOP_WORDS or low in _LEADING_FILLER:
                break
            if not _TOKEN_RE.match(bare):
                break
            name.append(bare)
            words += 1
            if words == _MAX_NAME_WORDS:
                break

        # A lone initial is not a usable name — "A." answers nothing. Two
        # letters is the floor, which lets "A. K." through and rejects "A.".
        # Counted across every character, not per token: "O'Brien".isalpha()
        # is False, so a per-token test reads a perfectly good surname as
        # having no letters at all.
        joined = " ".join(name)
        if sum(1 for c in joined if c.isalpha()) >= 2:
            return joined.title()

    return None


#: Words that make a whole message something other than a name answer.
#:
#: `extract_person_name` scans for a name at the *front* of a message and is
#: deliberately permissive — "cancel my booking" opens with a word that looks
#: like one, and it returns "Cancel". That is the right answer to the question
#: it asks ("is a name here?"), and the wrong answer to the question the
#: after-payment prompt asks ("did the guest answer with their name?"), where a
#: wrong yes renames a booking and swallows a cancellation.
_NOT_A_NAME_ANSWER = _STOP_WORDS | {
    "cancel", "cancelled", "stop", "change", "modify", "reschedule", "refund",
    "thanks", "thank", "hi", "hello", "hey", "ok", "okay", "call", "id",
    "aadhar", "aadhaar", "help", "what", "when", "where", "how", "why", "who",
    "reply", "done", "time", "sir", "madam", "resend", "receipt", "screenshot",
    "guests", "member", "members",
}


def looks_like_a_name_answer(text: str) -> str | None:
    """The guest's name, but only when the message is an answer to a name prompt.

    Stricter than `extract_person_name` and meant for the one place that reads a
    reply *as* a name: a question mark, any digit, or a single word that is an
    instruction ("cancel", "thanks", "ok") means this is not that answer, and
    `None` says so. Callers must keep whatever they had rather than falling back
    to the message — see `NAME_AFTER_PAYMENT_ENABLED`.
    """
    raw = (text or "").strip()
    if not raw or "?" in raw or any(ch.isdigit() for ch in raw):
        return None

    candidate = extract_person_name(raw)
    if not candidate:
        return None

    if any(
        w.strip(".,!-'’").lower() in _NOT_A_NAME_ANSWER for w in candidate.split()
    ):
        return None
    return candidate
