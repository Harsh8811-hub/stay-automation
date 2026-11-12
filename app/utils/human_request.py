"""
Does this message ask for a person, rather than for a different personality?

Why this exists as a separate predicate. `chat_runner` asks the personality
service whether the guest is asking the bot to change how it talks, and that
service asks an LLM. The LLM reads "I need to speak to a human agent" as "talk
like a human" and switches the bot into its Human-Like mode, whose reply is
"Sure! I'll keep it real with you from here. What do you need?" — no hand-off,
no reception number, no ticket. Measured 2026-10-28: the same sentence passed
on the 27th and failed on the 28th, because a temperature-0.1 model is still a
model.

This runs *before* that question is asked, so the precedence is structural
rather than a line in a prompt. It is deliberately narrow: it only fires on an
explicit request for a person, keyed on the preposition ("speak **to** a human")
so that "talk **like** a human" — a genuine persona request — still reaches the
personality path.
"""

import re

#: Verb + "to/with" + a person-noun. The preposition is the whole discriminator:
#: "talk to a human" is a hand-off, "talk like a human" is a style request.
_REQUEST_A_PERSON = re.compile(
    r"\b(?:speak|talk|chat|connect|transfer|put\s+me|get\s+me|forward\s+me)"
    r"\s+(?:to|with|through\s+to)\s+"
    r"(?:a\s+|an\s+|the\s+|your\s+|one\s+of\s+your\s+)?"
    r"(?:human|person|someone|somebody|agent|staff|representative|rep|manager|"
    r"supervisor|reception|front\s+desk|team|executive)\b"
    # "get me a representative" — same ask without the preposition.
    r"|\b(?:get|bring|send)\s+me\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:human|person|agent|staff|representative|rep|manager|supervisor)\b",
    re.IGNORECASE,
)

#: Requests phrased without a verb, or about the bot itself rather than a person.
#: "stop being a bot" is deliberately absent — that one really is a persona ask.
_BARE_REQUESTS = re.compile(
    r"\b(?:real|actual|human|live)\s+(?:person|being|agent|staff)\b"
    r"|\bhuman\s+(?:please|pls|needed|required)\b"
    r"|\bi\s+(?:do\s*n[o']?t|don'?t)\s+want\s+(?:to\s+(?:talk|chat)\s+(?:to|with)\s+)?"
    r"(?:a\s+|an\s+)?(?:ai|bot|robot)\b"
    r"|\b(?:connect|transfer)\s+me\b",
    re.IGNORECASE,
)


def asks_for_a_person(message: str | None) -> bool:
    """
    True when the guest is asking to be handed to a human.

    False negatives are cheaper than false positives: a missed human request
    still reaches the classifier, which reads intent HUMAN_REQUEST and routes to
    escalation. A false positive would swallow a genuine "be more friendly"
    request, so the patterns stay narrow.
    """
    text = (message or "").strip()
    if not text:
        return False
    return bool(_REQUEST_A_PERSON.search(text) or _BARE_REQUESTS.search(text))
