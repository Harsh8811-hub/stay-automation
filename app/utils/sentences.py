"""Split reply text into sentences without breaking URLs, decimals or "incl."."""

from __future__ import annotations

# A sentence ends at `.`, `!` or `?` when what follows is a newline, the end of
# the text, an uppercase letter, a digit or a symbol (emoji, ₹, brackets). Two
# things are deliberately *not* boundaries:
#
#   - a period inside a URL or a number — "razorpay.me", "17,880.50". The naive
#     `[^.!?]+` splitter cuts "https://razorpay.me/@hotelsahu" in two, so a
#     guard that removes the first half leaves the guest "me/@hotelsahu".
#   - a period inside a lowercase abbreviation — "17,880 (incl. taxes)".
#
# Newlines always end a chunk: chat replies are line-structured, and a line is
# the smallest unit a guest perceives.
_ABBREVIATION_SAFE_FOLLOWUPS = "\n"


def split_sentences(text: str) -> list[str]:
    if not text:
        return []

    parts: list[str] = []
    buf = ""
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        buf += ch
        if ch == "\n":
            parts.append(buf)
            buf = ""
        elif ch in ".!?":
            end_of_text = i + 1 >= n
            after = text[i + 1] if not end_of_text else ""
            if end_of_text:
                parts.append(buf)
                buf = ""
            elif after.isspace():
                # Consume the run of spaces/tabs; stop at a newline so it can
                # end the chunk itself.
                k = i + 1
                while k < n and text[k].isspace() and text[k] != "\n":
                    k += 1
                nxt = text[k] if k < n else ""
                is_boundary = (
                    nxt == ""
                    or nxt in _ABBREVIATION_SAFE_FOLLOWUPS
                    or nxt.isupper()
                    or nxt.isdigit()
                    or not nxt.isalnum()
                )
                if is_boundary:
                    buf += text[i + 1:k]
                    parts.append(buf)
                    buf = ""
                    i = k - 1
        i += 1

    if buf:
        parts.append(buf)
    return parts
