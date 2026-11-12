"""Deterministic stay-date extraction. No LLM. Python owns this logic."""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

# Known limits / TODOs:
# TODO: Add "this weekend" / "next Friday" relative date handling
# TODO: Add Hindi-Gujarati month names
# TODO: Note that numeric dates are currently DD/MM (India convention)

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_MON = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ORD = r"(?:st|nd|rd|th)?"
# 25 oct | 25th october 2026
_DM = re.compile(rf"\b(\d{{1,2}}){_ORD}\s*(?:of\s+)?{_MON}\b(?:[\s,]*(\d{{4}}))?", re.I)
# oct 25 | october 25th, 2026
_MD = re.compile(rf"\b{_MON}\s*(\d{{1,2}}){_ORD}\b(?:[\s,]*(\d{{4}}))?", re.I)
# 25/10, 25-10-2026, 25.10.26  (India: DD/MM)
_NUM = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?\b")
# Two days sharing one month name: "22-23 Nov", "20 to 22 nov", "22nd - 23rd Nov".
# Must run BEFORE `_DM`, which sees only the second day (the first has no month
# attached) and turns "22-23 Nov" into a check-in on the 23rd. Both are common
# Indian booking phrasings and both were mis-parsed against real transcripts.
_RANGE = re.compile(
    rf"\b(\d{{1,2}}){_ORD}\s*(?:-|–|—|\bto\b|\btill\b|\buntil\b)\s*(\d{{1,2}}){_ORD}\s*(?:of\s+)?{_MON}\b(?:[\s,]*(\d{{4}}))?",
    re.I,
)
_REL = {"today": 0, "tonight": 0, "tomorrow": 1, "tmrw": 1, "day after tomorrow": 2, "day after": 2}
_NIGHTS = re.compile(r"\b(\d{1,2})\s*(?:nights?|n)\b", re.I)


@dataclass
class StayDates:
    check_in: Optional[date] = None
    check_out: Optional[date] = None
    nights: Optional[int] = None

    @property
    def found(self) -> bool:
        return self.check_in is not None


def _mk(d: int, m: int, y: Optional[int], today: date) -> Optional[date]:
    try:
        if y is None:
            cand = date(today.year, m, d)
            if cand < today - timedelta(days=1):  # roll to next year only if clearly past
                cand = date(today.year + 1, m, d)
            return cand
        y = y + 2000 if y < 100 else y
        return date(y, m, d)
    except ValueError:
        return None


def extract_stay_dates(text: str, today: date) -> StayDates:
    t = text.lower().strip()
    hits: list[tuple[int, date]] = []  # (position, date)

    for rel, off in sorted(_REL.items(), key=lambda kv: -len(kv[0])):
        for m in re.finditer(rf"\b{re.escape(rel)}\b", t):
            hits.append((m.start(), today + timedelta(days=off)))
    spans = []
    for m in _RANGE.finditer(t):
        d1, d2 = int(m.group(1)), int(m.group(2))
        mon = MONTHS[m.group(3)[:3].lower()]
        y = int(m.group(4)) if m.group(4) else None
        a, b = _mk(d1, mon, y, today), _mk(d2, mon, y, today)
        if a and b:
            # Position-ordered, so `hits.sort` keeps the first day first.
            hits.append((m.start(), a))
            hits.append((m.start() + 1, b))
            spans.append(m.span())
    for rx, order in ((_DM, "dm"), (_MD, "md")):
        for m in rx.finditer(t):
            if order == "dm":
                d, mon, y = int(m.group(1)), MONTHS[m.group(2)[:3].lower()], m.group(3)
            else:
                mon, d, y = MONTHS[m.group(1)[:3].lower()], int(m.group(2)), m.group(3)
            dt = _mk(d, mon, int(y) if y else None, today)
            if dt and not any(s <= m.start() < e for s, e in spans):
                hits.append((m.start(), dt)); spans.append(m.span())
    for m in _NUM.finditer(t):
        if any(s <= m.start() < e for s, e in spans):
            continue
        d, mon, y = int(m.group(1)), int(m.group(2)), m.group(3)
        if 1 <= mon <= 12:
            dt = _mk(d, mon, int(y) if y else None, today)
            if dt:
                hits.append((m.start(), dt))

    hits.sort(key=lambda h: h[0])
    uniq = []
    for _, d in hits:
        if d not in uniq:
            uniq.append(d)
    out = StayDates()
    nights = _NIGHTS.search(t)
    if nights:
        out.nights = int(nights.group(1))
    if uniq:
        out.check_in = uniq[0]
        if len(uniq) > 1 and uniq[1] > uniq[0]:
            out.check_out = uniq[1]
            out.nights = (uniq[1] - uniq[0]).days
        elif out.nights:
            out.check_out = uniq[0] + timedelta(days=out.nights)
    return out
