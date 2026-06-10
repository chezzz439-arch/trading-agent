"""IPO watch — alert via Telegram when a watched private company files to go
public or starts trading.

Searches Google News RSS (no API key) for each watch's queries, then looks for
filing/trading trigger phrases in headlines. Rumor-stage phrasing ("eyes IPO",
"considering IPO") deliberately does NOT trigger. Alerted headlines are
remembered in ``logs/ipo_watch_state.json`` so the daily run never re-sends
the same story.

Runs as part of the morning briefing (scripts/market_open.py calls
``run_watches()``), or standalone:

    python scripts/ipo_watch.py            # search + alert
    python scripts/ipo_watch.py --dry-run  # search + print, no Telegram, no state
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings

# Each watch: company name, news queries, and a token that must appear in the
# headline (cuts unrelated stories that merely match the query).
WATCHES = [
    {
        "name": "d-Matrix",
        "queries": ['"d-Matrix" IPO', '"d-Matrix" stock ticker'],
        "must_mention": ["d-matrix", "d matrix"],
    },
]

# Concrete-event phrases only — a story is a trigger when one appears.
FILED_PHRASES = [
    "files for ipo", "filed for ipo", "files for an ipo", "filed for an ipo",
    "files for initial public offering", "filed for initial public offering",
    "files to go public", "filed to go public", "ipo filing",
    "files s-1", "s-1 filing", "registration statement",
    "confidentially filed", "confidential filing",
    "sets ipo terms", "sets terms for ipo", "prices ipo", "prices its ipo",
    "ipo priced",
]
TRADING_PHRASES = [
    "begins trading", "starts trading", "started trading", "trading debut",
    "makes its debut", "shares debut", "debuts on", "stock debut",
    "lists on the", "goes public on", "shares open", "stock opens",
    "in its debut", "first day of trading", "ipo opens",
]

STATE_PATH = os.path.join(settings.LOG_DIR, "ipo_watch_state.json")
_RSS_URL = ("https://news.google.com/rss/search?q={query}"
            "&hl=en-US&gl=US&ceid=US:en")
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def _load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {"alerted": {}}


def _save_state(state: dict) -> None:
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _fetch_headlines(query: str) -> list[dict]:
    """Google News RSS search -> [{title, link, source, date}]. Never raises."""
    import requests
    from urllib.parse import quote

    try:
        r = requests.get(_RSS_URL.format(query=quote(query)), headers=_UA, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for it in root.iter("item"):
            items.append({
                "title": (it.findtext("title") or "").strip(),
                "link": (it.findtext("link") or "").strip(),
                "source": (it.findtext("source") or "").strip(),
                "date": (it.findtext("pubDate") or "").strip(),
            })
        return items
    except Exception:
        return []


def _classify(title: str) -> str | None:
    t = re.sub(r"\s+", " ", title.lower())
    if any(p in t for p in TRADING_PHRASES):
        return "started trading"
    if any(p in t for p in FILED_PHRASES):
        return "filed to go public"
    return None


def check_watch(watch: dict) -> list[dict]:
    """Return trigger stories for one watch: [{title, link, source, date, event}]."""
    seen, hits = set(), []
    for q in watch["queries"]:
        for item in _fetch_headlines(q):
            title = item["title"]
            key = hashlib.sha1(title.lower().encode()).hexdigest()
            if not title or key in seen:
                continue
            seen.add(key)
            low = title.lower()
            if not any(tok in low for tok in watch["must_mention"]):
                continue
            event = _classify(title)
            if event:
                hits.append({**item, "event": event, "key": key})
    return hits


def run_watches(dry_run: bool = False) -> int:
    """Check all watches, Telegram-alert new triggers. Returns alerts sent."""
    from src.monitoring.telegram_bot import TelegramNotifier

    state = _load_state()
    alerted: dict = state.setdefault("alerted", {})
    notifier = TelegramNotifier()
    sent = 0

    for watch in WATCHES:
        hits = check_watch(watch)
        fresh = [h for h in hits if h["key"] not in alerted]
        print(f"[ipo_watch] {watch['name']}: {len(hits)} trigger headline(s), "
              f"{len(fresh)} new")
        if not fresh:
            continue

        event = ("started trading"
                 if any(h["event"] == "started trading" for h in fresh)
                 else "filed to go public")
        lines = [f"*IPO WATCH* 🚨 {watch['name']} looks like it has {event}:", ""]
        for h in fresh[:5]:
            src = f" — {h['source']}" if h["source"] else ""
            lines.append(f"• {h['title']}{src}")
            if h["link"]:
                lines.append(f"  {h['link']}")
        msg = "\n".join(lines)

        if dry_run:
            print("[ipo_watch] DRY RUN — would send:\n" + msg)
            continue
        if notifier.enabled:
            if notifier.send(msg):
                sent += 1
                print(f"[ipo_watch] alert sent for {watch['name']}")
        else:
            print("[ipo_watch] Telegram not configured — alert skipped:\n" + msg)
        for h in fresh:
            alerted[h["key"]] = {"title": h["title"], "event": h["event"],
                                 "date": h["date"],
                                 "alerted_at": datetime.now(timezone.utc).isoformat()}

    if not dry_run:
        _save_state(state)
    return sent


if __name__ == "__main__":
    run_watches(dry_run="--dry-run" in sys.argv)
