"""SEC EDGAR Form 4 insider-activity analysis (standalone).

Fetches recent Form 4 filings for a ticker from SEC EDGAR, parses open-market
insider buys/sells, and produces a compact :class:`InsiderResult` scoring the
recent insider sentiment (last 30 days).

Fully standalone: depends only on the stdlib + ``requests``. It imports no other
project modules so it can be dropped into any aggregator. All network/parse work
is wrapped in try/except and the analyzer NEVER raises — on failure it returns an
``InsiderResult`` with ``status="error"``.

Usage::

    from src.signals.sec_filings import SECInsiderActivity
    res = SECInsiderActivity().analyze("AAPL")
    print(res.points, res.summary)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger(__name__)

# SEC requires a descriptive User-Agent on every request or it returns 403.
_UA = {"User-Agent": "trading-agent research contact@example.com"}

# Endpoints.
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"

# Tuning knobs.
_WINDOW_DAYS = 30            # only consider Form 4s filed in the last 30 days
_MAX_FORM4 = 12             # parse at most the 12 most-recent Form 4s in window
_LARGE_BUY_USD = 500_000.0  # single-purchase value that flags a "large buy"
_HTTP_TIMEOUT = 20

# C-suite title fragments (case-insensitive substring match on officerTitle).
_CSUITE_TOKENS = ("ceo", "cfo", "coo", "chief", "president")

# transactionCode values we count as open-market activity.
_BUY_CODE = "P"   # open-market / private purchase
_SELL_CODE = "S"  # open-market / private sale


@dataclass
class InsiderResult:
    """Result of an insider-activity analysis for one symbol.

    The public field set is a stable contract consumed by the master-score
    aggregator; do not rename fields.
    """

    symbol: str
    points: int = 0
    status: str = "ok"  # "ok" | "error" | "insufficient"
    buys: int = 0
    sells: int = 0
    csuite_buys: int = 0
    multiple_buyers: bool = False
    large_buy: bool = False
    summary: str = ""
    emoji: str = "⚪"
    detail: str = ""


@dataclass
class _Txn:
    """A single parsed open-market transaction."""

    insider: str
    title: str
    is_csuite: bool
    side: str          # "buy" | "sell"
    shares: float
    price: float
    value: float       # shares * price


class SECInsiderActivity:
    """Analyze recent SEC Form 4 insider transactions for a ticker.

    Parameters
    ----------
    cache_ttl:
        Seconds to cache a per-symbol result. Defaults to 6 hours.
    """

    def __init__(self, cache_ttl: int = 21600) -> None:
        self.cache_ttl = cache_ttl
        # symbol -> (timestamp, InsiderResult)
        self._cache: dict[str, tuple[float, InsiderResult]] = {}
        # ticker (upper) -> CIK int; fetched lazily, once.
        self._cik_map: dict[str, int] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def analyze(self, symbol: str) -> InsiderResult:
        """Return an :class:`InsiderResult` for ``symbol`` (never raises)."""
        sym = (symbol or "").strip().upper()

        # Crypto pairs have no insiders / no Form 4 filings.
        if "/" in sym:
            return InsiderResult(
                symbol=sym,
                status="insufficient",
                points=0,
                emoji="⚪",
                summary="n/a",
                detail="Crypto symbol — no SEC insider filings.",
            )

        # Serve from cache when fresh.
        cached = self._cache.get(sym)
        if cached is not None and (time.time() - cached[0]) < self.cache_ttl:
            return cached[1]

        try:
            result = self._analyze_uncached(sym)
        except Exception as exc:  # noqa: BLE001 — must never propagate
            logger.warning("SEC insider analyze failed for %s: %s", sym, exc)
            result = InsiderResult(
                symbol=sym,
                status="error",
                points=0,
                buys=0,
                sells=0,
                csuite_buys=0,
                multiple_buyers=False,
                large_buy=False,
                emoji="⚪",
                summary="SEC fetch failed",
                detail=f"SEC fetch/parse error: {exc}",
            )

        self._cache[sym] = (time.time(), result)
        return result

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _analyze_uncached(self, sym: str) -> InsiderResult:
        cik = self._resolve_cik(sym)
        if cik is None:
            return InsiderResult(
                symbol=sym,
                status="insufficient",
                points=0,
                emoji="⚪",
                summary="No SEC filer",
                detail=f"No CIK found for {sym} in SEC ticker map.",
            )

        accessions = self._recent_form4_accessions(cik)
        if not accessions:
            return InsiderResult(
                symbol=sym,
                status="ok",
                points=0,
                emoji="⚪",
                summary="No insider activity (30d)",
                detail="No Form 4 filings in the last 30 days.",
            )

        txns: list[_Txn] = []
        for accn in accessions:
            txns.extend(self._parse_form4(cik, accn))

        return self._score(sym, txns)

    def _resolve_cik(self, sym: str) -> int | None:
        """Map a ticker to its zero-padded CIK, fetching the map once."""
        if self._cik_map is None:
            resp = requests.get(_TICKERS_URL, headers=_UA, timeout=_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            self._cik_map = {
                str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()
            }
        return self._cik_map.get(sym)

    def _recent_form4_accessions(self, cik: int) -> list[str]:
        """Return accession numbers of Form 4s filed in the last 30 days.

        Limited to the ``_MAX_FORM4`` most recent within the window.
        """
        cik10 = str(cik).zfill(10)
        resp = requests.get(
            _SUBMISSIONS_URL.format(cik10=cik10), headers=_UA, timeout=_HTTP_TIMEOUT
        )
        resp.raise_for_status()
        rec = resp.json().get("filings", {}).get("recent", {})

        forms = rec.get("form", [])
        dates = rec.get("filingDate", [])
        accns = rec.get("accessionNumber", [])

        cutoff = time.time() - _WINDOW_DAYS * 86400
        out: list[str] = []
        for form, date_str, accn in zip(forms, dates, accns):
            if form != "4":
                continue
            try:
                filed_ts = time.mktime(time.strptime(date_str, "%Y-%m-%d"))
            except (ValueError, TypeError):
                continue
            if filed_ts < cutoff:
                continue
            out.append(accn)
            if len(out) >= _MAX_FORM4:
                break
        return out

    def _parse_form4(self, cik: int, accession: str) -> list[_Txn]:
        """Fetch + parse one Form 4 XML, returning open-market transactions."""
        try:
            root = self._fetch_form4_xml(cik, accession)
        except Exception as exc:  # noqa: BLE001 — skip unreadable filings
            logger.warning("Skipping Form 4 %s (%s): %s", accession, cik, exc)
            return []
        if root is None:
            logger.warning("No XML found for Form 4 %s (%s)", accession, cik)
            return []

        insider, title, is_csuite = self._parse_owner(root)

        out: list[_Txn] = []
        for tx in root.findall(".//nonDerivativeTransaction"):
            code = (tx.findtext(".//transactionCoding/transactionCode") or "").strip().upper()
            ad = (
                tx.findtext(
                    ".//transactionAmounts/transactionAcquiredDisposedCode/value"
                )
                or ""
            ).strip().upper()

            shares = _to_float(
                tx.findtext(".//transactionAmounts/transactionShares/value")
            )
            price = _to_float(
                tx.findtext(
                    ".//transactionAmounts/transactionPricePerShare/value"
                )
            )

            # Determine side: only count genuine open-market buys/sells.
            if code == _BUY_CODE or (ad == "A" and price > 0):
                side = "buy"
            elif code == _SELL_CODE or (ad == "D" and price > 0):
                side = "sell"
            else:
                # Option exercises/grants/gifts/tax (G, M, A, F, ...) — ignore.
                continue

            out.append(
                _Txn(
                    insider=insider,
                    title=title,
                    is_csuite=is_csuite,
                    side=side,
                    shares=shares,
                    price=price,
                    value=shares * price,
                )
            )
        return out

    def _fetch_form4_xml(self, cik: int, accession: str) -> ET.Element | None:
        """Locate and parse the Form 4 XML for one filing.

        The primary XML document is usually ``form4.xml`` but the filename
        varies. We try the common name first, then fall back to the filing's
        JSON index to discover the real ``.xml`` document.
        """
        accn = accession.replace("-", "")
        base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn}"

        # Fast path: the conventional filename.
        resp = requests.get(f"{base}/form4.xml", headers=_UA, timeout=_HTTP_TIMEOUT)
        if resp.status_code == 200:
            return ET.fromstring(resp.content)

        # Fallback: read the filing index and find the XML primary document.
        idx = requests.get(f"{base}/index.json", headers=_UA, timeout=_HTTP_TIMEOUT)
        idx.raise_for_status()
        items = idx.json().get("directory", {}).get("item", [])
        xml_names = [
            it["name"]
            for it in items
            if str(it.get("name", "")).lower().endswith(".xml")
        ]
        # Prefer files that look like ownership docs over the index/header XML.
        xml_names.sort(key=lambda n: ("form" not in n.lower(), n))
        for name in xml_names:
            doc = requests.get(f"{base}/{name}", headers=_UA, timeout=_HTTP_TIMEOUT)
            if doc.status_code != 200:
                continue
            try:
                root = ET.fromstring(doc.content)
            except ET.ParseError:
                continue
            if root.tag.endswith("ownershipDocument"):
                return root
        return None

    @staticmethod
    def _parse_owner(root: ET.Element) -> tuple[str, str, bool]:
        """Extract (insider name, officer title, is_csuite) from a Form 4."""
        name = (
            root.findtext(
                ".//reportingOwner/reportingOwnerId/rptOwnerName"
            )
            or "Insider"
        ).strip()
        rel = ".//reportingOwner/reportingOwnerRelationship/"
        title = (root.findtext(rel + "officerTitle") or "").strip()
        is_officer = _is_true(root.findtext(rel + "isOfficer"))
        is_director = _is_true(root.findtext(rel + "isDirector"))

        title_l = title.lower()
        is_csuite = is_officer and any(tok in title_l for tok in _CSUITE_TOKENS)

        if not title:
            if is_officer:
                title = "Officer"
            elif is_director:
                title = "Director"
            else:
                title = "Insider"
        return name, title, is_csuite

    @staticmethod
    def _score(sym: str, txns: list[_Txn]) -> InsiderResult:
        """Aggregate parsed transactions into an InsiderResult."""
        buys = [t for t in txns if t.side == "buy"]
        sells = [t for t in txns if t.side == "sell"]
        csuite_buys = [t for t in buys if t.is_csuite]

        distinct_buyers = {t.insider for t in buys}
        multiple_buyers = len(distinct_buyers) >= 2
        large_buy = any(t.value > _LARGE_BUY_USD for t in buys)

        # Points.
        points = 0
        for t in buys:
            points += 20 if t.is_csuite else 10
        points -= 8 * len(sells)
        if multiple_buyers:
            points += 15
        if large_buy:
            points += 5

        # Emoji from net direction.
        if len(buys) > len(sells):
            emoji = "🟢"
        elif len(sells) > len(buys):
            emoji = "🔴"
        else:
            emoji = "⚪"

        total_buy_val = sum(t.value for t in buys)
        total_sell_val = sum(t.value for t in sells)

        # Human summary, most-informative case first.
        if multiple_buyers:
            summary = f"{len(distinct_buyers)} insiders buying"
        elif csuite_buys:
            top = max(csuite_buys, key=lambda t: t.value)
            summary = f"{top.title} bought {_money(top.value)}"
        elif buys:
            summary = f"Insider bought {_money(total_buy_val)}"
        elif sells:
            summary = f"Insider sold {_money(total_sell_val)}"
        else:
            summary = "No insider activity (30d)"

        detail = (
            f"{sym}: {len(buys)} buy / {len(sells)} sell over 30d "
            f"(buy {_money(total_buy_val)}, sell {_money(total_sell_val)}; "
            f"C-suite buys {len(csuite_buys)}, distinct buyers {len(distinct_buyers)})"
        )

        return InsiderResult(
            symbol=sym,
            status="ok",
            points=points,
            buys=len(buys),
            sells=len(sells),
            csuite_buys=len(csuite_buys),
            multiple_buyers=multiple_buyers,
            large_buy=large_buy,
            summary=summary,
            emoji=emoji,
            detail=detail,
        )


# ---------------------------------------------------------------------- #
# Small helpers
# ---------------------------------------------------------------------- #
def _to_float(text: str | None) -> float:
    """Parse a numeric XML field to float; 0.0 on failure."""
    if not text:
        return 0.0
    try:
        return float(str(text).strip())
    except (ValueError, TypeError):
        return 0.0


def _is_true(text: str | None) -> bool:
    """SEC booleans appear as '1'/'true'/'0'/'false'."""
    return str(text).strip().lower() in {"1", "true"}


def _money(value: float) -> str:
    """Format a dollar amount as $X.XM or $XXXk or $X."""
    if value >= 1e6:
        return f"${value / 1e6:.1f}M"
    if value >= 1e3:
        return f"${value / 1e3:.0f}k"
    return f"${value:.0f}"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = SECInsiderActivity()

    print("AAPL ->", analyzer.analyze("AAPL"))

    # A frequently sell-heavy large-cap insider profile (RSU sales etc.).
    print("NVDA ->", analyzer.analyze("NVDA"))
