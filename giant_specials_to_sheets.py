# giant_specials_to_sheets.py
# Weekly Ad → Google Sheets (store 2333, Fort Ave Baltimore)
# Strategy:
#   1) Open the Giant circular page for the store on circular.giantfood.com
#   2) Passively capture JSON responses the page requests (no DOM scraping)
#   3) Pick the response that actually contains item data and parse it
#   4) Write Description, Category, Original Price, Sale Price, % Discount to Google Sheets
#   5) On failure, save weekly_ad.json + page.png/page.html and exit non-zero (for GitHub Actions)

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from playwright.async_api import async_playwright

# -----------------------
# Config
# -----------------------

STORE_CODE = "2333"  # 857 E Fort Ave (Baltimore)
CIRCULAR_URL = (
    f"https://circular.giantfood.com/flyers/giantfood-weekly"
    f"?locale=en-US&store_code={STORE_CODE}&type=1"
)

# Use the full Google Sheet URL (recommended)
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1LbfYLSx2Pj_V5B_1qg4gSdcRB_k1slVrMN2LMdwTpaw/edit"
WORKSHEET_NAME = f"Weekly Ad - Store {STORE_CODE}"

# Path to service account JSON (provided by env in GitHub Actions)
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

# Diagnostics/artifacts
DEBUG = True
ARTIFACT_JSON = "weekly_ad.json"
ARTIFACT_PAGE_PNG = "page.png"
ARTIFACT_PAGE_HTML = "page.html"


# -----------------------
# Utilities
# -----------------------

def clean_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or None


def to_float(v: Any) -> Optional[float]:
    """
    Convert various numeric/price forms to float.
    Handles:
      - 2/$5  (returns 2.50)
      - "$3.99", "3.99", "$3.99/lb", "3,499.00"
      - numeric types
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)

    s = str(v)
    s = s.replace(",", "").strip()

    # Multi-buy like "2/$5"
    m = re.search(r"(\d+)\s*/\s*\$?(\d+(?:\.\d+)?)", s)
    if m:
        qty = float(m.group(1))
        total = float(m.group(2))
        if qty > 0:
            return round(total / qty, 2)

    # $x.xx or bare number (allow trailing unit text like /lb)
    m = re.search(r"\$?(\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def pct_off(orig: Optional[float], sale: Optional[float]) -> Optional[float]:
    if orig and sale and orig > 0 and sale <= orig:
        return round(100 * (orig - sale) / orig, 1)
    return None


# -----------------------
# Playwright helpers
# -----------------------

async def open_circular_page(context):
    page = await context.new_page()
    await page.goto(CIRCULAR_URL)

    # Cookie banner (OneTrust variants)
    try:
        await page.locator(
            "#onetrust-accept-btn-handler, button:has-text('Accept All'), button:has-text('Accept all')"
        ).first.click(timeout=3000)
    except Exception:
        pass

    await page.wait_for_load_state("domcontentloaded")
    return page


async def is_botwall(page) -> bool:
    """Detect obvious CAPTCHA/bot-wall pages early."""
    try:
        html = await page.content()
        if "captcha-delivery.com" in html.lower():
            return True
        if await page.locator("text=Verification Required").count() > 0:
            return True
        if await page.locator("iframe[src*='captcha']").count() > 0:
            return True
    except Exception:
        pass
    return False


# -----------------------
# JSON discovery and parsing
# -----------------------

INTERESTING_URL_HINTS = (
    "flyer", "item", "items", "product", "offers", "deals", "json", "api", "graphql"
)

def looks_interesting(url: str, content_type: str) -> bool:
    if "application/json" in (content_type or "").lower():
        return True
    u = url.lower()
    return any(h in u for h in INTERESTING_URL_HINTS)


def tree_find_item_arrays(obj: Any, depth: int = 0) -> List[List[Dict[str, Any]]]:
    """
    Recursively search for arrays of dicts that look like item objects.
    Heuristics: list of dicts with at least one of these keys per element:
      name/title/description AND some price-ish key.
    """
    results: List[List[Dict[str, Any]]] = []
    if depth > 6:
        return results

    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        # Check if this list looks like item objects
        price_keys = {"price", "prices", "sale_price", "current_price", "regular_price", "compare_at_price"}
        name_keys = {"name", "title", "description", "desc"}
        matches = 0
        sample = obj[:8]
        for it in sample:
            keys = set(k.lower() for k in it.keys())
            if keys & name_keys:
                # price present either as direct keys or nested 'price' object
                if (keys & price_keys) or ("price" in keys) or ("pricing" in keys):
                    matches += 1
        if matches >= max(2, len(sample) // 2):
            results.append(obj)  # looks like items
            return results

    if isinstance(obj, dict):
        for v in obj.values():
            results.extend(tree_find_item_arrays(v, depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            results.extend(tree_find_item_arrays(v, depth + 1))
    return results


def parse_item_row(it: Dict[str, Any]) -> Dict[str, Any]:
    # description
    desc = clean_text(
        it.get("name") or it.get("title") or it.get("description") or it.get("desc")
    )

    # category/department
    cat = (
        clean_text(it.get("category"))
        or clean_text(it.get("department"))
        or clean_text(it.get("dept"))
        or clean_text(
            (it.get("hierarchy") or {}).get("department")
            if isinstance(it.get("hierarchy"), dict) else None
        )
    )

    # Price resolution (many schemas possible)
    sale = None
    orig = None

    # common top-level fields
    sale = sale or to_float(it.get("sale_price") or it.get("current_price") or it.get("promo_price"))
    orig = orig or to_float(it.get("regular_price") or it.get("compare_at_price") or it.get("original_price"))

    # nested price objects
    price = it.get("price") or it.get("prices") or it.get("pricing")
    if isinstance(price, dict):
        sale = sale or to_float(price.get("sale") or price.get("current") or price.get("promo"))
        orig = orig or to_float(price.get("regular") or price.get("was") or price.get("compare_at"))

        # Sometimes multi-buy shown as strings inside pricing
        if sale is None:
            sale = to_float(price.get("display"))  # e.g. "2/$5"
        if orig is None:
            orig = to_float(price.get("strikethrough") or price.get("compare_at_display"))

    # fallback: parse any obvious strings
    if sale is None:
        sale = to_float(it.get("price_text") or it.get("sale_text") or it.get("offer_text"))
    if orig is None:
        orig = to_float(it.get("regular_text") or it.get("was_text"))

    return {
        "Description": desc,
        "Category": cat,
        "Original Price": orig,
        "Sale Price": sale,
        "% Discount": pct_off(orig, sale),
    }


async def discover_weekly_json(page) -> Tuple[Optional[dict], Optional[str]]:
    """
    Listen for JSON responses while the circular page loads.
    Return (parsed_json, url) for the best candidate, saving to weekly_ad.json
    """
    captured: List[Tuple[str, int, str, bytes]] = []  # (url, size, ctype, body)

    def score_candidate(url: str, size: int, ctype: str, obj: Any) -> int:
        # Higher is better: prefer JSON with item arrays
        base = 0
        if looks_interesting(url, ctype):
            base += 5
        base += min(size // 2048, 20)  # size heuristic (2KB units capped)
        item_sets = tree_find_item_arrays(obj)
        base += 10 * len(item_sets)
        if item_sets and len(item_sets[0]) >= 10:
            base += 10
        return base

    @page.on("response")
    async def handle_response(resp):
        try:
            url = resp.url
            ctype = resp.headers.get("content-type", "")
            if not looks_interesting(url, ctype):
                return
            body = await resp.body()
            # Try decode JSON
            try:
                text = body.decode("utf-8", errors="ignore")
                obj = json.loads(text)
            except Exception:
                return
            captured.append((url, len(text), ctype, body))
        except Exception:
            pass

    # Navigate + allow XHRs to complete
    await page.route("**/*", lambda route: route.continue_())
    await page.goto(CIRCULAR_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(8000)  # give late requests time

    # Rank candidates
    best: Tuple[int, Optional[dict], Optional[str]] = (0, None, None)
    for url, size, ctype, body in captured:
        try:
            text = body.decode("utf-8", errors="ignore")
            obj = json.loads(text)
            sc = score_candidate(url, size, ctype, obj)
            if sc > best[0]:
                best = (sc, obj, url)
        except Exception:
            continue

    obj, url = best[1], best[2]
    if obj:
        # Save artifact
        try:
            with open(ARTIFACT_JSON, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
        except Exception:
            pass
    return obj, url


def parse_rows_from_payload(payload: dict) -> List[Dict[str, Any]]:
    """
    From the discovered JSON, pull out the best 'items' list(s) and convert to rows.
    """
    rows: List[Dict[str, Any]] = []
    item_sets = tree_find_item_arrays(payload)

    # choose the largest item set
    items: List[Dict[str, Any]] = []
    if item_sets:
        items = max(item_sets, key=lambda arr: len(arr))

    for it in items:
        if not isinstance(it, dict):
            continue
        row = parse_item_row(it)
        if row.get("Description") and row.get("Sale Price") is not None:
            rows.append(row)
    return rows


# -----------------------
# Google Sheets
# -----------------------

def write_to_google_sheet(df: pd.DataFrame):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=scopes)

    # Helpful log (which SA is writing)
    try:
        with open(SERVICE_ACCOUNT_JSON, "r") as f:
            svc = json.load(f)
        print("[INFO] Using service account:", svc.get("client_email"))
    except Exception as e:
        print("[WARN] Could not read service account email:", e)

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(GOOGLE_SHEET_URL)
    print("[INFO] Opened sheet by URL:", GOOGLE_SHEET_URL)

    # Replace target worksheet
    try:
        ws = sh.worksheet(WORKSHEET_NAME)
        sh.del_worksheet(ws)
        print("[INFO] Deleted old worksheet:", WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        print("[INFO] Worksheet not found; creating:", WORKSHEET_NAME)

    ws = sh.add_worksheet(title=WORKSHEET_NAME, rows=str(len(df) + 10), cols=str(len(df.columns) + 2))
    ws.update([df.columns.tolist()] + df.astype(object).where(pd.notna(df), "").values.tolist())
    ws.freeze(rows=1)
    print(f"[INFO] Wrote {len(df)} rows to worksheet '{WORKSHEET_NAME}'")


# -----------------------
# Main
# -----------------------

async def run() -> int:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
            timezone_id="America/New_York",
            locale="en-US",
            viewport={"width": 1366, "height": 900},
        )

        page = await open_circular_page(context)

        if await is_botwall(page):
            # Save artifacts for inspection
            try:
                await page.screenshot(path=ARTIFACT_PAGE_PNG, full_page=True)
                with open(ARTIFACT_PAGE_HTML, "w", encoding="utf-8") as f:
                    f.write(await page.content())
            except Exception:
                pass
            print("[ERROR] Blocked by bot protection (CAPTCHA) on circular site.")
            await browser.close()
            return 2

        payload, src_url = await discover_weekly_json(page)
        if payload:
            print("[INFO] Weekly JSON discovered from:", src_url)
        else:
            # Capture page artifacts for debugging
            if DEBUG:
                try:
                    await page.screenshot(path=ARTIFACT_PAGE_PNG, full_page=True)
                    with open(ARTIFACT_PAGE_HTML, "w", encoding="utf-8") as f:
                        f.write(await page.content())
                except Exception:
                    pass
            print("[ERROR] Could not discover weekly ad JSON from the circular page.")
            await browser.close()
            return 3

        await browser.close()

    # Parse → DataFrame
    rows = parse_rows_from_payload(payload)
    df = pd.DataFrame(rows).drop_duplicates().copy()

    if not df.empty and "% Discount" in df.columns:
        df["% Discount"] = pd.to_numeric(df["% Discount"], errors="coerce")
        df = df.sort_values(by=["% Discount", "Description"], ascending=[False, True])

    # Local artifacts (handy to inspect in Actions)
    stamp = datetime.now().strftime("%Y%m%d")
    if not df.empty:
        df.to_csv(f"weekly_ad_parsed_{STORE_CODE}_{stamp}.csv", index=False)
        df.to_excel(f"weekly_ad_parsed_{STORE_CODE}_{stamp}.xlsx", index=False)

    print(f"[INFO] Parsed rows: {len(df)}; preview:", df.head(3).to_dict(orient="records"))

    if df.empty:
        print("[ERROR] Zero rows parsed from the weekly ad JSON. See", ARTIFACT_JSON, "and page artifacts.")
        return 4

    # Push to Google Sheets
    write_to_google_sheet(df)
    return 0


def main():
    code = asyncio.run(run())
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
