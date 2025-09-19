# giant_specials_to_sheets.py
# Weekly Ad → Google Sheets (store 2333, Fort Ave Baltimore)
# Steps:
#   1) Load circular page (circular.giantfood.com), capture JSON responses
#   2) Pick the best payload and parse items
#   3) Enrich Category via generic {id,name} lookup; infer Original Price and % off
#   4) Write to Google Sheets
#
# Notes:
# - Robust to schema drift (hunts for item arrays; scans IDs and deals text)
# - Saves weekly_ad.json and optional page artifacts for troubleshooting

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

GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1LbfYLSx2Pj_V5B_1qg4gSdcRB_k1slVrMN2LMdwTpaw/edit"
WORKSHEET_NAME = f"Weekly Ad - Store {STORE_CODE}"
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

# Diagnostics/artifacts
DEBUG = True
INCLUDE_DEAL_TEXT = True  # set False to omit Deal Text column in the sheet
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

    s = str(v).replace(",", "").strip()

    # Multi-buy like "2/$5"
    m = re.search(r"(\d+)\s*/\s*\$?(\d+(?:\.\d+)?)", s)
    if m:
        qty = float(m.group(1)); total = float(m.group(2))
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
        html = (await page.content()).lower()
        if "captcha-delivery.com" in html:
            return True
        if await page.locator("text=Verification Required").count() > 0:
            return True
        if await page.locator("iframe[src*='captcha']").count() > 0:
            return True
    except Exception:
        pass
    return False

# -----------------------
# JSON discovery and ranking
# -----------------------

INTERESTING_URL_HINTS = ("flyer", "item", "items", "product", "offers", "deals", "json", "api", "graphql")

def looks_interesting(url: str, content_type: str) -> bool:
    if "application/json" in (content_type or "").lower():
        return True
    u = url.lower()
    return any(h in u for h in INTERESTING_URL_HINTS)

def tree_find_item_arrays(obj: Any, depth: int = 0) -> List[List[Dict[str, Any]]]:
    """
    Recursively search for arrays of dicts that look like item objects.
    Heuristic: list of dicts with 'name/title/description' and some price-ish key.
    """
    results: List[List[Dict[str, Any]]] = []
    if depth > 7:
        return results

    if isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        price_keys = {"price", "prices", "sale_price", "current_price", "regular_price", "compare_at_price", "pricing"}
        name_keys = {"name", "title", "description", "desc"}
        sample = obj[:10]
        matches = 0
        for it in sample:
            keys = set(k.lower() for k in it.keys())
            if keys & name_keys and (keys & price_keys or "price" in keys or "pricing" in keys):
                matches += 1
        if matches >= max(2, len(sample)//2):
            results.append(obj)
            # Don't recurse deeper from here (already an item list)
            return results

    if isinstance(obj, dict):
        for v in obj.values():
            results.extend(tree_find_item_arrays(v, depth + 1))
    elif isinstance(obj, list):
        for v in obj:
            results.extend(tree_find_item_arrays(v, depth + 1))
    return results

async def discover_weekly_json(page) -> Tuple[Optional[dict], Optional[str]]:
    """
    Listen for JSON responses while the circular page loads.
    Return (parsed_json, url) for the best candidate, saving to weekly_ad.json
    """
    captured: List[Tuple[str, int, str, bytes]] = []  # (url, size, ctype, body)

    def score_candidate(url: str, size: int, ctype: str, obj: Any) -> int:
        base = 0
        if looks_interesting(url, ctype):
            base += 5
        base += min(size // 2048, 20)  # size heuristic (~2KB units capped)
        item_sets = tree_find_item_arrays(obj)
        base += 10 * len(item_sets)
        if item_sets and len(item_sets[0]) >= 10:
            base += 10
        return base

    async def handle_response(resp):
        try:
            url = resp.url
            ctype = resp.headers.get("content-type", "")
            if not looks_interesting(url, ctype):
                return
            body = await resp.body()
            try:
                text = body.decode("utf-8", errors="ignore")
                obj = json.loads(text)
            except Exception:
                return
            captured.append((url, len(text), ctype, body))
        except Exception:
            pass

    page.on("response", lambda r: asyncio.create_task(handle_response(r)))

    async def _passthrough(route):
        await route.continue_()
    await page.route("**/*", _passthrough)

    await page.goto(CIRCULAR_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(8000)

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
        try:
            with open(ARTIFACT_JSON, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
        except Exception:
            pass
    return obj, url

# -----------------------
# Enrichment: category lookup & price inference
# -----------------------

ID_FIELD_RX = re.compile(r"(category|department|section|taxonomy|group|node).*(_ids?|Id|ID|IDs?)$", re.I)

def build_id_name_lookup(payload: Any) -> Dict[str, str]:
    """
    Build a generic {id -> name} mapping by scanning the payload for dicts
    that contain id/name or code/name pairs. Works even if categories live
    under data/meta/taxonomy/sections/etc.
    """
    lookup: Dict[str, str] = {}

    def visit(x: Any, depth: int = 0):
        if depth > 8:
            return
        if isinstance(x, dict):
            keys = {k.lower() for k in x.keys()}
            # common id/name shapes
            if (("id" in x and ("name" in x or "title" in x))
                or ("code" in x and ("name" in x or "title" in x))):
                try:
                    _id = str(x.get("id", x.get("code")))
                    _nm = str(x.get("name", x.get("title"))).strip()
                    if _id and _nm:
                        lookup[_id] = _nm
                except Exception:
                    pass
            for v in x.values():
                visit(v, depth + 1)
        elif isinstance(x, list):
            for v in x:
                visit(v, depth + 1)

    visit(payload)
    return lookup

def extract_related_ids(it: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    From an item dict, extract (fieldName, id) pairs for any key that looks
    like a category/department/section/taxonomy id or ids.
    """
    pairs: List[Tuple[str, str]] = []
    for k, v in it.items():
        if not isinstance(k, str):
            continue
        if ID_FIELD_RX.search(k):
            if isinstance(v, list):
                for _id in v:
                    if _id is not None:
                        pairs.append((k, str(_id)))
            else:
                if v is not None:
                    pairs.append((k, str(v)))
    return pairs

def infer_prices_from_deal_text(orig: Optional[float], sale: Optional[float], deal_text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Try to fill missing original/sale from marketing text such as:
      - "Save $2"          => orig = sale + 2
      - "Save up to $4.00" => same
      - "25% off"          => orig = sale / 0.75
      - "BOGO"             => sale ≈ orig/2
      - "Buy 2 Get 1 Free" => sale ≈ orig*(2/3)
      - "2/$5"             => sale = 2.50
    """
    if not deal_text:
        return orig, sale
    s = deal_text.lower()

    # 2/$5 etc.
    per = to_float(deal_text)
    if sale is None and per is not None:
        sale = per

    # Save $X
    m = re.search(r"save\s*\$?\s*(\d+(?:\.\d+)?)", s, re.I)
    if m and sale is not None and (orig is None or orig < sale):
        try:
            sv = float(m.group(1))
            if sv > 0:
                orig = round(sale + sv, 2)
        except Exception:
            pass

    # X% off
    m = re.search(r"(\d{1,2}|100)\s*%?\s*off", s)
    if m and sale is not None and (orig is None or orig < sale):
        try:
            pct = float(m.group(1))
            if 0 < pct < 100:
                orig = round(sale / (1 - pct/100.0), 2)
        except Exception:
            pass

    # BOGO / buy one get one free
    if orig and sale is None and ("bogo" in s or ("buy 1" in s and "get 1" in s and "free" in s)):
        sale = round(orig / 2.0, 2)
    # Buy 2 get 1 free
    if orig and sale is None and ("buy 2" in s and "get 1" in s and "free" in s):
        sale = round(orig * (2.0/3.0), 2)

    return orig, sale

def parse_item_row(it: Dict[str, Any], cat_lookup: dict) -> Dict[str, Any]:
    # Description
    desc = clean_text(it.get("name") or it.get("title") or it.get("description") or it.get("desc"))

    # Category via direct string or via referenced IDs
    cat = clean_text(it.get("category") or it.get("department") or it.get("dept"))
    if not cat:
        rels = extract_related_ids(it)
        # Prefer fields whose name contains 'category' over others
        preferred = [name for key, _ in rels for name in [cat_lookup.get(_)] if name and "category" in key.lower()]
        fallback  = [name for _, _id in rels if (name := cat_lookup.get(_id))]
        cat = preferred[0] if preferred else (fallback[0] if fallback else None)

    # Prices (direct and nested)
    sale = to_float(it.get("sale_price") or it.get("current_price") or it.get("promo_price"))
    orig = to_float(it.get("regular_price") or it.get("compare_at_price") or it.get("original_price"))

    price = it.get("price") or it.get("prices") or it.get("pricing") or {}
    if isinstance(price, dict):
        sale = sale or to_float(price.get("sale") or price.get("current") or price.get("promo") or price.get("display"))
        orig = orig or to_float(price.get("regular") or price.get("was") or price.get("compare_at") or price.get("strikethrough") or price.get("compare_at_display"))

    # Savings fields sometimes exist
    savings_amt = to_float(it.get("savings") or it.get("savings_amount") or (price.get("savings") if isinstance(price, dict) else None))
    savings_pct = to_float(it.get("savings_percent") or (price.get("savings_percent") if isinstance(price, dict) else None))

    # Deal/marketing text
    deal_text = clean_text(
        it.get("offer_text") or it.get("badge_text") or it.get("tagline") or
        it.get("promotion_text") or it.get("promo_text") or it.get("deal_text") or
        (price.get("label") if isinstance(price, dict) else None)
    )

    # Infer missing prices from savings / deal text
    if orig is None and sale is not None and savings_amt:
        orig = round(sale + savings_amt, 2)
    if orig is None and sale is not None and savings_pct and 0 < savings_pct < 100:
        try:
            orig = round(sale / (1 - (savings_pct/100.0)), 2)
        except Exception:
            pass

    orig, sale = infer_prices_from_deal_text(orig, sale, deal_text)

    row = {
        "Description": desc,
        "Category": cat,
        "Original Price": orig,
        "Sale Price": sale,
        "% Discount": pct_off(orig, sale),
    }
    if INCLUDE_DEAL_TEXT:
        row["Deal Text"] = deal_text
    return row

def parse_rows_from_payload(payload: dict) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    cat_lookup = build_id_name_lookup(payload)

    item_sets = tree_find_item_arrays(payload)
    items: List[Dict[str, Any]] = max(item_sets, key=lambda arr: len(arr)) if item_sets else []

    for it in items:
        if not isinstance(it, dict):
            continue
        row = parse_item_row(it, cat_lookup)
        if row.get("Description") and (row.get("Sale Price") is not None or row.get("Original Price") is not None):
            rows.append(row)
    return rows

# -----------------------
# Google Sheets
# -----------------------

def write_to_google_sheet(df: pd.DataFrame):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=scopes)

    try:
        with open(SERVICE_ACCOUNT_JSON, "r") as f:
            svc = json.load(f)
        print("[INFO] Using service account:", svc.get("client_email"))
    except Exception as e:
        print("[WARN] Could not read service account email:", e)

    gc = gspread.authorize(creds)
    sh = gc.open_by_url(GOOGLE_SHEET_URL)
    print("[INFO] Opened sheet by URL:", GOOGLE_SHEET_URL)

    try:
        ws = sh.worksheet(WORKSHEET_NAME)
        sh.del_worksheet(ws)
        print("[INFO] Deleted old worksheet:", WORKSHEET_NAME)
    except gspread.SpreadsheetNotFound:
        raise
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

    # Sort for readability
    if "% Discount" in df.columns:
        df["% Discount"] = pd.to_numeric(df["% Discount"], errors="coerce")
        df = df.sort_values(by=["% Discount", "Description"], ascending=[False, True])

    # Local artifacts for inspection
    stamp = datetime.now().strftime("%Y%m%d")
    if not df.empty:
        df.to_csv(f"weekly_ad_parsed_{STORE_CODE}_{stamp}.csv", index=False)
        df.to_excel(f"weekly_ad_parsed_{STORE_CODE}_{stamp}.xlsx", index=False)

    print(f"[INFO] Parsed rows: {len(df)}; preview:", df.head(3).to_dict(orient="records"))

    if df.empty:
        print("[ERROR] Zero rows parsed from the weekly ad JSON. See", ARTIFACT_JSON, "and page artifacts.")
        return 4

    write_to_google_sheet(df)
    return 0

def main():
    code = asyncio.run(run())
    if code != 0:
        raise SystemExit(code)

if __name__ == "__main__":
    main()
