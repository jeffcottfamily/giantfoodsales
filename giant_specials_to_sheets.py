# giant_specials_to_sheets.py
# Scrape Giant Food "All Specials" for a specific store and write to Google Sheets.
# Includes robust diagnostics for GitHub Actions: selector counts, service account info,
# and screenshot/HTML artifacts if zero rows are scraped.

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from playwright.async_api import async_playwright

# -----------------------
# Config
# -----------------------

ALL_SPECIALS_URL = "https://giantfood.com/savings/all-specials"
STORE_PAGE_URL = "https://stores.giantfood.com/md/baltimore-city/857-east-fort-avenue"
TARGET_STORE_NUM = "2333"  # 857 E Fort Ave (Baltimore) store number

# Use the full Google Sheet URL (recommended to avoid ghost sheets owned by the service account)
GOOGLE_SHEET_URL_OR_NAME = "https://docs.google.com/spreadsheets/d/1LbfYLSx2Pj_V5B_1qg4gSdcRB_k1slVrMN2LMdwTpaw/edit"
WORKSHEET_NAME = f"All Specials - Store {TARGET_STORE_NUM}"

# Path to service account JSON is provided by env var in GitHub Actions workflow
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "service_account.json")

# Enable extra logging + capture page.png/page.html if no rows are scraped
DEBUG = True


# -----------------------
# Helpers
# -----------------------

def clean_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def money_to_float(s: Optional[str]) -> Optional[float]:
    """
    Converts price-like strings to a per-unit float.
    Handles: "$2.99", "2/$5", "$3.99/lb", "$1.50 ea", etc.
    For "2/$5" returns 2.50; for "$3.99/lb" returns 3.99.
    """
    if not s:
        return None
    s = s.strip()

    # "2/$5" style
    m = re.search(r"(\d+)\s*/\s*\$?(\d+(?:\.\d+)?)", s)
    if m:
        qty = float(m.group(1))
        total = float(m.group(2))
        if qty:
            return round(total / qty, 2)

    # "$2.99", "2.99", "$3.99/lb"
    m = re.search(r"\$?(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def pct_off(orig: Optional[float], sale: Optional[float]) -> Optional[float]:
    if orig and sale and orig > 0 and sale <= orig:
        return round(100 * (orig - sale) / orig, 1)
    return None


async def text_or_none(locator) -> Optional[str]:
    """Safely get text_content() from a Locator (first match); return None if not found."""
    try:
        return clean_text(await locator.first.text_content(timeout=1000))
    except Exception:
        return None


async def all_texts(locator) -> List[str]:
    """Safely get all_text_contents() from a Locator; returns [] if not found."""
    try:
        contents = await locator.all_text_contents()
        return [t for t in (clean_text(x) for x in contents) if t]
    except Exception:
        return []


# -----------------------
# Store / Cookie Context
# -----------------------

async def set_store_cookie(context):
    """
    Establish store context by visiting the official store page and clicking through,
    then add commonly used store cookies as a fallback.
    """
    page = await context.new_page()
    try:
        await page.goto(STORE_PAGE_URL, wait_until="domcontentloaded")
        # Try "Order Groceries Online" / "Shop this store" style links (sets store context)
        links = page.locator(
            "a:has-text('Order Groceries Online'), a:has-text('Shop'), a[href*='giantfood.com']"
        )
        if await links.count() > 0:
            try:
                await links.nth(0).click()
                await page.wait_for_load_state("domcontentloaded")
            except Exception:
                pass
    except Exception:
        pass

    # Fallback cookies (harmless if ignored by site)
    try:
        await context.add_cookies([
            {"name": "preferred_store", "value": TARGET_STORE_NUM, "domain": ".giantfood.com", "path": "/"},
            {"name": "storeId", "value": TARGET_STORE_NUM, "domain": ".giantfood.com", "path": "/"},
        ])
    except Exception:
        pass
    await page.close()


async def open_all_specials_page(context):
    """
    Open the All Specials page and accept common cookie banners if present.
    """
    page = await context.new_page()
    await page.goto(ALL_SPECIALS_URL)

    # Accept cookies (OneTrust + fallback labels)
    try:
        await page.locator(
            "#onetrust-accept-btn-handler, button:has-text('Accept All'), button:has-text('Accept all')"
        ).first.click(timeout=3000)
    except Exception:
        pass

    await page.wait_for_load_state("networkidle")
    return page


# -----------------------
# Scraper
# -----------------------

async def scrape_all_specials() -> List[Dict]:
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

        # Set store context and open the specials page
        await set_store_cookie(context)
        page = await open_all_specials_page(context)

        # Infinite scroll / lazy load: scroll until page height stabilizes twice
        last_h, stable = 0, 0
        while True:
            try:
                h = await page.evaluate("document.body.scrollHeight")
            except Exception:
                h = last_h
            stable = stable + 1 if h == last_h else 0
            last_h = h
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(900)
            if stable >= 2:
                break

        # Probe likely product card selectors and log counts
        sel_candidates = [
            "[data-testid*='product']",
            "article:has([class*='price'])",
            "div[class*='card']:has([class*='price'])",
            "li:has([class*='price'])",
        ]
        count_map = {}
        for sel in sel_candidates:
            try:
                count_map[sel] = await page.locator(sel).count()
            except Exception:
                count_map[sel] = -1
        print("[INFO] Candidate card counts:", count_map)

        # Choose the first selector that yields a sensible number of cards
        cards = None
        for sel in sel_candidates:
            loc = page.locator(sel)
            try:
                cnt = await loc.count()
                if cnt >= 10:  # expect many specials; adjust if needed
                    cards = loc
                    break
            except Exception:
                continue
        if cards is None:
            # Fall back to a broad net (may include non-product nodes)
            cards = page.locator("section, article, li, div")

        results: List[Dict] = []
        try:
            n = await cards.count()
        except Exception:
            n = 0

        for i in range(n):
            el = cards.nth(i)

            # Title/Description
            title = await text_or_none(
                el.locator("h3, h2, [class*='title'], [data-testid*='title'], [class*='name']")
            )
            # Price blobs
            sale_text = await text_or_none(
                el.locator("[class*='sale'], [data-testid*='sale'], [data-test*='sale'], [class*='price']")
            )
            orig_text = await text_or_none(
                el.locator("[class*='was'], [class*='strikethrough'], [data-testid*='was']")
            )

            # If we only captured one price blob, attempt to parse "Was $X ... $Y"
            if sale_text and not orig_text and "was" in sale_text.lower():
                m1 = re.search(r"was[^$\d]*\$?(\d+(?:\.\d+)?)", sale_text, re.I)
                if m1:
                    # include the matching token as orig_text
                    orig_text = m1.group(0)

            sale_val = money_to_float(sale_text or "")
            orig_val = money_to_float(orig_text or "")

            # Sometimes the "price" class is the regular price. If so, swap.
            if orig_val and sale_val and sale_val > orig_val:
                sale_val, orig_val = orig_val, sale_val

            # Filter out non-cards / incomplete data
            if not title or sale_val is None:
                continue

            # Category heuristics: badges, chips, category links, aria labels
            cat_candidates: List[str] = []
            for sel in [
                "[data-testid*='tag']",
                "[class*='badge']",
                "[class*='chip']",
                "[class*='category']",
                "a[href*='/categories/']",
                "a[aria-label*='category']",
            ]:
                cat_candidates.extend(await all_texts(el.locator(sel)))

            # remove price-like tokens; prefer shortest remaining label
            category = None
            if cat_candidates:
                non_price = [
                    c for c in cat_candidates
                    if not re.search(r"\$|\d/\$|\d+\s*for\s*\d", c, re.I)
                ]
                chosen = non_price or cat_candidates
                # avoid very long marketing blurbs; choose the shortest token
                category = sorted(chosen, key=len)[0] if chosen else None

            results.append({
                "Description": title,
                "Category": category,
                "Original Price": orig_val,
                "Sale Price": sale_val,
                "% Discount": pct_off(orig_val, sale_val),
            })

        # If nothing scraped, capture artifacts to help debug
        if DEBUG and len(results) == 0:
            try:
                await page.screenshot(path="page.png", full_page=True)
                html = await page.content()
                with open("page.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print("[ERROR] Zero rows scraped; saved page.png/page.html for inspection.")
            except Exception as e:
                print("[WARN] Could not save debug artifacts:", e)

        await browser.close()
        return results


# -----------------------
# Google Sheets
# -----------------------

def write_to_google_sheet(df: pd.DataFrame):
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_JSON, scopes=scopes)

    # Log which service account we're using (helps catch wrong project/secret)
    try:
        with open(SERVICE_ACCOUNT_JSON, "r") as f:
            svc = json.load(f)
        print("[INFO] Using service account:", svc.get("client_email"))
    except Exception as e:
        print("[WARN] Could not read service account email:", e)

    gc = gspread.authorize(creds)
    if GOOGLE_SHEET_URL_OR_NAME.startswith("http"):
        sh = gc.open_by_url(GOOGLE_SHEET_URL_OR_NAME)
        print("[INFO] Opened sheet by URL:", GOOGLE_SHEET_URL_OR_NAME)
    else:
        sh = gc.open(GOOGLE_SHEET_URL_OR_NAME)
        print("[INFO] Opened sheet by NAME:", GOOGLE_SHEET_URL_OR_NAME)

    # Delete/recreate target worksheet
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

def main():
    rows = asyncio.run(scrape_all_specials())

    # Build DataFrame
    df = pd.DataFrame(rows).drop_duplicates().copy()

    # Sort for readability
    if "% Discount" in df.columns:
        df["% Discount"] = pd.to_numeric(df["% Discount"], errors="coerce")
        df = df.sort_values(by=["% Discount", "Description"], ascending=[False, True])

    # Log a preview
    try:
        preview = df.head(3).to_dict(orient="records")
    except Exception:
        preview = []
    print(f"[INFO] Scraped {len(df)} rows; first few:", preview)

    # Save local backups (handy as GitHub Action artifacts)
    stamp = datetime.now().strftime("%Y%m%d")
    if len(df) > 0:
        df.to_csv(f"giant_all_specials_{TARGET_STORE_NUM}_{stamp}.csv", index=False)
        df.to_excel(f"giant_all_specials_{TARGET_STORE_NUM}_{stamp}.xlsx", index=False)

    # Fail loudly on zero rows so Actions surfaces the problem
    if len(df) == 0:
        raise SystemExit("0 rows scraped. See logs and artifacts (page.png/page.html).")

    # Push to Google Sheets
    write_to_google_sheet(df)


if __name__ == "__main__":
    main()
