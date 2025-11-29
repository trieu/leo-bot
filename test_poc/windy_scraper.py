#!/usr/bin/env python3
"""
Windy Forecast Scraper — Firefox / accuracy-first
Saves LLM-ready structured text to file.txt

Usage:
    python windy_scraper_firefox.py [lat] [lon]

Requirements:
    pip install playwright
    playwright install firefox
"""

import sys
import asyncio
from datetime import datetime
from pathlib import Path
from playwright.async_api import async_playwright, TimeoutError

# Defaults
DEFAULT_LAT = 10.776
DEFAULT_LON = 106.702
DEFAULT_ZOOM = 10

# The element we want to extract text from (Table View)
CSS_SELECTOR = "table.forecast-table__table"

# The element we wait for to ensure the page is fully rendered (Meteogram Canvas)
CANVAS_SELECTOR = "canvas.forecast-table__canvas"

# Desktop UA
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/117 Safari/537.36"
)

# Tunables
NAV_TIMEOUT = 90_000       # 90s
WAIT_TIMEOUT = 45_000      # 45s
RETRIES = 2
DEBUG_SCREENSHOT = "windy_debug.png"


def build_url(lat, lon, zoom):
    return f"https://www.windy.com/{lat}/{lon}?satellite,{lat},{lon},{zoom}"


async def run_once(page, url):
    """
    Navigate and wait until the canvas is rendered, then extract text.
    Returns inner_text or raises on failure.
    """
    # 1. Navigate
    await page.goto(url, wait_until="domcontentloaded", timeout=NAV_TIMEOUT)

    # 2. Wait for the Canvas to be fully loaded
    # We define "fully loaded" as: element exists AND has non-zero dimensions
    print("Waiting for forecast canvas to render...")
    try:
        await page.wait_for_function(
            f"""() => {{
                const canvas = document.querySelector('{CANVAS_SELECTOR}');
                if (!canvas) return false;
                // Check if canvas has been drawn (has dimensions)
                return canvas.width > 0 && canvas.height > 0;
            }}""",
            timeout=WAIT_TIMEOUT
        )
    except TimeoutError:
        raise TimeoutError(
            f"Canvas '{CANVAS_SELECTOR}' did not load within {WAIT_TIMEOUT}ms")

    # 3. Extract Text from the data table
    table_node = await page.query_selector(CSS_SELECTOR)
    if not table_node:
        raise RuntimeError(
            f"Canvas loaded, but text selector '{CSS_SELECTOR}' not found."
        )

    # 👉 Delete 3rd row in the table BEFORE extracting HTML
    await page.evaluate(
        """(selector) => {
            const table = document.querySelector(selector);
            if (!table) return;

            const rows = table.querySelectorAll('tr');
            if (rows.length >= 3) {
                rows[2].remove();
            }
        }""",
        CSS_SELECTOR
    )

    # Now safely extract clean HTML
    return await table_node.inner_html()



async def get_windy_forecast():
    # Parse CLI
    try:
        lat = float(sys.argv[1])
        lon = float(sys.argv[2])
    except (IndexError, ValueError):
        lat = DEFAULT_LAT
        lon = DEFAULT_LON

    url = build_url(lat, lon, DEFAULT_ZOOM)
    print(f"Scraping Windy Forecast @ {lat}, {lon}")
    print(f"→ {url}")

    async with async_playwright() as p:
        # Launch Firefox
        browser = await p.firefox.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox"],
        )

        context = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1366, "height": 768},
            java_script_enabled=True,
            device_scale_factor=1,
        )

        page = await context.new_page()

        last_error = None
        inner_text = None

        for attempt in range(1, RETRIES + 2):
            try:
                print(f"Attempt {attempt} — navigating...")
                inner_text = await run_once(page, url)
                break  # Success
            except Exception as exc:
                last_error = exc
                print(f"Attempt {attempt} failed: {exc}")
                await asyncio.sleep(2 * attempt)
                try:
                    await page.reload(wait_until="domcontentloaded", timeout=30_000)
                except Exception:
                    pass

        # Handle Final Failure
        if inner_text is None:
            print("❌ Final failure — capturing debugging screenshot.")
            try:
                await page.screenshot(path=DEBUG_SCREENSHOT, full_page=True)
                print(f"Screenshot saved to {DEBUG_SCREENSHOT}")
            except Exception:
                pass
            await browser.close()
            sys.exit(1)

        await browser.close()

    # Build Output
    timestamp = datetime.now().isoformat()
    formatted = f"""
# Windy Forecast — LLM Parsing Output
timestamp: {timestamp}
latitude: {lat}
longitude: {lon}
source_url: {url}

## Table Meta-data

Table has total 7 rows:

- Row 1: Date in python format %A %d
- Row 2: Hour in 24h format
- Row 3: Temperature in Celsius
- Row 4: Rain in mm
- Row 5: Wind in m/s
- Row 6: Wind gusts in m/s
- Row 7: Wind direction

## Table Raw Data
<table class="forecast-table__table" >
{inner_text}
</table>

# end_of_record
    """.strip()

    Path("file2.txt").write_text(formatted, encoding="utf-8")
    print("✔ Forecast extracted and saved to file2.txt")

if __name__ == "__main__":
    asyncio.run(get_windy_forecast())
