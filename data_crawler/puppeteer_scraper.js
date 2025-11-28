/**
 * Windy Forecast Scraper → Saves LLM-ready text into file.txt
 *
 * Run:
 *   npm install puppeteer
 *   node windy_scraper.js 10.776 106.702
 */

const fs = require("fs");
const puppeteer = require("puppeteer");

// Defaults
const DEFAULT_LAT = 10.776;
const DEFAULT_LON = 106.702;
const DEFAULT_ZOOM = 10;
const CSS_SELECTOR = '#plugin-detail > section.main-table.bg-white.notap.svelte-zjesw3';

// Build URL
function buildUrl(lat, lon, zoom) {
  return `https://www.windy.com/${lat}/${lon}?satellite,${lat},${lon},${zoom}`;
}

async function getWindyForecast() {
  const lat = parseFloat(process.argv[2]) || DEFAULT_LAT;
  const lon = parseFloat(process.argv[3]) || DEFAULT_LON;
  const url = buildUrl(lat, lon, DEFAULT_ZOOM);

  console.log(`Scraping Windy Forecast @ ${lat}, ${lon}`);
  console.log(`→ ${url}`);

  let browser;

  try {
    browser = await puppeteer.launch({
      headless: true,
      args: ["--no-sandbox", "--disable-setuid-sandbox"],
    });

    const page = await browser.newPage();
    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117 Safari/537.36"
    );

    await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });
    await page.waitForSelector(CSS_SELECTOR, { timeout: 20000 });

    const innerText = await page.evaluate((selector) => {
      const el = document.querySelector(selector);
      return el ? el.innerText : "ERROR: No content found.";
    }, CSS_SELECTOR);

    // Build LLM-ready structured text
    const timestamp = new Date().toISOString();

    const formatted = `
# Windy Forecast — LLM Parsing Output
timestamp: ${timestamp}
latitude: ${lat}
longitude: ${lon}
source_url: ${url}

## forecast_raw_text:
${innerText}

# end_of_record
`.trim();

    // Write to file.txt
    fs.writeFileSync("file.txt", formatted, "utf8");

    console.log("✔ Forecast extracted and saved to file.txt");

  } catch (err) {
    console.error("❌ Scrape error:", err.message);
  } finally {
    if (browser) await browser.close();
  }
}

getWindyForecast();
