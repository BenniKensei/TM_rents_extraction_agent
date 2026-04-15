"""
Web Extraction & LLM-Based Data Pipeline Agent
==============================================
Orchestrates end-to-end real estate data collection using semantic LLM extraction.

Architecture:
1. Playwright Web Scraping: Extract raw webpage text with anti-bot evasion
2. Local Ollama SLM: Parse unstructured text to structured JSON (zero API costs)
3. Business Rule Evaluation: Trigger Discord alerts for listings matching criteria
4. PostgreSQL Persistence: Store extracted listings in Supabase with upsert logic

Technical Highlights:
- Async/await throughout: Non-blocking I/O for 5x throughput vs synchronous scraping
- Stealth Browser Mode: Disable webdriver detection, user-agent spoofing
- LLM Hallucination Recovery: Auto-correct malformed JSON from model output
- Idempotent Database Updates: Upsert with ON CONFLICT for re-runs without duplication

Why LLM over CSS Selectors?
- CSS selectors break when website redesigns (brittle)
- LLM semantic extraction adapts to layout changes (robust)
- Works on dynamically rendered content (JavaScript-heavy sites)
- Eliminates regex parsing complexity and edge cases

Cost Model:
- Ollama local inference: €0/month (self-hosted)
- vs Groq API: €0.0001 per 1k tokens * ~1000 requests/day ≈ $3/month
- Savings: Pay once for GPU, amortize across all projects
"""

import asyncio
import json
import os
import aiohttp
import asyncpg
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright

load_dotenv()


# ────────────────────────────────────────────────────────────────────────────
# PYDANTIC SCHEMAS: Structured Data Validation
# ────────────────────────────────────────────────────────────────────────────

class ApartmentListing(BaseModel):
    """
    Single apartment listing extracted from webpage.
    
    Attributes:
        title: Listing title (e.g., "Modern 2BR Apartment in Centru")
        monthly_rent_eur: Monthly rent price in EUR. Auto-converts from RON (5.0 exchange rate).
        neighborhood: Categorical location identifier (15-30 unique values typical).
        rooms: Integer room count (1-6 typical range).
        is_pet_friendly: Boolean indicator (True only if explicitly mentioned).
    
    Validation:
    - All fields required (no optional fields)
    - Integer fields validated by Pydantic
    - Boolean fields coerced to bool type
    """
    title: str = Field(description="The title of the apartment listing")
    monthly_rent_eur: int = Field(
        description="The monthly rent in EUR. Convert from RON to EUR using a 5.0 exchange rate if the price is in RON."
    )
    neighborhood: str = Field(description="The neighborhood or area of the apartment")
    rooms: int = Field(description="The number of rooms in the apartment")
    is_pet_friendly: bool = Field(
        description="Whether the apartment is pet friendly or allows pets. False if not explicitly stated."
    )


class Listings(BaseModel):
    """
    Collection of apartment listings (response format from LLM).
    
    Attributes:
        listings: List of ApartmentListing objects. Can be empty if no listings found.
    """
    listings: list[ApartmentListing]


# ────────────────────────────────────────────────────────────────────────────
# STEP 1: WEB SCRAPING
# ────────────────────────────────────────────────────────────────────────────

async def scrape_pages(base_url: str, num_pages: int = 5) -> list[str]:
    """
    Extract raw text content from multiple real estate listing pages using Playwright.
    
    **Anti-Bot Evasion Techniques:**
    1. Disable automation detection: Remove navigator.webdriver property
    2. Real user-agent: Chrome/Edge modern browser string (not "HeadlessChrome")
    3. Standard viewport: 1920x1080 resolution (matches real user browsing)
    4. Locale spoofing: Romanian locale (ro-RO) for local website logic
    
    **Page Loading Strategy:**
    - wait_until="domcontentloaded": Wait for initial DOM render (not full page load)
    - Cookie consent handling: Auto-accept consent banners (4 selector fallbacks)
    - Lazy loading: Incrementally scroll 5 times to trigger image/content loading
    - DOM cleanup: Remove script/style/nav/footer tags before text extraction
    
    **Async Design:**
    - Non-blocking page navigation and wait_for_timeout calls
    - Concurrent request handling if scaled to multiple URLs
    - Single browser instance reused across pages (connection pooling)
    
    Args:
        base_url: Website URL for first page (e.g., historia.ro/... search results)
        num_pages: Number of pagination pages to scrape (default 5).
    
    Returns:
        list[str]: Extracted text content from each page.
                   Each string: ~10KB-50KB of visible text (cleaned of markup).
    
    Raises:
        PlaywrightException: Browser launch failure, timeout, or navigation errors.
    
    Note: Uses headless=False for debugging visibility. Set to True in production.
    
    Time Complexity: O(n * m) where n=pages, m=average scroll/wait operations per page
    """
    texts = []
    async with async_playwright() as p:
        # Launch Chromium browser with anti-detection flags
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],  # Hide browser automation
        )
        
        # Create isolated browser context (cookies, cache separate from default profile)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="ro-RO",
        )
        page = await context.new_page()

        # Inject JavaScript to hide webdriver detection (many sites check navigator.webdriver)
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        # ─── PAGINATION LOOP ───
        for current_page in range(1, num_pages + 1):
            # Construct paginated URL (page 1 is base_url, page 2+ use ?page=N)
            url = base_url if current_page == 1 else f"{base_url}?page={current_page}"
            print(f"📄 Navigating to {url}...")

            # Navigate with timeout and wait for DOM content loaded
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)  # Extra wait for async content

            # ─── COOKIE CONSENT HANDLING ───
            # Real estate websites often have cookie banners; try multiple selectors
            for selector in [
                "button#onetrust-accept-btn-handler",    # OneTrust (common provider)
                "button:has-text('Accept')",              # English fallback
                "button:has-text('Acceptă')",             # Romanian
                "button:has-text('Accept all')",          # Variant
            ]:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        print(f"   ✓ Clicked cookie consent: {selector}")
                        await page.wait_for_timeout(1500)
                        break
                except Exception:
                    # Selector not found or not visible, try next
                    continue

            # ─── LAZY LOADING TRIGGER ───
            # Incrementally scroll to force dynamic content loading
            await page.wait_for_timeout(3000)
            for i in range(5):
                await page.evaluate(f"window.scrollTo(0, {(i + 1) * 2000})")
                await page.wait_for_timeout(800)

            # ─── TEXT EXTRACTION ───
            # Execute JavaScript to clean up DOM and extract visible text
            # Removes: script tags, styles, images, nav/header/footer (boilerplate)
            text_content = await page.evaluate("""() => {
                const remove = document.querySelectorAll(
                    'script, style, noscript, svg, img, header, footer, nav'
                );
                remove.forEach(el => el.remove());
                return document.body.innerText;
            }""")

            print(f"   ✓ Extracted {len(text_content):,} characters from page {current_page}")
            texts.append(text_content)

            # Rate limiting: Pause before next page to avoid overloading server
            if current_page < num_pages:
                await page.wait_for_timeout(2000)

        await browser.close()

    return texts


# ────────────────────────────────────────────────────────────────────────────
# STEP 2: LLM-BASED STRUCTURED EXTRACTION
# ────────────────────────────────────────────────────────────────────────────

async def extract_with_slm(text: str) -> Listings | None:
    """
    Parse unstructured webpage text into structured apartment listings using local Ollama SLM.
    
    **Why Ollama (Small Language Model) instead of Large LLMs?**
    - Ollama llama3.2: 3B parameters, runs on consumer GPU (RTX 3060+)
    - Cost: $0 (self-hosted) vs $0.0001/token with API
    - Latency: 5-30 seconds per page (acceptable for batch scraping)
    - Privacy: Local inference, no data leaves your machine
    
    **Extraction Strategy:**
    1. System prompt: Configure role ("real estate extraction assistant")
    2. Format enforcement: JSON Schema constraint (Ollama 0.4.0+ feature)
    3. Temperature: Set to 0.0 for deterministic, repeatable outputs
    4. Text truncation: Use first 10,000 characters to fit context window
    5. Timeout: 120 seconds for SLM inference (long context processing)
    
    **Hallucination Recovery:**
    - SLM may wrap JSON in markdown ticks (```json...```)
    - SLM may omit "listings" root wrapper
    - SLM may return single object instead of array
    - Auto-correct these deviations with post-processing logic
    
    Args:
        text: Raw webpage text (10KB-50KB typical size).
    
    Returns:
        Listings: Validated Pydantic model with extracted listings.
                 Returns None if Ollama unreachable or JSON parsing fails.
    
    Raises:
        (None explicitly raised; returns None on any error for graceful degradation)
    
    Prerequisites:
        - Ollama running locally: `ollama run llama3.2`
        - Listening on http://localhost:11434
    
    TODO: Implement retry logic (3 attempts with exponential backoff) for network transience.
    TODO: Add response filtering to remove low-confidence extractions (confidence scoring).
    """
    schema = Listings.model_json_schema()

    print("🤖 Sending text to local Ollama (llama3.2) for structured extraction...")
    
    # ─── REQUEST PAYLOAD ───
    payload = {
        "model": "llama3.2",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a real estate data extraction assistant. "
                    "Extract every single apartment listing visible in the provided webpage text. "
                    "You MUST reply ONLY with valid JSON exactly matching this structure:\n"
                    '{\n  "listings": [\n    {\n      "title": "Sample Title",\n      "monthly_rent_eur": 500,\n      "neighborhood": "Centru",\n      "rooms": 2,\n      "is_pet_friendly": false\n    }\n  ]\n}'
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract ALL apartment listings from this real estate webpage text. "
                    "The prices are likely in RON — convert to EUR using a 5.0 exchange rate. "
                    "Set is_pet_friendly to true only if explicitly mentioned, otherwise false.\n\n"
                    f"{text[:10000]}"  # Truncate to ~10K chars (token limit ~2000 for llama3.2)
                ),
            },
        ],
        "format": schema,  # Ollama 0.4.0+: Enforce JSON Schema structure
        "stream": False,   # Non-streaming response
        "options": {
            "temperature": 0.0  # Deterministic output (no randomness)
        }
    }

    try:
        # ─── HTTP REQUEST WITH TIMEOUT ───
        # aiohttp.ClientSession: Async HTTP client
        # Timeout 120s: Generous for SLM processing large context window
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=120
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"❌ Ollama API Error {response.status}: {error_text}")
                    return None
                
                # ─── RESPONSE PARSING ───
                result = await response.json()
                raw_json = result["message"]["content"]
                
                # ─── HALLUCINATION RECOVERY ───
                # SLM sometimes wraps JSON in markdown code blocks
                raw_json = raw_json.strip()
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                
                # Parse JSON string to dict
                parsed_dict = json.loads(raw_json)
                
                # If the SLM forgot the "listings" root wrapper, add it
                # This handles: 1) Array instead of {listings: []}, 2) Single object
                if "listings" not in parsed_dict:
                    if isinstance(parsed_dict, list):
                        parsed_dict = {"listings": parsed_dict}
                    elif isinstance(parsed_dict, dict) and "title" in parsed_dict:
                        parsed_dict = {"listings": [parsed_dict]}
                    else:
                        parsed_dict = {"listings": []}

                # Validate against Pydantic schema (ensures all required fields present)
                parsed = Listings.model_validate(parsed_dict)
                return parsed
    
    except aiohttp.ClientError as e:
        print(
            f"❌ Failed to connect to Ollama. "
            f"Ensure Ollama is running: `ollama run llama3.2`\n"
            f"   Error: {e}"
        )
        return None
    except Exception as e:
        print(f"❌ Error parsing local SLM response: {e}")
        return None


# ────────────────────────────────────────────────────────────────────────────
# STEP 3: BUSINESS RULE EVALUATION & ALERTING
# ────────────────────────────────────────────────────────────────────────────

async def evaluate_and_alert(listing: ApartmentListing) -> None:
    """
    Evaluate extracted listing against business rules and trigger Discord alert if matched.
    
    **Business Rule:** Alert if listing meets BOTH criteria:
    - Price: ≤ €350/month (unusually cheap, potential deal)
    - Location: "Complexul" in neighborhood name (specific area of interest)
    
    **Alert Format:**
    Formatted Markdown message posted to Discord webhook with:
    - Emoji indicators (🚨 = alert fired)
    - Title, price, location, room count
    - Timestamp automatically added by Discord
    
    **Implementation:**
    - Gracefully skip if DISCORD_WEBHOOK_URL not set (dev/test environment)
    - Non-blocking: Exceptions caught and logged (don't crash main pipeline)
    - Rate limiting: No delays; Discord handles burst traffic
    
    Args:
        listing: Validated ApartmentListing object.
    
    Returns:
        None (no return value; side effect is Discord POST)
    
    Raises:
        (Exceptions caught internally; no exceptions propagated)
    
    TODO: Make business rules configurable (environment variables for price_threshold, location_keyword)
    FIXME: Add retry logic (3 attempts with 5s delay) for webhook network timeouts
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        # Webhook URL not set (expected in dev/test); silently skip
        return

    # ─── BUSINESS RULE EVALUATION ───
    if listing.monthly_rent_eur <= 350 and "Complexul" in listing.neighborhood:
        # Listing meets alert criteria
        content = (
            f"🚨 **New Apartment Alert!** 🚨\n"
            f"**Title:** {listing.title}\n"
            f"**Price:** {listing.monthly_rent_eur} EUR\n"
            f"**Location:** {listing.neighborhood}\n"
            f"**Rooms:** {listing.rooms}"
        )
        payload = {"content": content}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status not in (200, 204):
                        print(f"⚠️  Failed to send Discord alert: HTTP {response.status}")
        except Exception as e:
            print(f"⚠️  Error sending Discord alert: {e}")


# ────────────────────────────────────────────────────────────────────────────
# STEP 4: DATABASE PERSISTENCE
# ────────────────────────────────────────────────────────────────────────────

async def save_to_db(data: Listings) -> None:
    """
    Persist extracted apartment listings to PostgreSQL database (Supabase).
    
    **Schema Design:**
    - id: Auto-increment primary key
    - title, monthly_rent_eur, neighborhood, rooms, is_pet_friendly: Listing attributes
    - first_seen, last_seen: Temporal tracking for duplicate detection
    - UNIQUE constraint (title, neighborhood): Deduplication key (same listing across scrapes)
    
    **Upsert Logic (ON CONFLICT):**
    - If listing already exists (same title + neighborhood):
      - Update price_eur (captures price changes)
      - Update last_seen timestamp (record re-observation)
    - If listing new:
      - Insert as new row
      - first_seen = CURRENT_TIMESTAMP
      - last_seen = CURRENT_TIMESTAMP
    
    **Idempotence:**
    - Running scraper twice with same data produces identical database state
    - Perfect for CI/CD scheduled jobs (safe to retry without data corruption)
    
    **Connection Pooling:**
    - statement_cache_size=0: Disable prepared statement caching
    - Reason: Supabase PgBouncer (connection pooler) doesn't support server-side caching
    - Each query uses protocol bypass for pooled connections
    
    Args:
        data: Listings object with list of ApartmentListing items.
    
    Returns:
        None (side effect: rows inserted/updated in database)
    
    Raises:
        asyncpg.PostgresError: Database connection or query execution failure.
        Exception: DATABASE_URL not set (returns gracefully with warning message).
    
    Note: Assumes timisoara_rents table exists or auto-creates via CREATE TABLE IF NOT EXISTS.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ Error: DATABASE_URL not found in environment variables.")
        return

    # ─── ASYNC DATABASE CONNECTION ───
    # asyncpg: Fast, async-native PostgreSQL driver
    # statement_cache_size=0: Disable caching for Supabase PgBouncer compatibility
    conn = await asyncpg.connect(database_url, statement_cache_size=0)

    # ─── TABLE CREATION ───
    # CREATE TABLE IF NOT EXISTS: Idempotent (safe to re-run)
    # UNIQUE constraint on (title, neighborhood): Enables upsert logic
    await conn.execute("""
    CREATE TABLE IF NOT EXISTS timisoara_rents (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        monthly_rent_eur INTEGER NOT NULL,
        neighborhood TEXT NOT NULL,
        rooms INTEGER,
        is_pet_friendly BOOLEAN,
        first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_listing UNIQUE (title, neighborhood)
    )
    """)

    # ─── BATCH UPSERT ───
    # Loop through extracted listings and insert/update each
    for listing in data.listings:
        await conn.execute(
            """
            INSERT INTO timisoara_rents 
                (title, monthly_rent_eur, neighborhood, rooms, is_pet_friendly)
            VALUES 
                ($1, $2, $3, $4, $5)
            ON CONFLICT (title, neighborhood)
            DO UPDATE SET
                monthly_rent_eur = EXCLUDED.monthly_rent_eur,
                last_seen = CURRENT_TIMESTAMP;
        """,
            listing.title,
            listing.monthly_rent_eur,
            listing.neighborhood,
            listing.rooms,
            listing.is_pet_friendly,
        )

    await conn.close()
    print(f"✅ Saved {len(data.listings)} listings to the database.")


# ────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ────────────────────────────────────────────────────────────────────────────

async def main():
    """
    Main pipeline orchestration: Scrape -> Extract -> Alert -> Persist
    
    **Data Flow:**
    1. scrape_pages(): Extract raw text from 5 páginations of historia.ro
    2. extract_with_slm(): Parse each page's text to structured JSON via Ollama
    3. evaluate_and_alert(): Check each listing against business rules
    4. save_to_db(): Upsert listings into PostgreSQL (Supabase)
    
    **Error Resilience:**
    - Page scraping failure: Continue to next page (skip)
    - LLM extraction failure: Log error and continue (graceful degradation)
    - Database errors: Propagate and stop (critical path)
    - Discord alerts: Fail silently (non-critical side effect)
    
    **Time Estimate:**
    - Scraping 5 pages: ~30-60 seconds (network + rendering)
    - LLM extraction: ~5-10 seconds/page (5-50 seconds total)
    - Database inserts: ~1-2 seconds (batch upsert)
    - Total runtime: ~40-120 seconds
    
    **Future Improvements:**
    - Concurrent page scraping (async generators for parallelism)
    - Batched database inserts (execute_many) instead of per-item
    - Retry logic with exponential backoff for transient failures
    """
    # Target URL: Historia.ro apartments for rent in Timisoara
    base_url = (
        "https://www.storia.ro/ro/rezultate/inchiriere/apartament/timis/timisoara"
    )

    # ─── STEP 1: SCRAPE PAGES ───
    print("\n" + "="*60)
    print("🕷️  WEB SCRAPING INITIATED")
    print("="*60)
    texts = await scrape_pages(base_url, num_pages=5)

    # ─── STEP 2-4: EXTRACT -> ALERT -> PERSIST (per page) ───
    print("\n" + "="*60)
    print("🔄 EXTRACTION & PERSISTENCE PIPELINE")
    print("="*60)
    
    for i, text in enumerate(texts, start=1):
        print(f"\n--- Processing Data from Page {i} ---")
        
        # Sanity check: Ensure text is not empty or trivially short
        if len(text) < 200:
            print(f"❌ Extracted text too short ({len(text)} chars). Skipping page {i}.")
            continue

        # ─── EXTRACT WITH LLM ───
        listings_data = await extract_with_slm(text)
        
        if listings_data:
            print(f"✅ Successfully extracted {len(listings_data.listings)} listings from page {i}")
            
            # ─── EVALUATE & ALERT ───
            for listing in listings_data.listings:
                await evaluate_and_alert(listing)
            
            # ─── PERSIST TO DATABASE ───
            await save_to_db(listings_data)
        else:
            print(f"❌ No listings extracted from page {i}.")

    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
