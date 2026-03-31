import asyncio
import json
import os
import aiohttp
import asyncpg
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from playwright.async_api import async_playwright

load_dotenv()


class ApartmentListing(BaseModel):
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
    listings: list[ApartmentListing]


async def scrape_pages(base_url: str, num_pages: int = 5) -> list[str]:
    """Launch a headed browser with stealth settings and extract page text across multiple pages."""
    texts = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-blink-features=AutomationControlled"],
        )
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

        # Remove the webdriver flag that many sites check
        await page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        for current_page in range(1, num_pages + 1):
            url = base_url if current_page == 1 else f"{base_url}?page={current_page}"
            print(f"Navigating to {url}...")

            await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)

            # Accept cookie consent if present
            for selector in [
                "button#onetrust-accept-btn-handler",
                "button:has-text('Accept')",
                "button:has-text('Acceptă')",
                "button:has-text('Accept all')",
            ]:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click()
                        print(f"Clicked cookie consent: {selector}")
                        await page.wait_for_timeout(1500)
                        break
                except Exception:
                    continue

            # Wait for listings to render
            await page.wait_for_timeout(3000)

            # Scroll incrementally to trigger lazy loading
            for i in range(5):
                await page.evaluate(f"window.scrollTo(0, {(i + 1) * 2000})")
                await page.wait_for_timeout(800)

            # Extract text from the page
            text_content = await page.evaluate("""() => {
                const remove = document.querySelectorAll(
                    'script, style, noscript, svg, img, header, footer, nav'
                );
                remove.forEach(el => el.remove());
                return document.body.innerText;
            }""")

            print(
                f"Extracted {len(text_content)} characters of text from page {current_page}."
            )
            texts.append(text_content)

            if current_page < num_pages:
                await page.wait_for_timeout(2000)

        await browser.close()

    return texts


async def extract_with_slm(text: str) -> Listings | None:
    """Send the extracted text to a local Ollama SLM for structured extraction."""
    schema = Listings.model_json_schema()

    print("Sending text to local Ollama (llama3.2) for structured extraction...")
    
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
                    f"{text[:10000]}"
                ),
            },
        ],
        "format": schema, # Enforces strict JSON Schema rendering natively in Ollama 0.4.0+
        "stream": False,
        "options": {
            "temperature": 0.0
        }
    }

    try:
        # Give the local SLM 120 seconds to process the large text block context
        async with aiohttp.ClientSession() as session:
            async with session.post("http://localhost:11434/api/chat", json=payload, timeout=120) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Ollama API Error {response.status}: {error_text}")
                    return None
                
                result = await response.json()
                raw_json = result["message"]["content"]
                
                # Sanitize markdown ticks if the SLM accidentally wraps the JSON output
                raw_json = raw_json.strip()
                if raw_json.startswith("```json"):
                    raw_json = raw_json[7:]
                if raw_json.endswith("```"):
                    raw_json = raw_json[:-3]
                
                # Attempt to parse and auto-correct structural SLM hallucinations
                parsed_dict = json.loads(raw_json)
                
                # If the SLM hallucinated and forgot the root "listings" array wrapper
                if "listings" not in parsed_dict:
                    if isinstance(parsed_dict, list):
                        parsed_dict = {"listings": parsed_dict}
                    elif isinstance(parsed_dict, dict) and "title" in parsed_dict:
                        parsed_dict = {"listings": [parsed_dict]}
                    else:
                        parsed_dict = {"listings": []}

                parsed = Listings.model_validate(parsed_dict)
                return parsed
    except aiohttp.ClientError as e:
        print(f"Failed to connect to Ollama. Ensure Ollama is running (`ollama run llama3.2`): {e}")
        return None
    except Exception as e:
        print(f"Error parsing local SLM response: {e}")
        return None


async def evaluate_and_alert(listing: ApartmentListing) -> None:
    """Evaluate a listing against criteria and send a Discord alert if met."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return

    if listing.monthly_rent_eur <= 350 and "Complexul" in listing.neighborhood:
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
                        print(f"Failed to send Discord alert: {response.status}")
        except Exception as e:
            print(f"Error sending Discord alert: {e}")


async def save_to_db(data: Listings) -> None:
    """Persist extracted listings into a PostgreSQL database URL."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not found in environment variables.")
        return

    conn = await asyncpg.connect(database_url, statement_cache_size=0)

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
    print(f"Saved {len(data.listings)} listings to the database.")


async def main():
    base_url = (
        "https://www.storia.ro/ro/rezultate/inchiriere/apartament/timis/timisoara"
    )

    texts = await scrape_pages(base_url, num_pages=5)

    for i, text in enumerate(texts, start=1):
        print(f"\n--- Processing Data from Page {i} ---")
        if len(text) < 200:
            print(f"Error: Extracted text is too short for page {i}. Skipping.")
            continue

        listings_data = await extract_with_slm(text)
        if listings_data:
            print(
                f"Successfully extracted {len(listings_data.listings)} listings from page {i}:\n"
            )
            print(listings_data.model_dump_json(indent=2))
            for listing in listings_data.listings:
                await evaluate_and_alert(listing)
            await save_to_db(listings_data)
        else:
            print(f"Error: No listings could be extracted from page {i}.")


if __name__ == "__main__":
    asyncio.run(main())
