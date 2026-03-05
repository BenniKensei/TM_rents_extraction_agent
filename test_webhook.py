import asyncio
import os
import aiohttp
from dotenv import load_dotenv

load_dotenv()


async def test_webhook():
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("FAIL: DISCORD_WEBHOOK_URL is not set.")
        return

    print("Testing Discord webhook...")

    content = (
        f"🚨 **TEST: New Apartment Alert!** 🚨\n"
        f"**Title:** Apartament 2 camere deosebit (TEST)\n"
        f"**Price:** 350 EUR\n"
        f"**Location:** Complexul Studentesc\n"
        f"**Rooms:** 2\n\n"
        f"*(This is an automated test to verify the webhook connection)*"
    )

    payload = {"content": content}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status in (200, 204):
                    print("✅ SUCCESS: Test message sent securely to Discord!")
                else:
                    print(f"❌ Failed to send alert. Status: {response.status}")
    except Exception as e:
        print(f"❌ ERROR sending alert: {e}")


if __name__ == "__main__":
    asyncio.run(test_webhook())
