import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()


async def test_db():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("FAIL: DATABASE_URL is not set.")
        return

    print(
        f"Testing connection to: {database_url.split('@')[1] if '@' in database_url else '***'}"
    )

    try:
        conn = await asyncpg.connect(database_url)
        print("✅ SUCCESS: Connected to the database successfully!")

        # Check if the table exists
        exists = await conn.fetchval(
            "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'timisoara_rents');"
        )
        print(f"📊 Table 'timisoara_rents' exists: {exists}")

        await conn.close()
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to the database. Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_db())
