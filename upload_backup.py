import pandas as pd
import asyncio
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def upload_backup():
    print("Loading local CSV backup...")
    try:
        df = pd.read_csv("timisoara_rents_extracted.csv")
        # Automatically scrub any NaN (null) values that pandas assigns to empty cells, avoiding math casting crashes
        df['monthly_rent_eur'] = df['monthly_rent_eur'].fillna(0)
        df['rooms'] = df['rooms'].fillna(1)
        df['is_pet_friendly'] = df['is_pet_friendly'].fillna(False)
        df['title'] = df['title'].fillna("Unknown")
        df['neighborhood'] = df['neighborhood'].fillna("Unknown")
    except Exception as e:
        print(f"Could not load CSV: {e}")
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("DATABASE_URL not found!")
        return

    print("Connecting to Supabase via PgBouncer Pooler...")
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

    print("Uploading 164 records to database natively...")
    # Seed individual rows logically
    for _, row in df.iterrows():
        await conn.execute(
            """
            INSERT INTO timisoara_rents 
                (title, monthly_rent_eur, neighborhood, rooms, is_pet_friendly)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (title, neighborhood) DO NOTHING;
            """,
            str(row.get('title', 'Unknown')),
            int(row.get('monthly_rent_eur', 0)),
            str(row.get('neighborhood', 'Unknown')),
            int(row.get('rooms', 1)),
            bool(row.get('is_pet_friendly', False))
        )
        
    await conn.close()
    print("✅ All local data successfully seeded into your new Supabase PostgreSQL database!")

if __name__ == "__main__":
    asyncio.run(upload_backup())
