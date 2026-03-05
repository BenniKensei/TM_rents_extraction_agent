import asyncio
import os
from dotenv import load_dotenv
from groq import AsyncGroq

load_dotenv()


async def main():
    client = AsyncGroq()
    models = await client.models.list()
    print("Available Groq Models:")
    for m in models.data:
        print(m.id)


if __name__ == "__main__":
    asyncio.run(main())
