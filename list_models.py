import os
import urllib.request
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {api_key}"},
)
try:
    with urllib.request.urlopen(req) as res:
        data = json.loads(res.read())
        models = [m["id"] for m in data.get("data", [])]
        print("Available models:")
        for m in models:
            print(f"- {m}")
except Exception as e:
    print(f"Error: {e}")
