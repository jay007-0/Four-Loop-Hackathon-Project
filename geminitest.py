
import os
from dotenv import load_dotenv
from google import genai

print("Step 1: Python is running")

load_dotenv("trial.env")
print("Step 2: Environment file loaded")

api_key = os.getenv("GEMINI_API_KEY")
print("Step 3: API key found:", bool(api_key))

if not api_key:
    raise ValueError("API key not found!")

client = genai.Client(api_key=api_key)
print("Step 4: Gemini client initialized")

print("Step 5: Sending request...")

response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents="Hello Gemini! Introduce yourself."
)

print("Step 6: Response received")
print(response.text)
