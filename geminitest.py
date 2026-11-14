#Ashrith
import os
from dotenv import load_dotenv
from google import genai
from speech_to_text import transcribe_audio

# Load Gemini API key
load_dotenv("../trial.env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key not found!")

client = genai.Client(api_key=api_key)

# Path to your WAV audio file
audio_path = "sample.wav"

# Convert speech to text using Whisper
transcription = transcribe_audio(audio_path)

print("Transcription:", transcription)

# Send transcription to Gemini
response = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=f"""
    You are a voice assistant.

    Understand the following speech and identify the user's intent.
    Extract important details such as dates, times, names and actions.

    Speech: {transcription}
    """
)

print("\nGemini response:")
print(response.text)
