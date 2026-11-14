
import os

from dotenv import load_dotenv
from google import genai


# Load API key from the parent HackUMass folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "trial.env")

load_dotenv(ENV_PATH)

MODEL = "gemini-3.5-flash-lite"

PROMPT = """You are a speaking coach. Below is a transcript of someone speaking.

Give 3 to 5 short, practical recommendations to make the message clearer.
Then write a clearer version of the same message.

Format your answer in Markdown:
Use bullet points for the recommendations, with the key idea in bold.
After the bullets, add a line that starts with Clearer version:
Answer in the same language as the transcript.

Transcript:
{transcript}
"""


def get_recommendations(transcript):
    """Takes the transcript. Returns recommendations as Markdown text."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check that trial.env "
            "is in the parent HackUMass folder."
        )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT.format(transcript=transcript),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response. Try again.")

    return response.text
