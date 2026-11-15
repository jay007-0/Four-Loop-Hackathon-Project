import os
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types

from coaching import EXERCISES

# Load API key from the parent HackUMass folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "trial.env")

load_dotenv(ENV_PATH)

MODEL = "gemini-3.5-flash-lite"

# The Whisper model in use (ggml-base.en.bin) is English-only, so feedback is in English.
# If you switch to the multilingual model, change this or detect the language.
FEEDBACK_LANGUAGE = "English"

AUDIENCE_GUIDANCE = {
    "nervous": (
        "This person gets nervous speaking in front of people. Be calm, "
        "reassuring and encouraging. Celebrate specific small wins, keep the "
        "improvement to ONE change, and make it feel doable. Do not label "
        "them as anxious or mention anxiety."
    ),
    "confident": (
        "This person is comfortable speaking and wants to sharpen their "
        "delivery. Be direct, efficient and candid. They can handle clear "
        "critique, so give more advanced tips such as emphasis, strategic "
        "pauses and structure."
    ),
}

# A random starting angle makes responses feel different from case to case.
OPENING_HINTS = [
    "Start by quoting a short phrase they actually said and what it did for a listener.",
    "Start with how the overall delivery would feel to someone in the audience.",
    "Start with the single strongest moment of the recording.",
    "Start with something specific about how they began the speech.",
    "Start with a concrete observation about their rhythm or flow.",
]

PROMPT = """You are an experienced, friendly public-speaking coach giving feedback on one practice recording.

{audience_guidance}

Transcript (automatic transcription, so it may contain small mistakes. Never point
out odd words or spelling, and only quote phrases that read naturally):
{transcript}

Observations about their delivery (for your understanding only, never repeat numbers):
{observations}

{focus_line}

Write the feedback as three short paragraphs, each beginning with a bold label:

**What came across well**: two or three specific strengths. Quote short phrases
(under 10 words) exactly as they appear in the transcript, in quotation marks,
and explain what each did for a listener. For example, a warm greeting can make
an audience feel welcome. Keep the tone warm and sincere, not gushing.

**What to try next**: explain the focus area in plain language and why it matters
to a listener. Then offer one concrete way to bring out a point they already made.
Take an idea from their speech and show how it could land even more strongly, for
example: One way to bring out that idea: "<a stronger version of the point>". This is
an extra option for them, not a correction. If the transcript is too short for that,
use a realistic example instead. 

**Try this next time**: one concrete exercise. {exercise}
Rephrase it naturally.

Length: about 220 to 300 words in total.

Style rules:
- {opening_hint}
- Be specific about what could be even better, but present every suggestion as
  something to add or try. Never say that anything they said was bad, weak,
  unclear, or wrong. Never quote their words as the "before" in a comparison,
  and never write "instead of". Offer the alternative wording on its own.
- The observations above are background for you. Do not repeat them as criticism or
  quote a phrase of theirs as the thing to fix.
- Never include numbers, units or measurements (no words per minute, decibels,
  semitones, seconds, percentages, or pitch values). Describe things in plain
  words instead, such as "you left room between sentences".
- Do not use any of these words: wonderful, wonderfully, lovely, fantastic,
  amazing, awesome, excellent, great job, well done, such a. Never repeat the same
  adjective twice, and use at most one exclamation mark.
- Do not start with "You" or with a praise word.
- Never diagnose or guess at any medical condition. Do not comment on accent or
  pronunciation.
- Use only paragraphs with the bold labels. No headings and no bullet lists.
- Write the whole feedback in {language}, including the bold labels. Quote their words
  exactly as transcribed, even if a phrase looks like another language.
"""


def get_feedback(transcript, obs, focus, user_chose_focus, audience="nervous"):
    """
    transcript: the text of what they said
    obs: list from coaching.observations()
    focus: 'pace' / 'pauses' / 'expression' / 'volume' / 'wording' / None
    user_chose_focus: True if the person picked this goal themselves
    audience: 'nervous' or 'confident'
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check that trial.env "
            "is in the parent HackUMass folder."
        )

    # Only the plain-language meaning is passed on, not the raw numbers,
    # so the feedback stays in words.
    obs_text = "\n".join(f"- {title}: {msg}" for _key, title, _value, _ok, msg in obs)

    if focus is None:
        focus_line = (
            "Everything looks solid, so there is no weak spot to fix. In the second "
            "paragraph, offer a stretch challenge instead."
        )
        exercise = "Suggest a small stretch challenge, such as adding one deliberate pause before the key point."
    elif user_chose_focus:
        focus_line = f"The person chose to work on: {focus}."
        exercise = EXERCISES[focus]
    else:
        focus_line = f"Focus area (picked automatically): {focus}."
        exercise = EXERCISES[focus]

    prompt = PROMPT.format(
        audience_guidance=AUDIENCE_GUIDANCE.get(audience, AUDIENCE_GUIDANCE["nervous"]),
        transcript=transcript,
        observations=obs_text,
        focus_line=focus_line,
        exercise=exercise,
        opening_hint=random.choice(OPENING_HINTS),
        language=FEEDBACK_LANGUAGE,
    )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.9),
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response. Try again.")

    return response.text