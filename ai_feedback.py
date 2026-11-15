
import os
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types

from coaching import EXERCISES, exercise_for

# Load API key from the parent HackUMass folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", "trial.env")

load_dotenv(ENV_PATH)

MODEL = "gemini-3.5-flash-lite"

# The Whisper model in use (ggml-base.en.bin) is English-only, so feedback is in English.
# If you switch to the multilingual model, change this or detect the language.
FEEDBACK_LANGUAGE = "English"

# The two audiences differ in voice, length, and depth, not just a polite adjective.
AUDIENCES = {
    "nervous": {
        "guidance": (
            "VOICE: a warm, steady friend who is also a coach. This person gets nervous "
            "speaking in front of people. Use short sentences and plain everyday words, "
            "with no speaking jargon. Make the strengths feel real and specific. Suggest "
            "exactly ONE change, make it feel small and doable, and end on a calm, "
            "encouraging note that leaves them wanting to try again. Do not label them as "
            "anxious or mention anxiety or nerves."
        ),
        "length": "170 to 220 words in total",
        "sentences": "three or four",
        "refinements": "one concrete way",
        "stretch": "Open your next recording with your main point in one clear sentence, then take one full breath before your first detail.",
    },
    "confident": {
        "guidance": (
            "VOICE: a sharp, experienced coach talking to a peer. This person is already "
            "comfortable speaking and wants to raise their delivery. Be concise and direct. "
            "Use plain professional language (emphasis, inflection, cadence, "
            "signposting, landing a point). Avoid inflated phrases such as 'acoustic "
            "baseline' or 'vocal mechanics'. Give TWO different refinements, not one. "
            "State suggestions firmly, without cushioning or reassurance. Never use words "
            "like 'doable', 'it's okay' or 'no pressure'."
        ),
        "length": "260 to 330 words in total",
        "sentences": "four or five",
        "refinements": "two different concrete ways",
        "stretch": "Vary your pace on purpose: slow down on your key sentence and move slightly faster through the supporting detail.",
    },
}

# A random starting angle makes responses feel different from case to case.
OPENING_HINTS = [
    "Start by quoting a short phrase they actually said and what it did for a listener.",
    "Start with how the overall delivery would feel to someone in the audience.",
    "Start with the single strongest moment of the recording.",
    "Start with something specific about how they began the speech.",
    "Start with a concrete observation about their rhythm or flow.",
]

PROMPT = """You are an experienced public-speaking coach giving feedback on one practice recording.

{audience_guidance}

Transcript (automatic transcription, so it may contain small mistakes. Never point
out odd words or spelling, and only quote phrases that read naturally):
{transcript}

Observations about their delivery (background for you only, never repeat numbers):
{observations}

{focus_line}

Write the feedback as three short paragraphs, each beginning with a bold label:

**What came across well**: two or three specific strengths. Quote short phrases
(under 10 words) exactly as they appear in the transcript, in quotation marks,
and explain what each did for a listener. For example, a warm greeting can make
an audience feel welcome.

{next_paragraph}

**Try this next time**: one concrete exercise. {exercise}
Rephrase it naturally, explain how to do it in a sentence or two, and say what a
listener will notice when they do.

Length: {length}. Every paragraph needs {sentences} full sentences. If the
transcript is short, spend the extra words on how the delivery sounded (rhythm,
pauses, volume, expression) rather than stopping early.

Style rules:
- {opening_hint}
- Keep the tone sincere and specific, never gushing.
- Be specific about what could be even better, but present every suggestion as
  something to add or try. Never say that anything they said was bad, weak,
  unclear, or wrong. Never quote their words as the "before" in a comparison,
  and never write "instead of". Offer the alternative wording on its own.
- Do not write sentences that only announce your suggestions (for example "Two
  refinements will sharpen your points"). Go straight to the suggestion.
- Your suggestions and exercise must agree with the observations. If the pace was
  slow, never advise slowing down or adding pauses, and if it was fast, never advise
  speeding up.
- The observations above are background for you. Do not repeat them as criticism or
  quote a phrase of theirs as the thing to fix.
- Only quote words that appear in the transcript, copied exactly. Never quote a
  phrase that looks garbled or ungrammatical (it is probably a transcription error);
  describe that moment instead. If you are not sure of the exact words, paraphrase.
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

NEXT_WORDING = """**What to try next**: explain the focus area in plain language and why it matters
to a listener. Then offer {refinements} to bring out a point they already made.
Take an idea from their speech and show how it could land even more strongly, for
example: One way to bring out that idea: "<a stronger version of the point>". After each
one, name the single word in it worth emphasizing when they say it, and why. These are
extra options for them, not corrections. If the transcript is too short for that,
use a realistic example instead."""

NEXT_DELIVERY = """**What to try next**: explain the focus area in plain language and why it matters
to a listener. Then give {refinements} to apply it to a moment from their own speech.
Each time, quote a phrase they actually said, pick ONE specific word in it to
emphasize, and say how: a touch slower, a little louder, a lift or drop in pitch, or a
short pause just before it. Then say what that does for the listener. For example:
Try emphasizing "<key word>" in "<their phrase>" by saying it a little slower, because
<what the listener will feel or understand>. Suggest ways of saying their words, not
different words. These are extra options for them, not corrections."""

NEXT_ALL_CLEAR = """**What to try next**: say in one plain sentence that nothing stands out as needing
to change. Do not invent a weakness or a focus area. Then write at least three more
sentences: explain why a small stretch challenge is worth trying once the basics are
solid, and show exactly where in their speech they could try it. Quote a phrase of
theirs, name one specific word in it to emphasize, and say how (slower, louder, a lift
in pitch, or a short pause before it) and what that does for the listener."""


def get_feedback(transcript, obs, focus, user_chose_focus, audience="nervous", metrics=None):
    """
    transcript: the text of what they said
    obs: list from coaching.observations()
    focus: 'pace' / 'pauses' / 'expression' / 'volume' / 'wording' / None
    user_chose_focus: True if the person picked this goal themselves
    audience: 'nervous' or 'confident'
    metrics: from coaching.compute_metrics(), used to pick an exercise that fits
    """

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found. Check that trial.env "
            "is in the parent HackUMass folder."
        )

    profile = AUDIENCES.get(audience, AUDIENCES["nervous"])

    # Only the plain-language meaning is passed on, not the raw numbers,
    # so the feedback stays in words.
    obs_text = "\n".join(f"- {title}: {msg}" for _key, title, _value, _ok, msg in obs)

    if focus is None:
        focus_line = "Everything looks solid, so there is no weak spot to fix."
        next_paragraph = NEXT_ALL_CLEAR
        exercise = profile["stretch"]
    else:
        who = "The person chose to work on" if user_chose_focus else "Focus area (picked automatically)"
        focus_line = f"{who}: {focus}."
        template = NEXT_WORDING if focus == "wording" else NEXT_DELIVERY
        next_paragraph = template.format(refinements=profile["refinements"])
        exercise = exercise_for(focus, metrics)

    prompt = PROMPT.format(
        audience_guidance=profile["guidance"],
        transcript=transcript,
        observations=obs_text,
        focus_line=focus_line,
        next_paragraph=next_paragraph,
        exercise=exercise,
        length=profile["length"],
        sentences=profile["sentences"],
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
