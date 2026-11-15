
"""
Turns raw audio measurements and the transcript into plain-language
observations and picks one thing to focus on. All thresholds are soft
guidelines, not rules. This is a practice tool, not a medical or clinical one.
"""

import re

GOALS = {
    "Let the app choose": None,
    "Pace (speaking speed)": "pace",
    "Pauses": "pauses",
    "Expression (vocal variety)": "expression",
    "Volume (staying strong to the end)": "volume",
    "Confident wording": "wording",
}

AUDIENCES = {
    "I get nervous speaking in front of people": "nervous",
    "I'm comfortable and want to sharpen my delivery": "confident",
}

EXERCISES = {
    "pace": "Take a calm breath at the end of each sentence before starting the next one.",
    "pauses": "Aim for pauses of about half a second to one second. Pause briefly before an important point.",
    "expression": "Pick one key word in each sentence and say it with a little more energy.",
    "volume": "Finish each sentence as strongly as you started it, especially the last word.",
    "wording": "Try opening with your main point as one clear, direct sentence, then add the context around it.",
}

# Pace is judged on speaking speed (pauses excluded), so deliberate pauses
# don't make a speaker look slow.
FAST_WPM = 185
SLOW_WPM = 110


def exercise_for(focus, m=None):
    """An exercise that matches the direction of the problem (e.g. slow vs fast)."""
    if m is None:
        return EXERCISES[focus]

    if focus == "pace":
        speed = m["speaking_wpm"]
        if speed > FAST_WPM:
            return "Take a calm breath at the end of each sentence before starting the next one."
        if speed < SLOW_WPM:
            return "Pick one sentence in each part of your talk and say it a little quicker, letting the words connect the way they do in relaxed conversation."
        return "Slow down slightly on your most important sentence, then move a little faster through the supporting detail."

    if focus == "pauses":
        if m["pause_count"] == 0:
            return "Add a short pause, about half a second, after each main idea."
        if m["avg_pause_s"] > 1.2:
            return "Shorten your pauses to about a second: take your breath quickly, then start the next thought right after it."
        return "Place one deliberate pause right before your most important point."

    return EXERCISES[focus]


# Phrases that can make a point sound less certain. A rough guide only:
# "just" or "maybe" are often perfectly fine in context.
SOFTENERS = [
    "i think", "i guess", "i feel like", "maybe", "kind of", "sort of",
    "just", "a little bit", "probably", "hopefully", "sorry", "i'm not sure",
]


def find_softeners(transcript):
    """Returns {phrase: count} for softening phrases found in the transcript."""
    text = transcript.lower()
    found = {}
    for phrase in SOFTENERS:
        n = len(re.findall(r"\b" + re.escape(phrase) + r"\b", text))
        if n:
            found[phrase] = n
    return found


def compute_metrics(features, transcript):
    """Combine audio features with the transcript into the numbers we show."""
    duration = features["duration_seconds"]
    words = len(transcript.split())

    sentences = [s.strip() for s in re.split(r"[.!?]+", transcript) if s.strip()]
    sentence_lengths = [len(s.split()) for s in sentences] or [words]

    pause_total = features["pauses"]["total_seconds"]
    speaking_time = max(duration - pause_total, 1)

    softeners = find_softeners(transcript)

    return {
        "words": words,
        "duration_s": duration,
        "wpm": round(words / (duration / 60)) if duration > 0 else 0,
        "speaking_wpm": round(words / (speaking_time / 60)),
        "pause_count": features["pauses"]["count"],
        "avg_pause_s": features["pauses"]["average_seconds"],
        "longest_pause_s": features["pauses"]["longest_seconds"],
        "pause_total_s": pause_total,
        "pause_pct": round(features["pauses"]["percentage_of_recording"]),
        "pitch_var": features["pitch_hz"]["variation_semitones"],
        "pitch_median": features["pitch_hz"]["median"],
        "fade_db": features["volume_fade_db"],
        "sentence_count": len(sentences),
        "avg_sentence_words": round(sum(sentence_lengths) / len(sentence_lengths), 1),
        "longest_sentence_words": max(sentence_lengths),
        "softener_total": sum(softeners.values()),
        "softeners": softeners,
    }


def observations(m):
    """
    One entry per area: (key, title, value, ok, message).
    'ok' only controls a friendly icon; it is never a score.
    """
    out = []

    # --- Pace (judged on speaking speed, pauses excluded) ---
    speed = m["speaking_wpm"]
    if speed > FAST_WPM:
        ok, msg = False, "On the faster side. A slightly slower pace gives listeners time to absorb each point."
    elif speed < SLOW_WPM:
        ok, msg = False, "On the slower side while speaking. That can feel deliberate, but a bit more flow may keep listeners with you."
    else:
        ok, msg = True, "In a comfortable range for most listeners."
    out.append(("pace", "Pace", f"{speed} words/min while speaking", ok, msg))

    # --- Pauses ---
    if m["pause_count"] == 0 and m["duration_s"] > 15:
        ok, msg = False, "Very few pauses. Short pauses help listeners keep up and give you time to breathe."
        value = "0 pauses"
    elif m["avg_pause_s"] > 1.2:
        ok, msg = False, "Pauses were fairly long. Shorter ones (about half a second to a second) keep the flow going."
        value = f"{m['pause_count']} pauses, avg {m['avg_pause_s']} s"
    else:
        ok, msg = True, "Pauses were a natural length."
        value = f"{m['pause_count']} pauses, avg {m['avg_pause_s']} s"
    out.append(("pauses", "Pauses", value, ok, msg))

    # --- Expression ---
    pv = m["pitch_var"]
    if pv is None:
        ok, msg, value = True, "Not enough voiced speech to measure.", "n/a"
    elif pv < 2:
        ok, msg, value = False, "Pitch stayed quite steady. Emphasising key words adds variety.", f"{pv} semitones"
    else:
        ok, msg, value = True, "The voice had good variety.", f"{pv} semitones"
    out.append(("expression", "Expression", value, ok, msg))

    # --- Volume ---
    fade = m["fade_db"]
    if fade < -4:
        ok, msg = False, "The voice got quieter toward the end."
    else:
        ok, msg = True, "Volume stayed steady."
    out.append(("volume", "Volume", f"{fade:+.1f} dB start to end", ok, msg))

    # --- Wording (softening phrases) ---
    total = m["softener_total"]
    per_100 = total / m["words"] * 100 if m["words"] else 0
    top = sorted(m["softeners"], key=m["softeners"].get, reverse=True)[:3]
    quoted = ", ".join(f'"{p}"' for p in top)
    if per_100 > 3:
        ok = False
        msg = f"Softening phrases appeared often ({quoted}). They can make a point sound less certain."
    elif total:
        ok = True
        msg = f"A few softening phrases ({quoted}), which is perfectly normal."
    else:
        ok, msg = True, "The wording sounded direct, with no softening phrases."
    out.append(("wording", "Wording", f"{total} softening phrases", ok, msg))

    return out


def pick_focus(obs, chosen_key=None):
    """
    The user's own choice always wins. Otherwise pick the first area that could
    use attention, or None if everything looks good.
    """
    if chosen_key:
        return chosen_key
    for key, _title, _value, ok, _msg in obs:
        if not ok:
            return key
    return None
