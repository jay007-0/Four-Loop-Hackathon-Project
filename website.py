
import os
import tempfile

import pandas as pd
import streamlit as st

from textfiles import describe
from audio_analysis import extract_audio_features
from coaching import AUDIENCES, GOALS, compute_metrics, observations, pick_focus
from ai_feedback import get_feedback


def save_temp_audio(audio):
    """Save recorded/uploaded audio to a temporary file and return its path."""
    extension = os.path.splitext(audio.name)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
        f.write(audio.getvalue())
        return f.name


def delta_text(current, previous, unit=""):
    """Change since the previous attempt, or None if there is nothing to compare."""
    if previous is None or current is None:
        return None
    diff = current - previous
    if abs(diff) < 0.05:
        return "same as last time"
    return f"{diff:+.1f}{unit} vs last time"


def show_stats(stats, previous):
    """Show one row of metric cards. stats: (label, value, key, delta_unit, note)."""
    cols = st.columns(len(stats))
    for col, (label, value, key, unit, note) in zip(cols, stats):
        with col:
            prev_value = previous.get(key) if previous else None
            current = value if isinstance(value, (int, float)) else None
            shown = "n/a" if value is None else (f"{value}{unit}" if unit else f"{value}")
            st.metric(
                label,
                shown,
                delta=delta_text(current, prev_value, unit),
                delta_color="off",
            )
            st.caption(note)


st.set_page_config(
    page_title="Speech Clarity Coach",
    page_icon="🎙️",
    layout="wide"
)


# Memory that survives Streamlit reruns
if "history" not in st.session_state:
    st.session_state.history = []     # metrics from each attempt this session
    st.session_state.result = None    # the latest full result


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎯 Your practice")

    audience_label = st.selectbox(
        "I'm practicing as someone who...",
        list(AUDIENCES.keys()),
        help="This only changes the tone of the feedback.",
    )
    audience = AUDIENCES[audience_label]

    goal_label = st.selectbox(
        "What would you like to work on?",
        list(GOALS.keys()),
        help="You decide what to practice. Pick 'Let the website choose' if you're not sure.",
    )
    chosen_key = GOALS[goal_label]

    st.divider()
    st.caption(
        "💬 These numbers are gentle guides, not scores. This is a practice "
        "tool, not a medical or clinical one."
    )
    st.caption(
        "🔒 Your audio is processed on this computer. Only the text of what "
        "you said and a short summary of the measurements are sent to the AI "
        "for feedback."
    )
    if st.session_state.history and st.button("Clear my practice history"):
        st.session_state.history = []
        st.session_state.result = None
        st.rerun()


st.title("🎙️ Speech Clarity Coach")
st.write(
    "Practice speaking in front of people. Record or upload a short clip, "
    "get specific feedback on what worked and what to try, then go again "
    "and see how you improve."
)


# ---------- 1. Recording / upload ----------
st.header("1. Add your audio")

tab_record, tab_upload = st.tabs(["🎤 Record", "📁 Upload"])

with tab_record:
    recorded = st.audio_input("Record your speech")

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an audio file",
        type=["wav", "mp3", "m4a", "ogg"]
    )

audio = recorded or uploaded

if audio is None:
    st.info("Record or upload audio to get started.")
    st.stop()

st.audio(audio)


# ---------- 2. Workflow ----------
st.header("2. Analyze")

if st.button("Analyze my speech", type="primary"):

    path = save_temp_audio(audio)

    try:
        with st.spinner("Converting speech to text..."):
            transcript = describe(path)

        if not transcript or not transcript.strip():
            st.warning(
                "No speech was detected. Try recording again "
                "and speak clearly."
            )
            st.session_state.result = None

        else:
            with st.spinner("Measuring your pace, pauses and voice..."):
                features = extract_audio_features(path)
                metrics = compute_metrics(features, transcript)
                obs = observations(metrics)
                focus = pick_focus(obs, chosen_key)

            with st.spinner("Writing your feedback..."):
                feedback = get_feedback(
                    transcript,
                    obs,
                    focus,
                    user_chose_focus=chosen_key is not None,
                    audience=audience,
                )

            st.session_state.history.append(metrics)
            st.session_state.result = {
                "transcript": transcript,
                "metrics": metrics,
                "obs": obs,
                "focus": focus,
                "feedback": feedback,
            }

    except Exception as e:
        st.error(f"Something went wrong: {e}")

    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------- 3. Results ----------
result = st.session_state.result

if result:

    st.header("3. Your results")

    m = result["metrics"]
    history = st.session_state.history
    previous = history[-2] if len(history) >= 2 else None

    # ----- Feedback -----
    st.subheader("💡 Your feedback")
    with st.container(border=True):
        st.markdown(result["feedback"])

    # ----- Numbers -----
    st.subheader("📊 Your numbers")
    st.caption(
        "The notes under each number are rough guides, not targets. "
        + ("Arrows show the change since your last attempt." if previous
           else "Record again to see how each number changes.")
    )

    with st.container(border=True):
        st.markdown("**⏱️ Pace and timing**")
        show_stats([
            ("Pace", m["wpm"], "wpm", " wpm", "Words per minute overall. Most listeners find 120-160 comfortable."),
            ("Pace while speaking", m["speaking_wpm"], "speaking_wpm", " wpm", "Speed during the moments you were actually talking, pauses excluded."),
            ("Length", m["duration_s"], "duration_s", " s", "Total recording length."),
            ("Words", m["words"], "words", "", "Words in the transcript."),
        ], previous)

    with st.container(border=True):
        st.markdown("**⏸️ Pauses**")
        show_stats([
            ("Pauses", m["pause_count"], "pause_count", "", "Breaks of at least 0.4 s between stretches of speech."),
            ("Average pause", m["avg_pause_s"], "avg_pause_s", " s", "About 0.5-1 s feels natural to most listeners."),
            ("Longest pause", m["longest_pause_s"], "longest_pause_s", " s", "Your single longest break."),
            ("Time pausing", m["pause_pct"], "pause_pct", "%", "Share of the recording spent in pauses."),
        ], previous)

    with st.container(border=True):
        st.markdown("**🎵 Your voice**")
        show_stats([
            ("Pitch variety", m["pitch_var"], "pitch_var", " st", "How much your pitch moves, in semitones (st). Higher sounds more expressive."),
            ("Typical pitch", m["pitch_median"], "pitch_median", " Hz", "Your middle pitch. This describes your voice, so there is no better or worse."),
            ("Volume change", m["fade_db"], "fade_db", " dB", "Loudness at the end compared with the start. Negative means quieter."),
        ], previous)

    with st.container(border=True):
        st.markdown("**🗣️ Sentences and wording**")
        show_stats([
            ("Sentences", m["sentence_count"], "sentence_count", "", "Sentences in the transcript."),
            ("Average sentence", m["avg_sentence_words"], "avg_sentence_words", " words", "Shorter sentences are easier to follow when spoken."),
            ("Longest sentence", m["longest_sentence_words"], "longest_sentence_words", " words", "Your longest single sentence."),
            ("Softening phrases", m["softener_total"], "softener_total", "", "Phrases like \"I think\" or \"maybe\" that can soften a point."),
        ], previous)
        if m["softeners"]:
            found = ", ".join(f'"{p}" x{n}' for p, n in m["softeners"].items())
            st.caption(f"Found: {found}")

    # What the numbers mean, in words
    st.markdown("**What this means**")
    for key, title, value, ok, msg in result["obs"]:
        icon = "✅" if ok else "💡"
        st.write(f"{icon} **{title}:** {msg}")

    # ----- Progress -----
    if len(history) >= 2:
        st.subheader("📈 Your progress this session")
        df = pd.DataFrame(
            {
                "Pace (words/min)": [h["wpm"] for h in history],
                "Average pause (s)": [h["avg_pause_s"] for h in history],
                "Softening phrases": [h["softener_total"] for h in history],
            },
            index=pd.Index(range(1, len(history) + 1), name="Attempt"),
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.line_chart(df[["Pace (words/min)"]])
        with c2:
            st.line_chart(df[["Average pause (s)"]])
        with c3:
            st.line_chart(df[["Softening phrases"]])
        st.caption("Compare yourself to your own earlier attempts, not to anyone else.")

    with st.expander("📝 What we heard (transcript)"):
        st.write(result["transcript"])
        st.caption(
            f"{m['words']} words. Automatic transcription can make mistakes, "
            "so some of the feedback may be based on a slightly different "
            "wording than what you actually said."
        )

    report = (
        f"TRANSCRIPT\n\n{result['transcript']}\n\n"
        f"FEEDBACK\n\n{result['feedback']}\n"
    )
    st.download_button(
        "⬇️ Download results",
        report,
        file_name="speech_feedback.txt"
    )
