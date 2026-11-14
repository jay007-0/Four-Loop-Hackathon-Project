import os
import tempfile

import streamlit as st

from textfiles import describe
from ai_feedback import get_recommendations

def save_temp_audio(audio):
    """Save recorded/uploaded audio to a temporary file and return its path."""
    extension = os.path.splitext(audio.name)[1] or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as f:
        f.write(audio.getvalue())
        return f.name


st.set_page_config(
    page_title="Speech Clarity Coach",
    page_icon="🎙️",
    layout="wide"
)


# Memory that survives Streamlit reruns
if "transcript" not in st.session_state:
    st.session_state.transcript = None
    st.session_state.recommendations = None


st.title("🎙️ Speech Clarity Coach")
st.write(
    "Record or upload your speech. We'll turn it into text "
    "and suggest how to make it clearer."
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
            st.session_state.transcript = None
            st.session_state.recommendations = None

        else:
            with st.spinner("Getting recommendations..."):
                recommendations = get_recommendations(transcript)

            st.session_state.transcript = transcript
            st.session_state.recommendations = recommendations

    except Exception as e:
        st.error(f"Something went wrong: {e}")

    finally:
        if os.path.exists(path):
            os.remove(path)


# ---------- 3. Results ----------
if st.session_state.transcript:

    st.header("3. Results")

    transcript = st.session_state.transcript
    recommendations = st.session_state.recommendations

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Transcript")
        st.info(transcript)
        st.caption(f"{len(transcript.split())} words")

    with col2:
        st.subheader("💡 How to make it clearer")
        st.markdown(recommendations)

    report = (
        f"TRANSCRIPT\n\n{transcript}\n\n"
        f"RECOMMENDATIONS\n\n{recommendations}"
    )

    st.download_button(
        "⬇️ Download results",
        report,
        file_name="speech_feedback.txt"
    )
