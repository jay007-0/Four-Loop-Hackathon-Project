# Speech Clarity Coach

> **Four-Loop Hackathon Project** — a browser-based prototype that turns a short speech recording into a transcript and practical, AI-generated suggestions for making the message clearer.

## At a glance

Speech Clarity Coach helps people review what they said and find ways to express the same message more clearly. Record directly in the app or upload an audio file. The app transcribes the recording locally with `whisper.cpp`, asks Google Gemini for concise coaching suggestions and a clearer rewrite, then presents both results side by side for download.

| | |
| --- | --- |
| **Interface** | Streamlit web app |
| **Speech recognition** | Local `whisper.cpp` command-line tool and `ggml-base.en.bin` model |
| **Writing feedback** | Google Gemini via the Google GenAI Python SDK |
| **Primary flow** | Record/upload → transcribe → get feedback → review/download |
| **Status** | Hackathon prototype; run locally using the setup instructions below |

## Why we built it

People often know what they want to say but would benefit from a quick, low-friction way to review how clearly they communicated it. Speech Clarity Coach makes that feedback loop concrete: users can speak naturally, see what the speech recognizer understood, and get actionable recommendations alongside a revised version of their message.

The app is intended as a supportive writing and communication aid—not as a judgment of a person's voice, accent, speaking ability, or worth. Users should treat generated feedback as suggestions and keep the final say over their own words.

## What the prototype does

- **Record or upload speech.** Use the browser recording control or upload a `wav`, `mp3`, `m4a`, or `ogg` file.
- **Transcribe the audio.** The Streamlit app uses the local `whisper.cpp` CLI and English base model through `textfiles.py`; uploaded audio is converted to 16 kHz, mono WAV with FFmpeg first.
- **Generate writing-focused coaching.** The transcript is sent to Gemini with instructions to return 3–5 short, practical recommendations and a clearer version in the transcript's language.
- **Review both outputs.** See the transcript and recommendations together, including a word count for the transcript.
- **Download a report.** Save the transcript and generated recommendations as `speech_feedback.txt`.
- **Handle common errors in the interface.** The app warns when no speech is detected and displays errors from the analysis workflow.

## Demo walkthrough for judges

1. Start the app using the instructions in [Running the app](#running-the-app).
2. Choose **Record** and record a short, natural message, or choose **Upload** and select a supported audio file.
3. Listen to the selected recording in the page.
4. Click **Analyze my speech** and wait for transcription and feedback generation to finish.
5. Review the transcript and the 3–5 recommendations plus clearer rewrite.
6. Click **Download results** to save the text report.

For a useful demo, try a short message with a clear idea but a little unnecessary repetition or an unclear sentence. The goal is to show how the transcript and rewrite can help a speaker reflect and revise—not to present the model's rewrite as the only correct answer.

## How it works

```text
Browser recording or audio upload
                 │
                 ▼
       Streamlit interface (website.py)
                 │
                 ▼
 textfiles.describe(audio) ── FFmpeg converts to 16 kHz mono WAV
                 │
                 ▼
 whisper-cli + models/ggml-base.en.bin (local transcription)
                 │
                 ▼
 ai_feedback.get_recommendations(transcript)
                 │
                 ▼
 Gemini returns recommendations and a clearer version
                 │
                 ▼
     Display transcript and feedback; download report
```

The web application keeps the transcript and recommendations in Streamlit session state so they remain visible across the reruns caused by interactions. The temporary audio file created for analysis is removed in a `finally` block after the analysis attempt.

### Important implementation distinction

`website.py` is the primary application entry point. Its active transcription path is `textfiles.py`, which handles audio conversion and invokes `whisper-cli`. `speech_to_text.py` is an additional simpler transcription helper and is not the path used by the Streamlit app. `audio_analysis.py` contains a separate Librosa-based numerical feature extractor (duration, RMS energy, spectral centroid, zero-crossing rate, estimated pitch, and quiet-section estimate); it is not currently connected to the web app's user-facing workflow.

## Technology

- **Python** — application and analysis code.
- **Streamlit** — browser-based UI and audio input/upload controls.
- **whisper.cpp** — local speech-to-text inference through `whisper-cli`.
- **FFmpeg** — audio normalization for the transcription pipeline.
- **Google GenAI Python SDK / Gemini** — language-model feedback and rewrite.
- **Librosa and NumPy** — used by the standalone audio feature-analysis module.
- **python-dotenv** — loads the Gemini API key from an environment file.

## Requirements

Install or prepare the following before running the app:

1. **Python 3.10 or later** is recommended.
2. **FFmpeg** must be installed and available on your `PATH` as `ffmpeg`.
3. **whisper.cpp** must be installed, and its `whisper-cli` executable must be available on your `PATH`.
4. The English base Whisper model file must exist at `models/ggml-base.en.bin` in the project directory. Obtain the model using the official whisper.cpp model instructions; the binary model is not included in this repository.
5. A **Google Gemini API key** is required for generating recommendations. Transcription can run locally, but the feedback step needs the external Gemini API.
6. Python packages used by the checked-in application and analysis modules:

```bash
python -m pip install streamlit python-dotenv google-genai librosa numpy
```

If you use a virtual environment, activate it before installing packages and starting Streamlit.

## Setup

### 1. Get the code

```bash
git clone https://github.com/jay007-0/Four-Loop-Hackathon-Project.git
cd Four-Loop-Hackathon-Project
```

This repository may be private; you need access to it to clone it.

### 2. Install and verify system tools

Install FFmpeg and whisper.cpp using the instructions for your operating system. Verify that both commands are discoverable:

```bash
ffmpeg -version
whisper-cli --help
```

Place the Whisper model at the exact path expected by the code:

```text
Four-Loop-Hackathon-Project/
└── models/
    └── ggml-base.en.bin
```

The application does not download the model automatically.

### 3. Install Python dependencies

```bash
python -m pip install streamlit python-dotenv google-genai librosa numpy
```

### 4. Configure the Gemini API key

The current `ai_feedback.py` looks for `GEMINI_API_KEY` in `../trial.env` relative to the project directory. For the standard clone layout above, that means placing the file next to the repository directory, not inside it:

```text
parent-directory/
├── trial.env
└── Four-Loop-Hackathon-Project/
```

The file should contain:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

Alternatively, export `GEMINI_API_KEY` in the shell before launching the app. **Never commit a real API key or put it in a public screenshot or demo recording.** The current code expects the key to be provided; it does not include one.

### 5. Launch the app

From the repository root:

```bash
streamlit run website.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Repository map

| File | Purpose |
| --- | --- |
| `website.py` | Main Streamlit app: audio capture/upload, analysis button, result display, and report download. |
| `textfiles.py` | Converts input audio with FFmpeg, runs `whisper-cli`, and cleans the transcript output. |
| `ai_feedback.py` | Loads the Gemini key, constructs the speaking-coach prompt, and returns recommendations and a clearer version. |
| `audio_analysis.py` | Standalone Librosa audio-feature extraction; not currently called by the web app. |
| `speech_to_text.py` | Simpler standalone `whisper-cli` transcription helper; not used by the main web workflow. |
| `geminitest.py` | Experimental script that exercises transcription and Gemini intent extraction; not the Streamlit app entry point. |

## Privacy and responsible use

- **Audio processing:** The main app passes the audio file to local FFmpeg and `whisper-cli` for transcription. Audio is written to a temporary file during the analysis and the app attempts to remove that file when processing ends.
- **External AI request:** The transcript—not just local audio features—is passed to the configured Gemini API to generate feedback. Do not use sensitive, confidential, or identifying speech unless you have reviewed the applicable provider terms and have permission to share it.
- **API key:** Keep `GEMINI_API_KEY` private. Do not commit credentials to the repository.
- **Model limitations:** Transcription can be inaccurate, especially with noise, overlapping speakers, accents, or unsupported languages. The configured Whisper model is the English base model. Gemini feedback can be generic, mistaken, or fail to preserve nuance. Review the transcript and suggestions before relying on them.
- **Not an assessment tool:** The project is a prototype for communication practice. It does not diagnose speech or language conditions, evaluate a speaker's identity, or provide professional advice.

## Current limitations and future work

This is a hackathon prototype. The checked-in implementation does not currently include automated tests, a dependency lockfile, a deployment configuration, or a built-in model downloader. The README intentionally describes the present code rather than implying those capabilities already exist.

Possible next steps include:

- Add a `requirements.txt` or `pyproject.toml` and documented, reproducible environment setup.
- Add automated tests for audio conversion, empty transcripts, API failures, and report generation.
- Improve the configuration experience so the API key can be supplied through standard environment variables or a project-local `.env` file.
- Add clearer validation for missing executables, unsupported audio, and model files.
- Connect the existing audio-feature analysis to the interface only if it can be presented with useful context and without overclaiming what the measurements mean.
- Explore additional language/model support and evaluate transcription quality across varied recordings.
- Add explicit retention controls and review data-handling expectations before any hosted deployment.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `whisper-cli` was not found | Install whisper.cpp and ensure its executable is on `PATH`; open a new terminal after changing `PATH`. |
| `ffmpeg` was not found | Install FFmpeg and ensure the `ffmpeg` executable is on `PATH`. |
| Whisper model not found | Confirm `models/ggml-base.en.bin` exists relative to the repository root. |
| `GEMINI_API_KEY` not found | Check the key is exported in the shell or saved in `../trial.env` relative to this project. Restart Streamlit after environment changes. |
| No speech was detected | Try a clearer, longer recording with less background noise, and confirm the audio plays in the app. |
| Gemini returned no feedback or an API error | Check the key, API access, network connectivity, and provider availability. The transcription step and the feedback step are separate. |
| Upload or conversion fails | Try one of the listed formats (`wav`, `mp3`, `m4a`, `ogg`) and confirm FFmpeg can read the file. |

## Contributors

Built by **Four Loop** for a hackathon. The repository currently credits Jayden Njuguna in `speech_to_text.py`; add or update the full team roster and individual roles here before submitting if the judging rubric expects contributor attribution.

## License

No license file is currently included in the repository. Contact the project maintainers before reusing or redistributing the code.
