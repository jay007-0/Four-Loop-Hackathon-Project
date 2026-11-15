# Speech Clarity Coach

A browser-based speech practice tool that records or uploads a short audio clip, transcribes it locally, and gives practical coaching feedback on clarity, pacing, and delivery.

This project was built as a hackathon prototype for the Four Loop team. It is designed to help people practice speaking in front of an audience without turning the experience into a performance scorecard.

## What it does

- Record speech directly in the browser or upload a supported audio file
- Convert audio to a Whisper-friendly format and transcribe it locally
- Measure rough speaking metrics such as pace, pauses, and sentence structure
- Generate AI feedback with a Gemini-powered coaching prompt
- Display the transcript, suggestions, and a downloadable report in the app

## Demo flow

1. Open the Streamlit app in a browser.
2. Record a short message or upload an audio file.
3. Click Analyze my speech.
4. Review the transcript and coaching feedback.
5. Download the results as a text report.

## Tech stack

- Python
- Streamlit
- FFmpeg
- whisper.cpp
- Google GenAI / Gemini
- Librosa and NumPy

## Project structure

- `website.py` — main Streamlit application UI and user workflow
- `textfiles.py` — FFmpeg conversion and local Whisper transcription
- `ai_feedback.py` — Gemini-based coaching feedback generation
- `coaching.py` — metrics, observations, and coaching logic
- `audio_analysis.py` — standalone audio feature extraction utility
- `speech_to_text.py` — separate transcription helper script
- `geminitest.py` — experimental Gemini/test script

## Prerequisites

Before running the app, install or prepare the following:

- Python 3.10+
- FFmpeg available on your PATH as `ffmpeg`
- whisper.cpp installed with `whisper-cli` on your PATH
- A Whisper English model at `models/ggml-base.en.bin`
- A Google Gemini API key stored in an environment file or exported in your shell

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/jay007-0/Four-Loop-Hackathon-Project.git
cd Four-Loop-Hackathon-Project
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit python-dotenv google-genai librosa numpy
```

### 4. Install system tools

Install FFmpeg and whisper.cpp for your OS. Make sure both commands work:

```bash
ffmpeg --version
whisper-cli --help
```

Place the English Whisper model in the project directory:

```text
Four-Loop-Hackathon-Project/
└── models/
    └── ggml-base.en.bin
```

The model is not downloaded automatically by this project.

### 5. Configure the Gemini API key

The app expects `GEMINI_API_KEY` to be available. One common setup is to place a file called `trial.env` in the parent directory of the repository:

```text
parent-directory/
├── trial.env
└── Four-Loop-Hackathon-Project/
```

`trial.env` should contain:

```dotenv
GEMINI_API_KEY=your_gemini_api_key_here
```

You can also export the key in your shell before launching the app:

```bash
export GEMINI_API_KEY="your_gemini_api_key_here"
```

Never commit a real API key to the repository.

### 6. Launch the app

From the repository root:

```bash
streamlit run website.py
```

Streamlit will print a local URL such as `http://localhost:8501`.

## How the app works

```text
Browser recording or upload
          │
          ▼
  Streamlit interface (website.py)
          │
          ▼
textfiles.describe(audio)
          │
          ├─ FFmpeg converts file to 16 kHz mono WAV
          │
          ▼
  whisper-cli + models/ggml-base.en.bin
          │
          ▼
        AI coaching feedback
          │
          ▼
Display transcript, suggestions, and report
```

## Notes on privacy and responsible use

- Audio is processed on the local machine during transcription.
- The transcript and a short summary of measurements are sent to the Gemini API for coaching feedback.
- Do not send sensitive or identifying audio to the model.
- This project is a communication practice tool, not a medical or clinical assessment tool.
- Transcription quality can vary with background noise, accents, and overlapping speakers.

## Current limitations

This is a hackathon prototype. The repository does not currently include:

- automated tests
- a dependency lockfile
- a deployment configuration
- automatic model download support

Future work could include:

- adding a `requirements.txt` or `pyproject.toml`
- improving environment configuration and setup UX
- adding validation for missing tools and model files
- expanding the audio analysis experience in a clear, evidence-based way

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `ffmpeg` not found | Install FFmpeg and ensure it is on your PATH. |
| `whisper-cli` not found | Install whisper.cpp and ensure the executable is on your PATH. |
| Whisper model missing | Confirm `models/ggml-base.en.bin` exists in the repository root. |
| `GEMINI_API_KEY` not found | Check your shell environment or the parent `trial.env` file. |
| No speech detected | Try a clearer recording with less background noise and speak directly into the mic. |
| Gemini returns an error | Check the API key, network access, and provider availability. |
| Upload fails | Use a supported format such as WAV, MP3, M4A, or OGG. |

## License

No license file is currently included in this repository. Contact the project maintainers before reusing or redistributing the code.

## Contributors

Built by **Team Four-Loop** for a hackathon project.
