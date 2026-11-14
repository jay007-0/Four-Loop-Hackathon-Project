# Four-Loop-Hackathon-Project
# Speech-to-Text Setup

This component uses Whisper.cpp to transcribe audio recordings into text.

## Requirements
- Python 3
- Whisper.cpp
- English Whisper base model

## 1. Install Whisper.cpp

On macOS, install Homebrew if necessary, then run:

```bash
brew install whisper-cpp
```

## 2. Download the Whisper model

From the project directory, run:

```bash
mkdir -p models

curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin -o models/ggml-base.en.bin
```

## 3. Run the transcription

```bash
python3 speech_to_text.py
```

The model is downloaded separately and is not included in this repository.
