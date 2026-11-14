# Jayden Njuguna
# Uses whisper.cpp library, converts speech to text, outputs str
# audio file is .wav

import subprocess

from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "ggml-base.en.bin"

def transcribe_audio(audio_path)->str:
    result = subprocess.run(
        [
            "whisper-cli",
            "-m", MODEL_PATH,
            "-f", audio_path,
            "-nt"
        ],
        capture_output=True,
        text=True,
    )

    # print("Return code:", result.returncode)
    # print("Whisper STDERR:", result.stderr)
    # print("Whisper STDOUT:", result.stdout)

    return result.stdout.strip() # outputs string
print(transcribe_audio("sample.wav"))