
import subprocess
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "models" / "ggml-base.en.bin"

# If these aren't on PATH, put the full path here,
# e.g. r"C:\whisper.cpp\build\bin\Release\whisper-cli.exe"
WHISPER_CLI = "whisper-cli"
FFMPEG = "ffmpeg"


def run(command, name):
    """Run a command-line program and return its output, or raise a clear error."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"'{command[0]}' was not found. Is {name} installed and on PATH?"
        )

    if result.returncode != 0:
        raise RuntimeError(f"{name} failed:\n{result.stderr[-500:]}")

    return result.stdout


def describe(audio_path):
    """Takes the path to an audio file. Returns the transcript as a string."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Whisper model not found at {MODEL_PATH}")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio_16k.wav"

        # 1. Convert any format to 16 kHz mono WAV, which Whisper expects
        run(
            [
                FFMPEG,
                "-y",
                "-i", str(audio_path),
                "-ar", "16000",
                "-ac", "1",
                "-c:a", "pcm_s16le",
                str(wav_path),
            ],
            "ffmpeg",
        )

        # 2. Transcribe without timestamps
        output = run(
            [
                WHISPER_CLI,
                "-m", str(MODEL_PATH),
                "-f", str(wav_path),
                "-nt",
            ],
            "whisper-cli",
        )

    # 3. Clean up: remove the silence marker and join lines into one paragraph
    output = output.replace("[BLANK_AUDIO]", "")
    lines = [line.strip() for line in output.splitlines() if line.strip()]

    return " ".join(lines)
