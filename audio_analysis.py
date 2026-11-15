#jaydon
import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

HOP_LENGTH = 512
MIN_PAUSE_SECONDS = 0.4   # silences shorter than this are just normal articulation


def _loudness(y):
    """Smoothed loudness per frame (dBFS) plus an adaptive 'quiet' threshold."""
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    db = librosa.amplitude_to_db(rms, ref=1.0)       # absolute level, not relative to the loudest spike
    db = uniform_filter1d(db, size=3)                # smooth out single-frame blips

    noise_floor = np.percentile(db, 10)
    speech_level = np.percentile(db, 90)

    # 35% of the way from this recording's noise floor up to its speaking volume
    threshold = noise_floor + 0.35 * (speech_level - noise_floor)
    return db, threshold


def find_pauses(y, sr):
    """
    Find pauses BETWEEN stretches of speech.
    Returns a list of pause lengths in seconds.
    """
    db, threshold = _loudness(y)
    is_quiet = db < threshold

    # Ignore silence before the first word and after the last word
    loud = np.where(~is_quiet)[0]
    if len(loud) == 0:
        return []
    is_quiet = is_quiet[loud[0]: loud[-1] + 1]

    frame_seconds = HOP_LENGTH / sr
    pauses = []
    run = 0
    for quiet in is_quiet:
        if quiet:
            run += 1
        else:
            if run:
                pauses.append(run * frame_seconds)
            run = 0

    return [p for p in pauses if p >= MIN_PAUSE_SECONDS]


def volume_fade_db(y):
    """
    How much louder/quieter the last third of the speaking is compared with
    the first third, in dB. Negative means the voice faded toward the end.
    """
    db, threshold = _loudness(y)
    speaking = db[db >= threshold]
    third = len(speaking) // 3
    if third < 5:
        return 0.0
    return float(np.mean(speaking[-third:]) - np.mean(speaking[:third]))


def extract_audio_features(audio_path) -> dict:
    """
    Extract numerical audio features from a recording.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Dictionary containing audio measurements.
    """

    y, sr = librosa.load(audio_path, sr=22050)

    if len(y) < 2048:
        raise ValueError("Audio recording is too short to analyze.")

    duration = len(y) / sr

    # 1. RMS ENERGY
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    # 2. SPECTRAL CENTROID
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    # 3. ZERO CROSSING RATE
    zcr = librosa.feature.zero_crossing_rate(y)[0]

    # 4. PITCH
    pitch, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    valid_pitch = pitch[~np.isnan(pitch)]

    if len(valid_pitch) > 0:
        pitch_median = float(np.median(valid_pitch))
        pitch_std = float(np.std(valid_pitch))
        # Variation in semitones is comparable across different voices
        pitch_var_semitones = float(
            np.std(12 * np.log2(valid_pitch / pitch_median))
        )
    else:
        pitch_median = pitch_std = pitch_var_semitones = None

    # 5. PAUSES
    pauses = find_pauses(y, sr)
    pause_total = float(sum(pauses))

    # 6. VOLUME FADE
    fade = volume_fade_db(y)

    return {
        "duration_seconds": round(float(duration), 2),

        "rms_energy": {
            "mean": round(float(np.mean(rms)), 4),
            "std": round(float(np.std(rms)), 4),
        },

        "spectral_centroid_hz": {
            "mean": round(float(np.mean(centroid)), 2),
            "std": round(float(np.std(centroid)), 2),
        },

        "zero_crossing_rate": {
            "mean": round(float(np.mean(zcr)), 4),
        },

        "pitch_hz": {
            "median": round(pitch_median, 2) if pitch_median is not None else None,
            "std": round(pitch_std, 2) if pitch_std is not None else None,
            "variation_semitones": (
                round(pitch_var_semitones, 2)
                if pitch_var_semitones is not None else None
            ),
        },

        "pauses": {
            "count": len(pauses),
            "average_seconds": round(float(np.mean(pauses)), 2) if pauses else 0.0,
            "longest_seconds": round(float(max(pauses)), 2) if pauses else 0.0,
            "total_seconds": round(pause_total, 2),
            "percentage_of_recording": round(pause_total / duration * 100, 2),
        },

        "volume_fade_db": round(fade, 2),
    }


if __name__ == "__main__":

    features = extract_audio_features("sample.wav")

    print("\n--- AUDIO ANALYSIS ---")

    for feature, value in features.items():
        print(f"{feature}: {value}")