import librosa
import numpy as np
from google import genai


def extract_audio_features(audio_path) -> dict:
    """
    Extract numerical audio features from a recording.

    Args:
        audio_path: Path to the audio file.

    Returns:/Users/home/Downloads/audio_analysis.py
        Dictionary containing audio measurements.
    """

    # Load audio at 22050 Hz
    y, sr = librosa.load(audio_path, sr=22050)

    duration = librosa.get_duration(y=y, sr=sr)

    if len(y) < 2048:
        raise ValueError("Audio recording is too short to analyze.")

    # ------------------------------------------
    # 1. RMS ENERGY
    # ------------------------------------------

    rms = librosa.feature.rms(y=y)[0]

    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))

    # ------------------------------------------
    # 2. SPECTRAL CENTROID
    # ------------------------------------------

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )[0]

    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))

    # ------------------------------------------
    # 3. ZERO CROSSING RATE
    # ------------------------------------------

    zcr = librosa.feature.zero_crossing_rate(y)[0]

    zcr_mean = float(np.mean(zcr))

    # ------------------------------------------
    # 4. FUNDAMENTAL FREQUENCY (PITCH)
    # ------------------------------------------

    pitch, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=65,
        fmax=400,
        sr=sr
    )

    valid_pitch = pitch[~np.isnan(pitch)]

    if len(valid_pitch) > 0:
        pitch_median = float(np.median(valid_pitch))
        pitch_std = float(np.std(valid_pitch))
    else:
        pitch_median = None
        pitch_std = None

    # ------------------------------------------
    # 5. SILENCE / PAUSE ESTIMATION
    # ------------------------------------------

    # RMS in decibels
    rms_db = librosa.amplitude_to_db(
        rms,
        ref=np.max
    )

    # Frames below -35 dB relative to the
    # recording's maximum are considered quiet.
    quiet_frames = rms_db < -35

    frame_duration = 512 / sr

    quiet_duration = float(
        np.sum(quiet_frames) * frame_duration
    )

    quiet_percentage = (
        (quiet_duration / duration) * 100
        if duration > 0 else 0
    )

    # ------------------------------------------
    # 6. RETURN FEATURES
    # ------------------------------------------

    features = {
        "duration_seconds": round(float(duration), 2),

        "rms_energy": {
            "mean": round(rms_mean, 4),
            "std": round(rms_std, 4)
        },

        "spectral_centroid_hz": {
            "mean": round(centroid_mean, 2),
            "std": round(centroid_std, 2)
        },

        "zero_crossing_rate": {
            "mean": round(zcr_mean, 4)
        },

        "pitch_hz": {
            "median": (
                round(pitch_median, 2)
                if pitch_median is not None else None
            ),
            "std": (
                round(pitch_std, 2)
                if pitch_std is not None else None
            )
        },

        "quiet_sections": {
            "estimated_duration_seconds": round(
                quiet_duration, 2
            ),
            "percentage_of_recording": round(
                quiet_percentage, 2
            )
        }
    }

    return features


# ------------------------------------------
# TESTING
# ------------------------------------------

if __name__ == "__main__":

    features = extract_audio_features("sample.wav")

    print("\n--- AUDIO ANALYSIS ---")

    for feature, value in features.items():
        print(f"{feature}: {value}")


