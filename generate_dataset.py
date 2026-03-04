# generate_dataset.py
import os
import csv
import numpy as np
from utils import load_audio, safe_filename
from voice_analysis import compute_voice_confidence, extract_confidence_features
from asr import transcribe_chunk, analyze_text_confidence

# -------------------------------
# Config
# -------------------------------
DATA_PATH = r"C:\Users\User\Downloads\voice confidence evaluator for online interview\dataset"
OUTPUT_CSV = "ravdess_dataset_full.csv"

EMOTION_LABELS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"
}


# -------------------------------
# Helpers
# -------------------------------
def get_emotion_from_filename(filename):
    parts = filename.split("-")
    if len(parts) >= 3:
        return EMOTION_LABELS.get(parts[2], "unknown")
    return "unknown"


# -------------------------------
# Dataset generator
# -------------------------------
def generate_dataset():
    results = []

    for root, _, files in os.walk(DATA_PATH):
        for file in files:
            if not file.endswith(".wav"):
                continue

            filepath = os.path.join(root, file)
            emotion = get_emotion_from_filename(file)
            if emotion == "unknown":
                continue

            print(f"🎧 Processing {file} ({emotion})")

            try:
                audio, sr = load_audio(filepath)
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
                continue

            # --- Voice confidence ---
            voice_conf, _, _, pred_emotion = compute_voice_confidence(
                audio, sample_rate=sr
            )

            # --- Text confidence ---
            transcript = transcribe_chunk(audio) or ""
            text_conf = analyze_text_confidence(transcript)

            # --- Final combined confidence ---
            final_conf = 0.7 * voice_conf + 0.3 * text_conf

            # --- Extract acoustic features ---
            audio_features = extract_confidence_features(audio, sr)
            features_dict = {
                f"feat_{i+1}": float(audio_features[i]) for i in range(len(audio_features))
            }

            results.append({
                "filename": safe_filename(file),
                "emotion": emotion,
                "pred_emotion": pred_emotion,
                "voice_conf": round(voice_conf, 2),
                "text_conf": round(text_conf, 2),
                "final_conf": round(final_conf, 2),
                "transcript": transcript,
                **features_dict
            })

    # -------------------------------
    # Save dataset
    # -------------------------------
    if not results:
        print("❌ No .wav files found!")
        return

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Dataset saved: {OUTPUT_CSV} ({len(results)} samples)")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    generate_dataset()