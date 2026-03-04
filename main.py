# main.py
import os
import librosa
import sounddevice as sd
from voice_analysis import compute_voice_confidence
from asr import transcribe_chunk

# -------------------------------
# Audio utilities
# -------------------------------
def load_audio(file_path, sr=16000):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    audio, sr = librosa.load(file_path, sr=sr)
    print(f"📂 Loaded audio file: {file_path}")
    return audio, sr


def record_audio(duration=5, sr=16000):
    print(f"🎙 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording complete.")
    return audio.flatten(), sr


# -------------------------------
# Evaluate audio (chunk by chunk)
# -------------------------------
def evaluate_audio(file_path=None, record=False):
    # Record or load
    if record:
        duration = int(input("Enter recording duration (seconds): ").strip() or 10)
        audio, sr = record_audio(duration=duration)
    else:
        audio, sr = load_audio(file_path)

    print("🎧 Processing speech...")
    transcript = transcribe_chunk(audio) or ""
    chunk_size = sr * 2  # 2-second chunks
    num_chunks = max(1, len(audio) // chunk_size)

    print("\n🔎 Analyzing in real time...\n")
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        audio_chunk = audio[start:end]

        voice_conf, text_conf, final_conf, pred_emotion = compute_voice_confidence(
            audio_chunk, transcript=transcript, sample_rate=sr
        )

        print(f"Chunk {i+1}/{num_chunks}: "
              f"🎤 Voice={voice_conf:.1f}% | 💬 Text={text_conf:.1f}% | "
              f"🔥 Final={final_conf:.1f}% | 🧠 Emotion={pred_emotion}")

    # -------------------------------
    # Final summary
    # -------------------------------
    print("\n📊 ===== FINAL REPORT =====")
    print(f"🗣 Transcript: {transcript}")
    print(f"🎤 Voice Confidence: {voice_conf:.2f}%")
    print(f"💬 Text Confidence: {text_conf:.2f}%")
    print(f"🔥 Final Confidence: {final_conf:.2f}%")
    print(f"🧠 Predicted Emotion: {pred_emotion}")

    if final_conf < 60:
        print("⚡ Feedback: Try to speak louder and more clearly with steady pacing.")
    elif final_conf < 80:
        print("💡 Good! A bit more energy and clarity can boost your confidence score.")
    else:
        print("✅ Excellent! Your delivery sounds confident and well-paced.")


# -------------------------------
# Main interactive menu
# -------------------------------
def main():
    print("🎧 Voice Confidence Evaluator\n")
    print("Select Mode:")
    print("1️⃣  Load Audio File")
    print("2️⃣  Record Real-Time Voice")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        file_path = input("Enter full path to your audio file (.wav or .mp3): ").strip()
        evaluate_audio(file_path=file_path)
    elif choice == "2":
        evaluate_audio(record=True)
    else:
        print("❌ Invalid choice. Exiting.")


if __name__ == "__main__":
    main()