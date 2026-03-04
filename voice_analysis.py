import numpy as np
import librosa
import joblib
import os

# -------------------------------
# Model paths
# -------------------------------
EMOTION_MODEL_PATH = "emotion_model.pkl"
EMOTION_SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

CONF_MODEL_PATH = "final_confidence_model.pkl"
CONF_SCALER_PATH = "final_conf_scaler.pkl"

# -------------------------------
# Load models if available
# -------------------------------
# Emotion model
if all(os.path.exists(p) for p in [EMOTION_MODEL_PATH, EMOTION_SCALER_PATH, ENCODER_PATH]):
    emotion_model = joblib.load(EMOTION_MODEL_PATH)
    emotion_scaler = joblib.load(EMOTION_SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    print("✅ Emotion model loaded.")
else:
    emotion_model = emotion_scaler = le = None
    print("⚠ Emotion model not found. Using heuristic fallback.")

# Confidence model
if all(os.path.exists(p) for p in [CONF_MODEL_PATH, CONF_SCALER_PATH]):
    conf_model = joblib.load(CONF_MODEL_PATH)
    conf_scaler = joblib.load(CONF_SCALER_PATH)
    print("✅ Confidence model loaded.")
else:
    conf_model = conf_scaler = None
    print("⚠ Confidence model not found. Using heuristic fallback.")

# -------------------------------
# Extract audio features for confidence
# -------------------------------
def extract_confidence_features(audio, sr=16000):
    pitches, _ = librosa.piptrack(y=audio, sr=sr)
    pitches = pitches[pitches > 0]
    pitch_mean = np.mean(pitches) if len(pitches) > 0 else 0
    pitch_var = np.var(pitches) if len(pitches) > 0 else 0

    rms = librosa.feature.rms(y=audio)
    energy_mean = np.mean(rms)
    energy_var = np.var(rms)

    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration_sec = len(audio) / sr
    rate = len(onsets) / duration_sec if duration_sec > 0 else 0

    return np.array([pitch_mean, pitch_var, energy_mean, energy_var, rate])

# -------------------------------
# Heuristic emotion predictor
# -------------------------------
def predict_emotion_heuristic(audio, sr=16000):
    feat = extract_confidence_features(audio, sr)
    pitch_var = feat[1]
    energy = feat[2]

    if pitch_var < 50 and 0.01 < energy < 0.04:
        return "calm"
    elif energy > 0.05:
        return "energetic"
    else:
        return "stressed"

# -------------------------------
# Compute voice confidence + emotion
# -------------------------------
def compute_voice_confidence(audio, transcript="", sample_rate=16000):
    """
    Returns:
    - voice_conf (0-100)
    - text_conf (0-100)
    - final_conf (0-100)
    - predicted emotion (string)
    """

    # -------------------------------
    # Text confidence (0-100)
    # -------------------------------
    words = transcript.split() if transcript else []
    text_conf = min(100, len(words) * 10)  # 10 points per word

    # -------------------------------
    # Voice confidence (tweaked)
    # -------------------------------
    feat = extract_confidence_features(audio, sr=sample_rate)

    # Tweaked weights for calm speech
    pitch_score = np.clip((feat[0]-75)/(300-75),0,1)*35      # moderate max
    pitch_var_score = np.clip(1-feat[1]/500,0,1)*35          # higher weight for stable pitch
    energy_score = np.clip(feat[2]/0.05,0,1)*20              # reduced dependency on energy
    rate_score = np.clip(1-abs(feat[4]-4)/4,0,1)*10          # speaking rate contribution
    voice_conf = pitch_score + pitch_var_score + energy_score + rate_score
    voice_conf = np.clip(voice_conf,0,100)

    # -------------------------------
    # Final confidence
    # -------------------------------
    final_conf = 0.5*voice_conf + 0.5*text_conf   # balance voice and text
    final_conf = float(np.clip(final_conf,0,100))

    # -------------------------------
    # Predicted emotion
    # -------------------------------
    if emotion_model is not None:
        try:
            audio_feat = extract_confidence_features(audio, sr=sample_rate).reshape(1,-1)
            feat_scaled = emotion_scaler.transform(audio_feat)
            probs = emotion_model.predict_proba(feat_scaled)[0]
            pred_idx = np.argmax(probs)
            pred_emotion = le.inverse_transform([pred_idx])[0]
        except Exception as e:
            print(f"[Warning] Emotion fallback: {e}")
            pred_emotion = predict_emotion_heuristic(audio, sr=sample_rate)
    else:
        pred_emotion = predict_emotion_heuristic(audio, sr=sample_rate)

    return voice_conf, text_conf, final_conf, pred_emotion