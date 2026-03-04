# train_voice_model.py
import os
import numpy as np
import librosa
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# -------------------------------
# Config
# -------------------------------
DATA_PATH = r"C:\Users\User\Downloads\voice confidence evaluator for online interview\dataset_ravdess"

EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

MODEL_PATH = "emotion_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

# -------------------------------
# Feature extraction
# -------------------------------
def extract_features(file_path, sr=16000):
    """Extracts audio features from a single file."""
    y, _ = librosa.load(file_path, sr=sr, mono=True)
    if len(y) == 0:
        return None  # skip empty or corrupted files
    
    # Extract audio features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Aggregate statistics
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(contrast, axis=1),
        np.std(mfcc, axis=1),
        np.std(chroma, axis=1),
        np.std(contrast, axis=1)
    ])

    return features

def get_emotion_from_filename(filename):
    """Extracts emotion label from RAVDESS filename."""
    parts = filename.split("-")
    if len(parts) >= 3:
        return EMOTION_LABELS.get(parts[2], None)
    return None

# -------------------------------
# Load dataset
# -------------------------------
features_list, labels = [], []

print("🎵 Extracting RAVDESS features...")

for root, _, files in os.walk(DATA_PATH):
    for file in files:
        if not file.endswith(".wav"):
            continue
        emotion = get_emotion_from_filename(file)
        if emotion is None:
            continue
        
        file_path = os.path.join(root, file)
        feats = extract_features(file_path)
        if feats is None:
            print(f"⚠ Skipping {file} (empty/corrupted)")
            continue
        
        features_list.append(feats)
        labels.append(emotion)

if not features_list:
    raise RuntimeError("❌ No audio features extracted. Check dataset path or file structure!")

features_array = np.array(features_list)
labels_array = np.array(labels)

print(f"✅ Extracted {len(features_array)} samples across {len(np.unique(labels_array))} emotions.")

# -------------------------------
# Encode + Scale
# -------------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(labels_array)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(features_array)

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# -------------------------------
# Train model
# -------------------------------
print("🧠 Training RandomForest emotion model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate
# -------------------------------
y_pred = model.predict(X_test)

print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Optional: show confusion matrix summary
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
print("\n🧩 Confusion Matrix:")
print(cm_df)

# -------------------------------
# Save model artifacts
# -------------------------------
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le, ENCODER_PATH)

print(f"\n✅ Emotion model saved: {MODEL_PATH}")
print(f"✅ Scaler saved: {SCALER_PATH}")
print(f"✅ Label encoder saved: {ENCODER_PATH}")