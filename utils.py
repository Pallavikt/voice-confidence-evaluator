# utils.py
import os
import librosa
import numpy as np

def load_audio(filepath, sr=16000):
    """Load a WAV file and return audio array and sample rate"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found!")
    audio, sr = librosa.load(filepath, sr=sr)
    return audio, sr

def safe_filename(filename):
    """Sanitize filenames for CSV or display"""
    return filename.replace(" ", "_")