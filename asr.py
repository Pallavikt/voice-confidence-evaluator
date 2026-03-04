# asr.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np

asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

reference_texts = ["happy","sad","angry","fear","disgust","neutral","surprise","calm"]

def transcribe_chunk(audio):
    if isinstance(audio, str):
        text = asr_pipeline(audio, chunk_length_s=30, ignore_warning=True)["text"]
    else:
        text = asr_pipeline(audio, chunk_length_s=30, ignore_warning=True)["text"]
    return text.strip()

def analyze_text_confidence(transcript):
    if not transcript:
        return 0
    words = transcript.split()
    length_score = min(100, len(words)*5)
    embeddings = sbert_model.encode([transcript])
    ref_embeddings = sbert_model.encode(reference_texts)
    sim_scores = util.cos_sim(embeddings, ref_embeddings).numpy()
    max_sim = np.max(sim_scores)*100
    return 0.6*length_score + 0.4*max_sim