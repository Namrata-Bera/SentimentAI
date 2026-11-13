"""Shared inference utilities."""
import json
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]

def load_assets():
    folder = ROOT / 'models'
    model = load_model(folder / 'sentiment_model.keras')
    tokenizer = tokenizer_from_json((folder / 'tokenizer.json').read_text(encoding='utf-8'))
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    return model, tokenizer, config

def predict_texts(texts, model, tokenizer, config):
    sequences = tokenizer.texts_to_sequences([clean_text(t) for t in texts])
    encoded = pad_sequences(sequences, maxlen=config['max_len'], padding='post', truncating='post')
    probabilities = model.predict(encoded, verbose=0)
    indexes = np.argmax(probabilities, axis=1)
    return [{'sentiment': config['labels'][int(i)], 'confidence': float(probabilities[j, i]),
             **{label.lower(): float(probabilities[j, k]) for k, label in enumerate(config['labels'])}}
            for j, i in enumerate(indexes)]
