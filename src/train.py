"""Download TweetEval sentiment data and train a three-class BiLSTM."""
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight
from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Negative', 'Neutral', 'Positive']
MAX_WORDS = 20000
MAX_LEN = 80

def main(epochs, batch_size, sample_size):
    tf.keras.utils.set_random_seed(42)
    (ROOT / 'models').mkdir(exist_ok=True)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    print('Downloading TweetEval sentiment dataset (internet required)...')
    ds = load_dataset('cardiffnlp/tweet_eval', 'sentiment')
    def extract(split):
        texts = [clean_text(x) for x in ds[split]['text']]
        labels = np.asarray(ds[split]['label'], dtype=np.int32)
        if sample_size:
            texts, labels = texts[:sample_size], labels[:sample_size]
        return texts, labels
    train_x, train_y = extract('train')
    val_x, val_y = extract('validation')
    test_x, test_y = extract('test')
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>', filters='')
    tokenizer.fit_on_texts(train_x)
    def encode(texts):
        return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding='post', truncating='post')
    model = tf.keras.Sequential([
        layers.Input(shape=(MAX_LEN,)),
        layers.Embedding(MAX_WORDS, 128, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    weights = compute_class_weight('balanced', classes=np.array([0,1,2]), y=train_y)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    history = model.fit(encode(train_x), train_y, validation_data=(encode(val_x), val_y),
                        epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                        class_weight=dict(enumerate(weights)), verbose=1)
    model.save(ROOT / 'models' / 'sentiment_model.keras')
    (ROOT / 'models' / 'tokenizer.json').write_text(tokenizer.to_json(), encoding='utf-8')
    (ROOT / 'models' / 'config.json').write_text(json.dumps({'max_len': MAX_LEN, 'labels': LABELS}), encoding='utf-8')
    (ROOT / 'outputs' / 'history.json').write_text(json.dumps(history.history), encoding='utf-8')
    np.savez_compressed(ROOT / 'outputs' / 'test_data.npz', x=encode(test_x), y=test_y)
    print('Saved trained model, tokenizer, config and held-out test set. Run python -m src.evaluate next.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--sample-size', type=int, default=None, help='Quick demo only: truncate each split')
    args = p.parse_args()
    main(args.epochs, args.batch_size, args.sample_size)
