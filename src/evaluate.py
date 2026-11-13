"""Evaluate trained model on held-out test split and plot metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.predict import load_assets

ROOT = Path(__file__).resolve().parents[1]

def main():
    model, _, config = load_assets()
    data = np.load(ROOT / 'outputs' / 'test_data.npz')
    actual = data['y']
    predicted = np.argmax(model.predict(data['x'], verbose=0), axis=1)
    labels = config['labels']
    report = classification_report(actual, predicted, labels=[0,1,2], target_names=labels, zero_division=0)
    accuracy = accuracy_score(actual, predicted)
    (ROOT / 'outputs' / 'classification_report.txt').write_text(f'Accuracy: {accuracy:.4f}\n\n{report}', encoding='utf-8')
    print(f'Accuracy: {accuracy:.4f}\n{report}')
    matrix = confusion_matrix(actual, predicted, labels=[0,1,2])
    plt.figure(figsize=(7,6))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.title('SentimentAI Confusion Matrix')
    plt.tight_layout(); plt.savefig(ROOT / 'outputs' / 'confusion_matrix.png', dpi=180); plt.close()
    history = json.loads((ROOT / 'outputs' / 'history.json').read_text(encoding='utf-8'))
    for metric in ['accuracy', 'loss']:
        plt.figure(figsize=(7,4))
        plt.plot(history[metric], label='Training')
        plt.plot(history['val_' + metric], label='Validation')
        plt.xlabel('Epoch'); plt.ylabel(metric.title()); plt.legend(); plt.tight_layout()
        plt.savefig(ROOT / 'outputs' / f'{metric}_plot.png', dpi=180); plt.close()
    print('Saved confusion matrix, classification report, and training plots in outputs/.')

if __name__ == '__main__':
    main()
