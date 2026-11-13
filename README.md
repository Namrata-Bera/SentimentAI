# SentimentAI — Review & Comment Intelligence

A three-class (Negative / Neutral / Positive) sentiment classifier using TensorFlow Bidirectional LSTM, with a Streamlit interface for single reviews and CSV batch analysis. Includes confusion matrix and training curves.

## Requirements
- Windows, macOS or Linux; Python 3.11 recommended; internet for initial dataset download.
- A few GB of free space; CPU training may take time.

## Quick start (Windows PowerShell)
```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python -m src.train --epochs 10 --batch-size 64
python -m src.evaluate
python -m streamlit run app.py
```
If PowerShell execution policy blocks activation, use Command Prompt and `venv\Scripts\activate.bat`.

## Optional fast smoke test
`python -m src.train --epochs 1 --sample-size 1000` (not suitable for reporting final model accuracy). Retrain without `--sample-size` for a proper experiment.

## Dataset
[CardiffNLP TweetEval sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval), downloaded with Hugging Face `datasets`. Label mapping: 0=Negative, 1=Neutral, 2=Positive. Uses its official train, validation and test splits. No leakage: tokenizer fitted only on training data. Data is primarily English tweets; review-domain generalization is not guaranteed. Review the dataset license and usage terms before redistribution.

## Generated outputs
- `models/sentiment_model.keras`, `models/tokenizer.json`, `models/config.json`
- `outputs/confusion_matrix.png`, `outputs/accuracy_plot.png`, `outputs/loss_plot.png`, `outputs/classification_report.txt`

These outputs **do not exist until training and evaluation complete**. No accuracy values are claimed before evaluation.

## CSV format
Upload a CSV with a column such as `review`. See `data/sample_reviews.csv`.

## Project layout
- `app.py`: Streamlit app
- `src/preprocessing.py`: text cleaning
- `src/train.py`: dataset loading and BiLSTM training
- `src/predict.py`: model inference
- `src/evaluate.py`: held-out test evaluation and figures
- `tests/`: unit tests

## GitHub and deployment
Do not commit your local virtual environment. Large model files are ignored by `.gitignore`; for deployment, provide trained model artifacts through a suitable artifact store or include them explicitly if size and licensing permit. Streamlit Community Cloud needs model artifacts accessible at runtime. Deploying directly from this repository without them will show a missing-model message.
