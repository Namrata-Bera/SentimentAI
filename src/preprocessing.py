"""Lightweight normalization; keep negation and emojis for sentiment."""
import re
import html

def clean_text(text):
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    text = re.sub(r"@\w+", " USER ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
