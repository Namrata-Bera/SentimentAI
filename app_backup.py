"""Professional Streamlit frontend for SentimentAI."""
import pandas as pd
import plotly.express as px
import streamlit as st
from src.predict import load_assets, predict_texts

st.set_page_config(page_title='SentimentAI | Review Intelligence', page_icon='💬', layout='wide')
st.markdown('''<style>
.block-container{padding-top:2rem;max-width:1150px}
h1{letter-spacing:-1.2px}
[data-testid="stMetric"]{border:1px solid #d8dee9;border-radius:12px;padding:14px}
</style>''', unsafe_allow_html=True)
st.title('💬 SentimentAI')
st.caption('Intelligent review & comment analysis · NLP + Bidirectional LSTM')

@st.cache_resource
def assets():
    return load_assets()

try:
    model, tokenizer, config = assets()
except (FileNotFoundError, OSError, ValueError) as exc:
    st.error('Model files are missing or invalid. Train first: `python -m src.train`, then run `python -m src.evaluate`.')
    st.caption(str(exc))
    st.stop()

single, bulk, about = st.tabs(['🔎 Analyze Review', '📊 Bulk Analytics', 'ℹ️ About'])
with single:
    st.subheader('Analyze a review')
    review = st.text_area('Review or comment', placeholder='Example: The delivery was fast and the product is excellent!', height=130)
    if st.button('Analyze sentiment', type='primary', disabled=not review.strip()):
        result = predict_texts([review], model, tokenizer, config)[0]
        a, b = st.columns(2)
        a.metric('Predicted sentiment', result['sentiment'])
        b.metric('Model confidence', f"{result['confidence']:.1%}")
        scores = pd.DataFrame({'Sentiment': config['labels'], 'Probability': [result[x.lower()] for x in config['labels']]})
        st.plotly_chart(px.bar(scores, x='Sentiment', y='Probability', color='Sentiment', range_y=[0,1], title='Class probabilities'), use_container_width=True)
        st.caption('Predictions are estimates and may be wrong, especially for sarcasm or ambiguous language.')
with bulk:
    st.subheader('Analyze a CSV of reviews')
    st.caption('Upload a CSV with a text column (for example: review, text, or comment). Maximum 5 MB / 5,000 rows per analysis.')
    uploaded = st.file_uploader('Choose CSV file', type='csv')
    if uploaded:
        if uploaded.size > 5 * 1024 * 1024:
            st.error('CSV exceeds the 5 MB limit.')
        else:
            try:
                df = pd.read_csv(uploaded)
                if df.empty:
                    st.warning('The uploaded CSV has no rows.')
                else:
                    column = st.selectbox('Select text column', df.columns.tolist())
                    if st.button('Run bulk analysis', type='primary'):
                        if len(df) > 5000:
                            st.error('Please upload at most 5,000 rows.')
                        else:
                            with st.spinner('Analyzing reviews...'):
                                results = predict_texts(df[column].fillna('').astype(str).tolist(), model, tokenizer, config)
                            output = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                            a, b, c = st.columns(3)
                            for metric, label in zip([a,b,c], config['labels']):
                                metric.metric(label, int((output['sentiment'] == label).sum()))
                            counts = output['sentiment'].value_counts().reindex(config['labels'], fill_value=0).reset_index()
                            counts.columns = ['Sentiment','Count']
                            st.plotly_chart(px.pie(counts, names='Sentiment', values='Count', title='Sentiment distribution'), use_container_width=True)
                            st.dataframe(output, use_container_width=True)
                            st.download_button('Download classified CSV', output.to_csv(index=False).encode('utf-8'), 'sentiment_results.csv', 'text/csv')
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f'Could not read CSV: {exc}')
with about:
    st.markdown('''### About this project
- **Model:** Embedding → Bidirectional LSTM → Dense → Softmax
- **Training data:** TweetEval sentiment (Negative, Neutral, Positive)
- **Preprocessing:** URL/mention normalization, tokenization, sequence padding
- **Evaluation:** held-out test split, confusion matrix, precision, recall, F1-score

**Note:** TweetEval consists of English tweets. Results may be less reliable on long product reviews, other languages, and domain-specific content. This tool is not a fact checker.''')
