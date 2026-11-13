
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.predict import load_assets, predict_texts

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------
st.set_page_config(
    page_title="SentimentAI | Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
from auth import show_authentication, logout
from database import initialize_database

initialize_database()

if not st.session_state.get("authenticated", False):
    show_authentication()
    st.stop()
from auth import show_authentication, logout

if not st.session_state.get("authenticated", False):
    show_authentication()
    st.stop()

ROOT = Path(__file__).resolve().parent

COLORS = {
    "Positive": "#22c55e",
    "Negative": "#fb7185",
    "Neutral": "#38bdf8"
}

# ---------------------------------------
# CUSTOM CSS
# ---------------------------------------
st.markdown("""
<style>
.stApp {
    background: #090e1c;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111a2d;
    border-right: 1px solid #27344b;
}

.block-container {
    max-width: 1350px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #f8fafc !important;
    letter-spacing: -0.5px;
}

.hero {
    background: linear-gradient(115deg, #192c54, #172044, #311e55);
    border: 1px solid #34446a;
    border-radius: 22px;
    padding: 34px;
    margin-bottom: 26px;
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    color: white;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #b9c5df;
    font-size: 16px;
}

.metric-card {
    background: #141e32;
    border: 1px solid #2b3953;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 15px;
}

.metric-label {
    color: #9caecb;
    font-size: 14px;
    margin-bottom: 12px;
}

.metric-value {
    color: #f8fafc;
    font-size: 32px;
    font-weight: 750;
}

.section-label {
    font-size: 18px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 22px 0 12px;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 650;
    padding: 12px 22px;
}

.stButton > button[kind="primary"]:hover {
    background: #7161ef;
    color: white;
}

[data-testid="stFileUploader"] {
    border-radius: 14px;
}

[data-testid="stMetric"] {
    background: #141e32;
    border: 1px solid #2b3953;
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------
@st.cache_resource
def get_model():
    return load_assets()


try:
    model, tokenizer, config = get_model()
except (OSError, ValueError, KeyError) as error:
    st.error("Model could not be loaded.")
    st.info("Train your model using: python -m src.train")
    st.code(str(error))
    st.stop()


# ---------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------
def metric_card(label, value, color="#f8fafc"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">
                {value}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def probability_chart(result):
    labels = config["labels"]
    probabilities = [result[label.lower()] for label in labels]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=probabilities,
        y=labels,
        orientation="h",
        marker_color=[COLORS.get(x, "#818cf8") for x in labels],
        text=[f"{p:.1%}" for p in probabilities],
        textposition="auto"
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141e32",
        plot_bgcolor="#141e32",
        font_color="#e2e8f0",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis_title="",
        xaxis_title="Prediction probability",
        height=300,
        margin=dict(l=20, r=25, t=25, b=35),
        showlegend=False
    )

    return fig


# ---------------------------------------
# SIDEBAR
# ---------------------------------------
with st.sidebar:
    user = st.session_state["user"]

    st.markdown(f"### 👋 {user['name']}")

    if st.button("🚪 Logout", use_container_width=True):
        logout()

    st.divider()
    st.markdown("## 🧠 SentimentAI")
    st.caption("INTELLIGENT REVIEW ANALYTICS")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔍 Analyze Review",
            "📊 Bulk Analytics",
            "📈 Model Performance",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.success("● LSTM model loaded")
    st.caption("NLP + TensorFlow + Bidirectional LSTM")
    st.caption("SentimentAI | AI/ML Project")


# ---------------------------------------
# OVERVIEW
# ---------------------------------------
if page == "🏠 Overview":

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Sentiment Intelligence</div>
        <div class="hero-subtitle">
            Transform customer reviews and comments into
            actionable insights using deep learning.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Classification Categories", "3")

    with c2:
        metric_card("Deep Learning Architecture", "BiLSTM")

    with c3:
        metric_card("Model Status", "Ready", "#22c55e")

    st.markdown(
        '<div class="section-label">Platform Capabilities</div>',
        unsafe_allow_html=True
    )

    a, b = st.columns(2)

    with a:
        st.info(
            "🔍 **Review Analysis**\n\n"
            "Classify individual customer reviews and "
            "inspect prediction probabilities."
        )

        st.info(
            "📊 **Bulk Analytics**\n\n"
            "Upload a CSV dataset, analyze reviews, "
            "and export classified results."
        )

    with b:
        st.info(
            "🧠 **Deep Learning**\n\n"
            "Uses a trained Bidirectional LSTM network "
            "for three-class sentiment classification."
        )

        st.info(
            "📈 **Performance Evaluation**\n\n"
            "Inspect evaluation reports, confusion matrix, "
            "and model training charts."
        )


# ---------------------------------------
# SINGLE REVIEW ANALYSIS
# ---------------------------------------
elif page == "🔍 Analyze Review":

    st.title("🔍 Review Sentiment Analysis")
    st.caption("Analyze text using your trained LSTM model.")

    review = st.text_area(
        "Enter customer review or comment",
        height=160,
        placeholder="Type a customer review here..."
    )

    if st.button(
        "✨ Analyze Sentiment",
        type="primary",
        disabled=not review.strip()
    ):
        with st.spinner("Analyzing sentiment..."):
            result = predict_texts(
                [review], model, tokenizer, config
            )[0]

        sentiment = result["sentiment"]
        confidence = result["confidence"]

        st.markdown(
            '<div class="section-label">Prediction Result</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            metric_card(
                "Predicted Sentiment",
                sentiment,
                COLORS.get(sentiment, "#f8fafc")
            )

        with col2:
            metric_card(
                "Model Confidence",
                f"{confidence:.1%}"
            )

        st.progress(float(confidence))

        st.markdown(
            '<div class="section-label">Class Probabilities</div>',
            unsafe_allow_html=True
        )

        st.plotly_chart(
            probability_chart(result),
            use_container_width=True
        )

        st.caption(
            "Confidence is the model's predicted probability, "
            "not a guarantee of correctness. Sarcasm, mixed "
            "opinions, and unfamiliar reviews may be misclassified."
        )


# ---------------------------------------
# BULK ANALYTICS
# ---------------------------------------
elif page == "📊 Bulk Analytics":

    st.title("📊 Bulk Sentiment Analytics")
    st.caption("Analyze up to 5,000 reviews from a CSV file.")

    uploaded = st.file_uploader(
        "Upload review dataset",
        type=["csv"]
    )

    if uploaded is not None:

        if uploaded.size > 5 * 1024 * 1024:
            st.error("Maximum supported file size is 5 MB.")
            st.stop()

        try:
            df = pd.read_csv(uploaded)
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as e:
            st.error(f"Unable to read CSV: {e}")
            st.stop()

        if df.empty:
            st.warning("The uploaded file contains no rows.")
            st.stop()

        st.dataframe(df.head(), use_container_width=True)

        text_column = st.selectbox(
            "Select review text column",
            df.columns.tolist()
        )

        if st.button("🚀 Run Bulk Analysis", type="primary"):

            if len(df) > 5000:
                st.error("Maximum 5,000 rows per analysis.")
                st.stop()

            reviews = df[text_column].fillna("").astype(str).tolist()

            with st.spinner("Classifying reviews..."):
                results = predict_texts(
                    reviews, model, tokenizer, config
                )

            output = pd.concat(
                [
                    df.reset_index(drop=True),
                    pd.DataFrame(results)
                ],
                axis=1
            )

            st.session_state["bulk_results"] = output

    if "bulk_results" in st.session_state:

        output = st.session_state["bulk_results"]

        counts = output["sentiment"].value_counts()

        a, b, c, d = st.columns(4)

        a.metric("Total Reviews", len(output))
        b.metric("Positive", counts.get("Positive", 0))
        c.metric("Negative", counts.get("Negative", 0))
        d.metric("Neutral", counts.get("Neutral", 0))

        chart_data = pd.DataFrame({
            "Sentiment": config["labels"],
            "Count": [
                counts.get(label, 0)
                for label in config["labels"]
            ]
        })

        left, right = st.columns(2)

        with left:
            fig = px.pie(
                chart_data,
                names="Sentiment",
                values="Count",
                hole=0.55,
                color="Sentiment",
                color_discrete_map=COLORS
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#141e32",
                font_color="#e2e8f0"
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.bar(
                chart_data,
                x="Sentiment",
                y="Count",
                color="Sentiment",
                color_discrete_map=COLORS
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#141e32",
                font_color="#e2e8f0",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Classified Reviews")
        st.dataframe(output, use_container_width=True)

        st.download_button(
            "⬇️ Download Analysis Results",
            data=output.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_analysis_results.csv",
            mime="text/csv"
        )


# ---------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------
elif page == "📈 Model Performance":

    st.title("📈 Model Performance")
    st.caption("Evaluation artifacts from the held-out test dataset.")

    st.info(
        "These charts are generated by the evaluation script. "
        "No performance metrics are estimated or invented."
    )

    files = [
        ("Confusion Matrix", "confusion_matrix.png"),
        ("Training Accuracy", "accuracy_plot.png"),
        ("Training Loss", "loss_plot.png")
    ]

    for title, filename in files:
        path = ROOT / "outputs" / filename

        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
        else:
            st.warning(f"{filename} has not been generated.")

    report = ROOT / "outputs" / "classification_report.txt"

    if report.exists():
        st.subheader("Classification Report")
        st.code(report.read_text(encoding="utf-8"))
    else:
        st.info(
            "Run python -m src.evaluate to generate "
            "the classification report."
        )


# ---------------------------------------
# ABOUT PROJECT
# ---------------------------------------
elif page == "ℹ️ About Project":

    st.title("ℹ️ About SentimentAI")

    st.markdown("""
    ### Project Overview

    SentimentAI is a deep learning application designed
    to classify text into Positive, Negative, and Neutral
    sentiment categories.

    ### Technology Stack

    - **Language:** Python
    - **NLP:** NLTK
    - **Deep Learning:** TensorFlow / Keras
    - **Architecture:** Bidirectional LSTM
    - **Frontend:** Streamlit
    - **Visualization:** Plotly
    - **Data Processing:** Pandas, NumPy
    - **Evaluation:** Scikit-learn

    ### Model Architecture

    Input Text → Text Preprocessing → Tokenization
    → Embedding → Bidirectional LSTM → Dense → Softmax

    ### Dataset

    TweetEval sentiment dataset containing English tweets
    labeled as Negative, Neutral, or Positive.

    ### Limitations

    Predictions may be unreliable for sarcasm, mixed
    sentiment, long product reviews, and languages or
    domains that differ from the training data.
    """)

st.divider()
st.caption(
    "SentimentAI | Intelligent Review Analytics "
    "| Powered by NLP & Deep Learning"
)
