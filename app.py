import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stButton>button {
    background-color: #00BFFF;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #009ACD;
    color: white;
}

[data-testid="metric-container"] {
    background-color: #1F2937;
    border-radius: 15px;
    padding: 15px;
    text-align: center;
}

h1, h2, h3 {
    color: #00BFFF;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown("""
<h1 style='text-align:center;'>
📰 Fake News Detection System
</h1>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# LOAD & TRAIN MODEL ONLY ONCE
# =========================
@st.cache_resource
def load_model():

    # Load dataset
    fake = pd.read_csv("data/Fake.csv")
    true = pd.read_csv("data/True.csv")

    # Labels
    fake["label"] = 0
    true["label"] = 1

    # Merge datasets
    df = pd.concat([fake, true], axis=0)

    # Keep required columns
    df = df[["text", "label"]]

    # Shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # Remove null values
    df = df.fillna("")

    # Features & Labels
    X = df["text"]
    y = df["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.7
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    # Model
    model = MultinomialNB()

    # Train model
    model.fit(X_train_vec, y_train)

    return model, vectorizer, fake, true, df

# Load cached model
model, vectorizer, fake, true, df = load_model()

# =========================
# SIDEBAR
# =========================
st.sidebar.title("📌 About Project")

st.sidebar.info("""
This project uses:
- NLP
- TF-IDF
- Machine Learning
- Naive Bayes Classifier
- Streamlit Dashboard
""")

# =========================
# METRICS
# =========================
col1, col2 = st.columns(2)

with col1:
    st.metric("📰 Fake News Articles", len(fake))

with col2:
    st.metric("✅ Real News Articles", len(true))

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.subheader("✍️ Enter News Article")

input_text = st.text_area(
    "Paste news article here"
)

# =========================
# PREDICTION
# =========================
if st.button("🚀 Predict News"):

    # Empty input check
    if input_text.strip() == "":
        st.warning("⚠️ Please enter news text")
        st.stop()

    # Short text check
    if len(input_text.split()) < 20:
        st.warning(
            "⚠️ Please enter longer news article for accurate prediction"
        )
        st.stop()

    # Transform input
    input_vec = vectorizer.transform([input_text])

    # Prediction
    prediction = model.predict(input_vec)[0]

    # Probability
    prob = model.predict_proba(input_vec)[0]

    confidence = max(prob) * 100

    st.markdown("---")

    st.subheader("📊 Prediction Result")

    st.progress(int(confidence))

    # Display result
    if prediction == 0:

        st.error(
            f"⚠️ This News is FAKE ({confidence:.2f}% confidence)"
        )

        result = "FAKE"

    else:

        st.success(
            f"✅ This News is REAL ({confidence:.2f}% confidence)"
        )

        result = "REAL"

    # =========================
    # REPORT GENERATION
    # =========================
    report_df = pd.DataFrame({
        "Prediction": [result],
        "Confidence": [f"{confidence:.2f}%"],
        "Text Length": [len(input_text.split())]
    })

    st.subheader("📄 Generated Report")

    st.dataframe(report_df)

    # =========================
    # DOWNLOAD REPORT
    # =========================
    csv = report_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Report",
        data=csv,
        file_name="fake_news_report.csv",
        mime="text/csv"
    )

# =========================
# DATASET PREVIEW
# =========================
st.markdown("---")

st.subheader("📂 Dataset Preview")

st.dataframe(df.head())

# =========================
# FOOTER
# =========================
st.markdown("---")

st.markdown("""
<center>
<h4>
Developed with ❤️ using NLP, Machine Learning & Streamlit
</h4>
</center>
""", unsafe_allow_html=True)