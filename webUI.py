import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="AutoJudge - Problem Difficulty Predictor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================
# CUSTOM CSS
# ================================
st.markdown("""
<style>
.main { padding: 2rem; }
.stButton>button {
    width: 100%;
    background-color: #4CAF50;
    color: white;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
    border-radius: 10px;
}
.prediction-box {
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
    text-align: center;
}
.easy-box { background-color: #d4edda; border: 2px solid #28a745; }
.medium-box { background-color: #fff3cd; border: 2px solid #ffc107; }
.hard-box { background-color: #f8d7da; border: 2px solid #dc3545; }
.score-box { background-color: #d1ecf1; border: 2px solid #17a2b8; }
</style>
""", unsafe_allow_html=True)

# ================================
# FEATURE ENGINEERING
# ================================
def combine_text(df):
    sample_io = df["sample_io"].apply(
        lambda x: " ".join(map(str, x)) if isinstance(x, list) else str(x)
    ) if "sample_io" in df.columns else ""

    return (
        df["title"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["input_description"].astype(str) + " " +
        df["output_description"].astype(str) + " " +
        sample_io
    )

class TextExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return combine_text(X)

class HandcraftedTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        text = combine_text(X).str.lower()
        features = pd.DataFrame(index=text.index)

        features["char_len"] = text.str.len()
        features["word_count"] = text.str.split().apply(len)
        features["digit_count"] = text.str.count(r"\d")
        features["math_symbols"] = text.str.count(r"[\+\-\*/=%]")
        features["line_count"] = text.str.count(r"\n")

        keywords = [
            "graph", "tree", "dp", "dynamic", "greedy",
            "dfs", "bfs", "binary", "search", "sort",
            "mod", "prime", "gcd", "lcm", "array",
            "matrix", "string", "bitmask"
        ]

        for kw in keywords:
            features[f"kw_{kw}"] = text.str.count(rf"\b{kw}\b")

        return features.fillna(0)

# ================================
# LOAD MODELS (ONLY 2 FILES)
# ================================
@st.cache_resource
def load_models():
    try:
        with open("autojudge_classifier.pkl", "rb") as f:
            classifier = pickle.load(f)

        with open("autojudge_regressor.pkl", "rb") as f:
            regressor = pickle.load(f)

        return classifier, regressor
    except FileNotFoundError:
        st.error("❌ Model files not found! Place both .pkl files in this folder.")
        return None, None

# ================================
# HEADER
# ================================
st.title("⚖️ AutoJudge: Programming Problem Difficulty Predictor")
st.markdown("---")

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("📊 About")
    st.write("""
    Predicts difficulty of programming problems using **machine learning**.
    - Easy / Medium / Hard classification
    - Numeric difficulty score
    - Text-based analysis
    """)

    st.markdown("---")
    st.header("👤 Developer")

# ================================
# MAIN UI
# ================================
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Enter Problem Details")

    title = st.text_input("Problem Title")
    description = st.text_area("Problem Description", height=200)
    input_description = st.text_area("Input Description", height=100)
    output_description = st.text_area("Output Description", height=100)

    predict_btn = st.button("🔮 Predict Difficulty")

with col2:
    st.header("🎯 Prediction")
    result_box = st.empty()

# ================================
# PREDICTION LOGIC
# ================================
if predict_btn:
    if not title or not description:
        st.error("⚠️ Title and Description are required.")
    else:
        classifier, regressor = load_models()

        if classifier is not None and regressor is not None:
            with st.spinner("Analyzing..."):
                input_df = pd.DataFrame({
                    "title": [title],
                    "description": [description],
                    "input_description": [input_description],
                    "output_description": [output_description],
                    "sample_io": [""]
                })

                try:
                    pred_class = classifier.predict(input_df)[0]
                    pred_score = float(regressor.predict(input_df)[0])
                    pred_score = max(0.0, pred_score)

                    with result_box.container():
                        st.success("✅ Prediction Complete")

                        if pred_class.lower() == "easy":
                            st.markdown("<div class='prediction-box easy-box'><h2>🟢 EASY</h2></div>", unsafe_allow_html=True)
                        elif pred_class.lower() == "medium":
                            st.markdown("<div class='prediction-box medium-box'><h2>🟡 MEDIUM</h2></div>", unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='prediction-box hard-box'><h2>🔴 HARD</h2></div>", unsafe_allow_html=True)

                        st.markdown(
                            f"<div class='prediction-box score-box'><h2>📊 Score: {pred_score:.2f}</h2></div>",
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

# ================================
# FOOTER
# ================================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>AutoJudge | Streamlit ML Project</p>",
    unsafe_allow_html=True
)

