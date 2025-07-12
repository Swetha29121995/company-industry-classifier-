import subprocess, sys

# Ensure spacy and joblib are installed
for pkg in ("spacy", "joblib"):
    try:
        __import__(pkg)
    except ModuleNotFoundError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)

# Ensure spaCy model is installed
import importlib
if not importlib.util.find_spec("en_core_web_sm"):
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)

import streamlit as st
import spacy
import joblib


nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

def tokenize_lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Load models and vectorizer
tfidf = joblib.load("tfidf_vectorizer.pkl")
lr_model = joblib.load("logreg_model.pkl")
svm_model = joblib.load("svm_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(page_title="Company Industry Classifier")
st.title("Company Industry Classifier")

st.markdown("Enter a company's business description to predict its industry using trained ML models.")

input_text = st.text_area("Company Description", "We develop AI-powered diagnostic tools for hospitals and medical research.")

if st.button("Predict Industry"):
    if input_text.strip():
        # Preprocess
        clean_text = tokenize_lemmatize(input_text)
        vectorized = tfidf.transform([clean_text])

        # Predict
        lr_pred = lr_model.predict(vectorized)
        svm_pred = svm_model.predict(vectorized)

        # Decode labels
        lr_label = le.inverse_transform(lr_pred)[0]
        svm_label = le.inverse_transform(svm_pred)[0]

        st.success("Predictions")
        st.write(f"**Logistic Regression:** {lr_label}")
        st.write(f"**SVM (LinearSVC):** {svm_label}")
    else:
        st.warning("Please enter a description to classify.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and spaCy")
