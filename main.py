import streamlit as st
import spacy
import joblib
import os
import sys # Import sys for exiting gracefully

# --- Initial setup for package listing (optional, for debugging/info) ---
# This part is useful for checking installed packages in a deployed environment.
# You can uncomment this if you want to see the installed packages in your Streamlit app's sidebar.
# try:
#     import importlib.metadata
#     installed_packages = sorted(d.metadata['Name'] for d in importlib.metadata.distributions())
#     st.sidebar.write("🔧 Installed packages:\n", "\n".join(installed_packages))
# except ImportError:
#     pass # For Python versions older than 3.8 or if importlib.metadata fails

# --- SpaCy Model Loading ---
SPACY_MODEL_NAME = "en_core_web_sm"

@st.cache_resource # Use st.cache_resource for heavy objects like spaCy models
def load_spacy_model():
    """
    Loads the spaCy model. If not found, attempts to download it.
    This function is cached to avoid re-downloading/reloading on every rerun.
    """
    try:
        nlp_model = spacy.load(SPACY_MODEL_NAME, disable=["ner", "parser"])
        return nlp_model
    except OSError:
        st.warning(f"SpaCy model '{SPACY_MODEL_NAME}' not found. Attempting to download...")
        with st.spinner(f"Downloading spaCy model '{SPACY_MODEL_NAME}'... This might take a moment."):
            try:
                # Use spacy.cli.download to get the model
                spacy.cli.download(SPACY_MODEL_NAME)
                nlp_model = spacy.load(SPACY_MODEL_NAME, disable=["ner", "parser"])
                st.success(f"SpaCy model '{SPACY_MODEL_NAME}' downloaded and loaded successfully!")
                return nlp_model
            except Exception as e:
                st.error(f"Failed to download spaCy model '{SPACY_MODEL_NAME}'. Please ensure your internet connection is stable or try again later.")
                st.error(f"Error details: {e}")
                st.stop() # Stop the app if model cannot be loaded/downloaded

nlp = load_spacy_model()

# Function to preprocess text using spaCy
def tokenize_lemmatize(text):
    """
    Tokenizes and lemmatizes the input text, removing non-alphabetic characters and stop words.
    """
    doc = nlp(text)
    # Filter for alphabetic tokens that are not stop words, then get their lemmas.
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# --- Load models and vectorizer ---
# Define the paths to your model files
TFIDF_MODEL_PATH = "tfidf_vectorizer.pkl"
LOGREG_MODEL_PATH = "logreg_model.pkl"
SVM_MODEL_PATH = "svm_model.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"

@st.cache_resource # Cache the loading of joblib models too
def load_ml_models():
    """
    Loads the machine learning models and vectorizer.
    This function is cached to avoid re-loading on every rerun.
    """
    # Check if all model files exist before attempting to load them
    model_files_exist = True
    for path in [TFIDF_MODEL_PATH, LOGREG_MODEL_PATH, SVM_MODEL_PATH, LABEL_ENCODER_PATH]:
        if not os.path.exists(path):
            st.error(f"Required model file not found: `{path}`. Please ensure all .pkl files are committed to your GitHub repository.")
            model_files_exist = False
            break

    if not model_files_exist:
        st.stop() # Stop the app if any model file is missing

    # Load the models only if all files are present
    try:
        tfidf_vec = joblib.load(TFIDF_MODEL_PATH)
        lr_m = joblib.load(LOGREG_MODEL_PATH)
        svm_m = joblib.load(SVM_MODEL_PATH)
        le_obj = joblib.load(LABEL_ENCODER_PATH)
        return tfidf_vec, lr_m, svm_m, le_obj
    except Exception as e:
        st.error(f"Error loading model files: {e}. Please check if the files are valid .pkl files.")
        st.stop() # Stop the app if there's an error loading models

tfidf, lr_model, svm_model, le = load_ml_models()

# --- Streamlit UI ---
st.set_page_config(page_title="Company Industry Classifier", layout="centered")

st.title("Company Industry Classifier")

st.markdown("""
    <p style='font-size: 1.1em;'>
    Enter a company's business description below to predict its industry
    using pre-trained Machine Learning models (Logistic Regression and SVM).
    </p>
""", unsafe_allow_html=True)

# Text area for user input with a default example
input_text = st.text_area(
    "Company Description",
    "We develop AI-powered diagnostic tools for hospitals and medical research.",
    height=150,
    help="Provide a detailed description of the company's core business activities."
)

# Predict button
if st.button("Predict Industry", type="primary"):
    if input_text.strip(): # Check if the input is not empty or just whitespace
        # Preprocess the input text
        with st.spinner("Processing description and predicting..."):
            clean_text = tokenize_lemmatize(input_text)
            vectorized = tfidf.transform([clean_text])

            # Make predictions
            lr_pred = lr_model.predict(vectorized)
            svm_pred = svm_model.predict(vectorized)

            # Decode numerical labels back to original industry names
            lr_label = le.inverse_transform(lr_pred)[0]
            svm_label = le.inverse_transform(svm_pred)[0]

        st.success("Predictions Complete!")
        st.write("---")
        st.subheader("Predicted Industries:")
        st.info(f"**Logistic Regression:** `{lr_label}`")
        st.info(f"**SVM (LinearSVC):** `{svm_label}`")
        st.write("---")
        st.markdown("These predictions are based on the patterns learned during the model training phase.")
    else:
        st.warning("Please enter a description to classify. The input field cannot be empty.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, scikit-learn, and spaCy.")
st.caption("Ensure all `.pkl` model files are committed to your GitHub repository.")
