import streamlit as st
import spacy
import joblib
import os # Import os to check for file existence

# --- Initial setup for package listing (optional, for debugging/info) ---
# This part is useful for checking installed packages in a deployed environment.
try:
    import importlib.metadata
    installed_packages = sorted(d.metadata['Name'] for d in importlib.metadata.distributions())
    # st.sidebar.write("🔧 Installed packages:\n", "\n".join(installed_packages)) # You can uncomment to show in sidebar
except ImportError:
    # For Python versions older than 3.8 where importlib.metadata might not be available directly
    # or if there's an issue with the import.
    # print("importlib.metadata not available or failed to import.")
    pass

# --- Load spaCy model ---
# Ensure the spaCy model is downloaded. If not, run: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    st.error("SpaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm' in your terminal.")
    st.stop() # Stop the app if the model isn't available

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

# Check if all model files exist before attempting to load them
model_files_exist = True
for path in [TFIDF_MODEL_PATH, LOGREG_MODEL_PATH, SVM_MODEL_PATH, LABEL_ENCODER_PATH]:
    if not os.path.exists(path):
        st.error(f"Required model file not found: `{path}`. Please ensure all .pkl files are in the same directory.")
        model_files_exist = False
        break

if not model_files_exist:
    st.stop() # Stop the app if any model file is missing

# Load the models only if all files are present
try:
    tfidf = joblib.load(TFIDF_MODEL_PATH)
    lr_model = joblib.load(LOGREG_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
except Exception as e:
    st.error(f"Error loading model files: {e}. Please check if the files are valid .pkl files.")
    st.stop() # Stop the app if there's an error loading models

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
st.caption("Ensure `en_core_web_sm` spaCy model and all `.pkl` model files are present.")
