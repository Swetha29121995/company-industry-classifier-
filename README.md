# 🏢 Company Industry Classifier

A Streamlit-based machine learning app that classifies companies into their respective industries using Natural Language Processing (NLP) and TF-IDF-based feature extraction.

---

## 🚀 Demo

🔗 [Streamlit App Link](#) ← *Add your Streamlit link after deployment*

---

## 📌 Project Overview

This project classifies companies based on their descriptions using supervised machine learning models:

- ✅ Text Cleaning, Tokenization, Lemmatization
- ✅ TF-IDF Vectorization
- ✅ Multiclass Classification using:
  - Logistic Regression
  - SVM (LinearSVC)
  - Multinomial Naive Bayes (for comparison)

---

## 📁 Folder Structure

company-industry-classifier/
│
├── app.py # Streamlit web app
├── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
├── logreg_model.pkl # Logistic Regression model
├── svm_model.pkl # SVM model
├── label_encoder.pkl # Label encoder for target labels
├── requirements.txt # Python dependencies
└── company_classification.ipynb # (optional) training notebook


---

## 🔍 Example Predictions

| Company Description | Logistic Regression | SVM |
|---------------------|---------------------|-----|
| *We develop AI-powered diagnostic tools for hospitals.* | Healthcare | Healthcare |
| *We offer cloud infrastructure and cybersecurity solutions.* | IT Services | IT Services |
| *Leading textile manufacturer and exporter.* | Textiles | Textiles |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/company-industry-classifier.git
cd company-industry-classifier
pip install -r requirements.txt
streamlit run app.py


Models & Techniques
TF-IDF with ngram_range=(1, 2), max_features=1000

Encoding: LabelEncoder

Handling Imbalance: RandomOverSampler

Evaluation:

Accuracy

Classification Report

Confusion Matrix

Cross-Validation

Model Performance
Model	Accuracy	Notes
Naive Bayes	0.45	Weak for sparse features
Logistic Reg.	0.80–1.0	Performs very well
SVM (Linear)	1.0	Best performance

Deployment
Deployed via Streamlit Cloud

License
This project is licensed under the MIT License.

Author
Swetha Ravi
🔗 [LinkedIn](https://www.linkedin.com/in/swetha-ravi-618144196/)
📧 Email: swetha.ravi.sr@gmail.com
