import streamlit as st
import pickle
import string
import nltk
import os
from utils import transform_text
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
import json
import time
from datetime import datetime
from train_model import train_and_save
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# -------------------- NLTK SETUP (FIX FOR STREAMLIT) --------------------
@st.cache_resource
def setup_nltk():
    nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', download_dir=nltk_data_dir)
        nltk.download('punkt_tab', download_dir=nltk_data_dir)

setup_nltk()
# -------------------------------------------------------------------------

# Page Config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")

# Sidebar
st.sidebar.title("Navigation")
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
app_mode = st.sidebar.selectbox("Choose Mode", ["Classifier", "History & Analytics", "Retrain Model"])

# Theme Styling
if theme == "Dark":
    st.markdown("""
    <style>
        .stApp { background-color: #0E1117; color: #FAFAFA; }
        .stTextArea textarea { background-color: #262730; color: #FAFAFA; }
        .stButton>button { color: #ffffff; background-color: #FF4B4B; border-radius: 5px; }
        .stSidebar { background-color: #262730; }
    </style>
    """, unsafe_allow_html=True)
    plt.style.use('dark_background')
else:
    st.markdown("""
    <style>
        .stApp { background-color: #FFFFFF; color: #000000; }
        .stTextArea textarea { background-color: #F0F2F6; color: #000000; }
        .stButton>button { color: #ffffff; background-color: #FF4B4B; border-radius: 5px; }
        .stSidebar { background-color: #F0F2F6; }
    </style>
    """, unsafe_allow_html=True)
    plt.style.use('default')

# Load Model
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}. Please retrain from sidebar.")
    model = None
    tfidf = None

HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(text, prediction, proba):
    history = load_history()
    entry = {
        'text': text,
        'prediction': prediction,
        'probability': proba,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'feedback': None,
        'actual_label': None
    }
    history.append(entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def update_last_history_feedback(feedback, actual_label):
    history = load_history()
    if history:
        history[-1]['feedback'] = feedback
        history[-1]['actual_label'] = actual_label
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)

def explain_prediction(transformed_text, model, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    input_vector = vectorizer.transform([transformed_text]).toarray()[0]
    feature_indices = input_vector.nonzero()[0]

    log_prob_ham = model.feature_log_prob_[0]
    log_prob_spam = model.feature_log_prob_[1]

    contributions = {}
    for idx in feature_indices:
        word = feature_names[idx]
        score = log_prob_spam[idx] - log_prob_ham[idx]
        contributions[word] = score

    sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    return sorted_contributions

# ================= CLASSIFIER =================
if app_mode == "Classifier":
    st.title("📩 Email/SMS Spam Classifier")
    input_sms = st.text_area("Enter the message", height=150)
    translate_check = st.checkbox("Enable Auto-Translation")

    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    if st.button('Predict'):
        if not input_sms.strip():
            st.warning("Please enter a message.")
        else:
            processed_text = input_sms

            # Translation
            if translate_check:
                try:
                    lang = detect(input_sms)
                    if lang != 'en':
                        translator = GoogleTranslator(source='auto', target='en')
                        processed_text = translator.translate(input_sms)
                        st.info(f"Translated Text: {processed_text}")
                except:
                    st.error("Translation failed.")

            transformed_sms, steps = transform_text(processed_text, debug=True)

            if tfidf:
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                proba = model.predict_proba(vector_input)[0]

                prediction_label = "Spam" if result == 1 else "Ham"
                confidence = proba[1] if result == 1 else proba[0]

                st.session_state.prediction_made = True
                st.session_state.result = result
                st.session_state.proba = proba
                st.session_state.transformed_sms = transformed_sms
                st.session_state.steps = steps

                save_history(input_sms, prediction_label, confidence)

    if st.session_state.prediction_made:
        result = st.session_state.result
        proba = st.session_state.proba

        if result == 1:
            st.header("🚨 Spam")
            st.markdown(f"Confidence: **{proba[1]*100:.2f}%**")
        else:
            st.header("✅ Not Spam")
            st.markdown(f"Confidence: **{proba[0]*100:.2f}%**")

# ================= HISTORY =================
elif app_mode == "History & Analytics":
    st.title("📊 Message History & Analytics")
    history = load_history()

    if history:
        df_history = pd.DataFrame(history)
        st.metric("Total Messages", len(df_history))
        st.dataframe(df_history.sort_values(by='timestamp', ascending=False))
    else:
        st.info("No history available yet.")

# ================= RETRAIN =================
elif app_mode == "Retrain Model":
    st.title("🔄 Retrain Model")
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding='latin-1')
        st.dataframe(df.head())

        if st.button("Retrain Model"):
            with st.spinner("Training..."):
                temp_path = "temp_dataset.csv"
                df.to_csv(temp_path, index=False)
                msg = train_and_save(temp_path)
                st.success(msg)
                os.remove(temp_path)
