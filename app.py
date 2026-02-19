import streamlit as st
import pickle
import string
import nltk
from utils import transform_text
import pandas as pd
from langdetect import detect
from deep_translator import GoogleTranslator
import json
import os
import time
from datetime import datetime
from train_model import train_and_save
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Page Config
st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩", layout="wide")

# Sidebar - Moved to top for Theme selection
st.sidebar.title("Navigation")
theme = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)
app_mode = st.sidebar.selectbox("Choose Mode", ["Classifier", "History & Analytics", "Retrain Model"])

# Custom CSS for Dark/Light Mode
if theme == "Dark":
    st.markdown("""
    <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stTextArea textarea {
            background-color: #262730;
            color: #FAFAFA;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #FF4B4B;
            border-radius: 5px;
        }
        .stSidebar {
            background-color: #262730;
        }
    </style>
    """, unsafe_allow_html=True)
    plt.style.use('dark_background')
else:
    st.markdown("""
    <style>
        .stApp {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stTextArea textarea {
            background-color: #F0F2F6;
            color: #000000;
        }
        .stButton>button {
            color: #ffffff;
            background-color: #FF4B4B;
            border-radius: 5px;
        }
        .stSidebar {
            background-color: #F0F2F6;
        }
    </style>
    """, unsafe_allow_html=True)
    plt.style.use('default')

# Load artifacts
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except Exception as e:
    st.error(f"Error loading model files: {e}. Please try retraining from the sidebar.")
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

def save_history(text, prediction, proba, feedback=None, actual_label=None):
    history = load_history()
    # Check if we are updating the last entry or adding a new one
    # Simple logic: append new. Updating is harder without unique IDs.
    # For now, we will append a new entry when prediction happens.
    # If feedback is provided later, we might need to update.
    # To simplify: We save history only on prediction. Feedback might need to handle last item.
    
    entry = {
        'text': text,
        'prediction': prediction,
        'probability': proba,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'feedback': feedback,
        'actual_label': actual_label
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
    """
    Explain why a message is spam using model coefficients (Log Likelihood Ratio).
    """
    feature_names = vectorizer.get_feature_names_out()
    
    # Get vector for the input
    input_vector = vectorizer.transform([transformed_text]).toarray()[0]
    
    # Get feature indices present in the input
    feature_indices = input_vector.nonzero()[0]
    
    # Calculate contribution: (Log Prob Spam - Log Prob Ham)
    # This represents how much more likely a word is to appear in Spam vs Ham
    log_prob_ham = model.feature_log_prob_[0]
    log_prob_spam = model.feature_log_prob_[1]
    
    contributions = {}
    for idx in feature_indices:
        word = feature_names[idx]
        # Importance score: (P(w|Spam) - P(w|Ham))
        score = log_prob_spam[idx] - log_prob_ham[idx]
        contributions[word] = score
        
    # Sort by importance
    sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_contributions

if app_mode == "Classifier":
    st.title("📩 Email/SMS Spam Classifier")
    st.markdown("Enter a message to check if it's **Spam** or **Ham** (Legitimate).")

    # Multi-language Support
    input_sms = st.text_area("Enter the message", height=150)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        translate_check = st.checkbox("Enable Auto-Translation (Non-English -> English)")
    
    # Initialize session state for feedback
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'last_text' not in st.session_state:
        st.session_state.last_text = ""
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    
    if st.button('Predict'):
        if not input_sms.strip():
            st.warning("Please enter a message.")
        else:
            processed_text = input_sms
            
            # 1. Language Detection & Translation
            if translate_check:
                try:
                    lang = detect(input_sms)
                    if lang != 'en':
                        with st.spinner(f"Detected language: {lang}. Translating..."):
                            translator = GoogleTranslator(source='auto', target='en')
                            processed_text = translator.translate(input_sms)
                            st.info(f"Translated Text: {processed_text}")
                except Exception as e:
                    st.error(f"Translation failed: {e}")

            # 2. Preprocess with Debug Info
            transformed_sms, steps = transform_text(processed_text, debug=True)
            
            # 3. Vectorize
            if tfidf:
                vector_input = tfidf.transform([transformed_sms])
                
                # 4. Predict
                result = model.predict(vector_input)[0]
                proba = model.predict_proba(vector_input)[0]
                
                prediction_label = "Spam" if result == 1 else "Ham"
                
                # Save to session state
                st.session_state.prediction_made = True
                st.session_state.last_text = input_sms
                st.session_state.last_prediction = prediction_label
                st.session_state.last_proba = proba
                st.session_state.transformed_sms = transformed_sms
                st.session_state.steps = steps
                st.session_state.result = result
                
                # Save initial history
                save_history(input_sms, prediction_label, proba[1] if result==1 else proba[0])

    # Display Results if prediction was made
    if st.session_state.prediction_made:
        result = st.session_state.result
        proba = st.session_state.last_proba
        steps = st.session_state.steps
        transformed_sms = st.session_state.transformed_sms
        
        # 5. Display Result
        if result == 1:
            st.header("🚨 Spam")
            st.markdown(f"Confidence: **{proba[1]*100:.2f}%**")
            
            # 6. Interpretability (Explain Why)
            st.subheader("🧐 Why is this Spam?")
            contributions = explain_prediction(transformed_sms, model, tfidf)
            
            # Filter positive contributions (Spam indicators)
            spam_words = [item for item in contributions if item[1] > 0]
            
            if spam_words:
                st.write("Top spam-triggering words:")
                
                # Create a DataFrame for visualization
                df_contrib = pd.DataFrame(spam_words, columns=['Word', 'Spam Score'])
                st.dataframe(df_contrib.head(10))
                
                # Visual: Bar Chart
                fig, ax = plt.subplots()
                sns.barplot(x='Spam Score', y='Word', data=df_contrib.head(10), ax=ax, palette='Reds_r')
                st.pyplot(fig)
            else:
                st.write("No specific spam keywords found, but the combination suggests spam.")
                
        else:
            st.header("✅ Not Spam")
            st.markdown(f"Confidence: **{proba[0]*100:.2f}%**")

        # 7. Testing Phase (Step-by-Step Visualization)
        with st.expander("🧪 Testing Phase: View Processing Steps", expanded=True):
            st.write("Here is how the model processed your message step-by-step:")
            
            cols = st.columns(5)
            with cols[0]:
                st.markdown("**1. Original**")
                st.code(steps.get('original', ''))
            with cols[1]:
                st.markdown("**2. Lowercase**")
                st.code(steps.get('lowercase', ''))
            with cols[2]:
                st.markdown("**3. Tokenized**")
                st.code(steps.get('tokenized', ''))
            with cols[3]:
                st.markdown("**4. No Stopwords**")
                st.code(steps.get('no_stopwords', ''))
            with cols[4]:
                st.markdown("**5. Stemmed (Final)**")
                st.code(steps.get('stemmed', ''))
                
            st.info(f"Final Input to Model: `{transformed_sms}`")

        # 8. Feedback Loop
        st.subheader("📝 Was this prediction correct?")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👍 Correct"):
                update_last_history_feedback("Correct", st.session_state.last_prediction)
                st.success("Thanks for your feedback! We've recorded this as a correct prediction.")
        
        with col2:
            if st.button("👎 Incorrect"):
                correct_label = "Ham" if st.session_state.last_prediction == "Spam" else "Spam"
                update_last_history_feedback("Incorrect", correct_label)
                st.error(f"Thanks for the correction! We've noted that this was actually {correct_label}.")

elif app_mode == "History & Analytics":
    st.title("📊 Message History & Analytics")
    
    history = load_history()
    
    if history:
        df_history = pd.DataFrame(history)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", len(df_history))
        col2.metric("Spam Count", len(df_history[df_history['prediction'] == "Spam"]))
        col3.metric("Ham Count", len(df_history[df_history['prediction'] == "Ham"]))
        
        # User Reported Accuracy
        if 'feedback' in df_history.columns:
            feedback_counts = df_history['feedback'].value_counts()
            correct_count = feedback_counts.get("Correct", 0)
            incorrect_count = feedback_counts.get("Incorrect", 0)
            total_feedback = correct_count + incorrect_count
            
            if total_feedback > 0:
                accuracy = (correct_count / total_feedback) * 100
                col4.metric("User Reported Accuracy", f"{accuracy:.1f}%")
            else:
                col4.metric("User Reported Accuracy", "N/A")
        else:
            col4.metric("User Reported Accuracy", "N/A")
        
        # Charts
        st.subheader("Spam vs Ham Distribution")
        fig, ax = plt.subplots()
        df_history['prediction'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'])
        st.pyplot(fig)
        
        st.subheader("Recent Messages")
        # Handle potential missing columns if old history exists
        display_cols = ['timestamp', 'prediction', 'text']
        if 'feedback' in df_history.columns:
            display_cols.append('feedback')
        if 'actual_label' in df_history.columns:
            display_cols.append('actual_label')
            
        st.dataframe(df_history[display_cols].sort_values(by='timestamp', ascending=False))
    else:
        st.info("No history available yet.")

elif app_mode == "Retrain Model":
    st.title("🔄 Retrain Model with New Data")
    
    uploaded_file = st.file_uploader("Upload a CSV file (Must have 'v1' and 'v2' columns or 'target' and 'text')", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            st.write("Preview:")
            st.dataframe(df.head())
            
            if st.button("Retrain Model"):
                with st.spinner("Retraining model... This may take a moment."):
                    # Save uploaded file temporarily
                    temp_path = "temp_dataset.csv"
                    df.to_csv(temp_path, index=False)
                    
                    # Call train function
                    msg = train_and_save(temp_path)
                    
                    st.success(msg)
                    st.info("Please reload the app (F5) to use the new model.")
                    
                    # Cleanup
                    os.remove(temp_path)
        except Exception as e:
            st.error(f"Error processing file: {e}")
