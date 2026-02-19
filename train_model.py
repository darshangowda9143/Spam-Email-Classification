import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from utils import transform_text
import os

def train_and_save(data_path='spam.csv', model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    # Load data
    try:
        df = pd.read_csv(data_path, encoding='latin-1')
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding='utf-8')

    # Drop extra columns and rename
    df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')
    
    # Check if required columns exist
    if 'v1' in df.columns and 'v2' in df.columns:
        df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
    elif 'target' in df.columns and 'text' in df.columns:
        pass
    else:
        raise ValueError("CSV must have 'v1'/'v2' or 'target'/'text' columns.")

    # Encode target
    encoder = LabelEncoder()
    df['target'] = encoder.fit_transform(df['target'])

    # Remove duplicates
    df = df.drop_duplicates(keep='first')

    # Transform text
    print("Transforming text...")
    df['transformed_text'] = df['text'].apply(transform_text)

    # Vectorize
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df['transformed_text']).toarray()
    y = df['target'].values

    # Train model
    print("Training model...")
    mnb = MultinomialNB()
    mnb.fit(X, y)

    # Save
    print("Saving artifacts...")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf, f)
    with open(model_path, 'wb') as f:
        pickle.dump(mnb, f)

    return "Training completed successfully."

if __name__ == "__main__":
    train_and_save()
