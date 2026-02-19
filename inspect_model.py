import pickle
import sklearn

try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    print(f"Model type: {type(model)}")
    print(f"Model: {model}")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"Vectorizer type: {type(vectorizer)}")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
