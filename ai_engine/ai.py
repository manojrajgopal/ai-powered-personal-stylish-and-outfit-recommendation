import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load pre-trained models and encoders
try:
    with open("models/fashion_recommend/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("models/fashion_recommend/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("models/fashion_recommend/trained_feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    with open("models/fashion_recommend/label_encoders.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("models/fashion_recommend/fashion_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

def recommend_outfit(text_input):
    """Recommend an outfit based on text description."""
    try:
        # Transform input text using TF-IDF
        text_tfidf = tfidf.transform([text_input])
        
        # Ensure input feature size matches training feature size
        missing_features = set(feature_names) - set(tfidf.get_feature_names_out())
        if missing_features:
            print("Warning: Some features are missing in input text.")
        
        # Scale input features
        text_scaled = scaler.transform(text_tfidf.toarray())
        
        # Predict outfit category
        prediction = model.predict(text_scaled)
        outfit_category = label_encoder.inverse_transform(prediction)[0]
        
        return outfit_category
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return "No recommendation available."

# Example usage
if __name__ == "__main__":
    user_input = input("Enter a description of your outfit preference: ")
    recommended_outfit = recommend_outfit(user_input)
    print(f"Recommended Outfit: {recommended_outfit}")
