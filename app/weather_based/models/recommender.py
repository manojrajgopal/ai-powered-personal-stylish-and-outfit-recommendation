# models/recommender.py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from app.weather_based.app_config import DATA_CONFIG, MODEL_CONFIG

class FashionRecommender:
    def __init__(self):
        self.model_path = DATA_CONFIG['model_path']
        self.num_recommendations = MODEL_CONFIG['num_recommendations']
        
    def preprocess_data(self, df, image_features):
        """Prepare data for training"""
        df = df[df['id'].isin(image_features.keys())].copy()
        df = df.reset_index(drop=True)
        df['image_features'] = df['id'].map(image_features)
        df['description'] = df['description'].fillna('').astype(str)
        return df
    
    def train_model(self, df):
        """Train the recommendation model with proper ID preservation"""
        try:
            # Create a clean copy of the DataFrame to avoid modifying the original
            self.df = df.copy().reset_index(drop=True)
        
            # Verify ID column exists
            if 'id' not in self.df.columns:
                raise ValueError("DataFrame must contain 'id' column")

            # Process categorical features
            cat_features = ['gender', 'category', 'subcategory', 
                       'article_type', 'color', 'season', 'usage', 'age_group']
            cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            X_cat = cat_encoder.fit_transform(self.df[cat_features])
        
            # Process numerical features
            num_features = ['price']
            num_scaler = StandardScaler()
            X_num = num_scaler.fit_transform(self.df[num_features])
        
            # Process text features
            text_vectorizer = TfidfVectorizer(max_features=100)
            X_text = text_vectorizer.fit_transform(self.df['description'])
        
            # Process image features
            X_img = np.stack(self.df['image_features'].values)
        
            # Combine all features
            from scipy.sparse import hstack
            X_combined = hstack([X_cat, X_num, X_text]).toarray()
            X_final = np.hstack([X_combined, X_img])
        
            # Train model
            print("Training recommendation model...")
            self.model = NearestNeighbors(
                n_neighbors=self.num_recommendations * 3,
                metric=MODEL_CONFIG['similarity_metric'])
            self.model.fit(X_final)
        
            # Save model components
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            df_path = os.path.join(os.path.dirname(self.model_path), 'recommendation_data.pkl')
        
            # Save the DataFrame with original IDs
            self.df.to_pickle(df_path)
    
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'cat_encoder': cat_encoder,
                    'num_scaler': num_scaler,
                    'text_vectorizer': text_vectorizer,
                    'feature_columns': self.df.columns.tolist(),
                    'image_feature_dim': X_img.shape[1],
                    'df_path': df_path
                }, f)
        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            raise
    
    def load_model(self):
        """Load trained model components and DataFrame"""
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.cat_encoder = data['cat_encoder']
            self.num_scaler = data['num_scaler']
            self.text_vectorizer = data['text_vectorizer']
            self.feature_columns = data['feature_columns']
            self.image_feature_dim = data['image_feature_dim']
        
            # Load the DataFrame
            df_path = data['df_path']
            self.df = pd.read_pickle(df_path)
        
        return self.model 
    
    def get_recommendations(self, season, gender, age_group, n_recommendations=20):
        """Get recommendations with correct ID mapping"""
        try:
            # Filter with original IDs preserved
            filtered = self.df[
                (self.df['season'].str.lower() == season.lower()) &
                (self.df['gender'].str.lower() == gender.lower()) &
                (self.df['age_group'].str.lower() == age_group.lower())
            ].copy()
        
            if len(filtered) == 0:
                return pd.DataFrame()

            # Return with original IDs and relevant columns
            return filtered[['id', 'name', 'category', 'price']].sample(
                min(n_recommendations, len(filtered)),
                random_state=42  # For reproducible results
            ).reset_index(drop=True)
        
        except Exception as e:
            print(f"Recommendation error: {str(e)}")
            return pd.DataFrame()