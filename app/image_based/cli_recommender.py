import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from numpy.linalg import norm
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
from sklearn.neighbors import NearestNeighbors

class FashionRecommenderCLI:
    def __init__(self):
        # Initialize paths
        self.embeddings_path = 'app/image_based/embeddings.pkl'
        self.filenames_path = 'app/image_based/filenames.pkl'
        self.csv_path = os.path.join("data", "fashion-dataset", "fashion.csv")
        self.uploads_dir = 'uploads'
        os.makedirs(self.uploads_dir, exist_ok=True)
        
        # Load data
        self.feature_list = np.array(pickle.load(open(self.embeddings_path, 'rb')))
        self.filenames = pickle.load(open(self.filenames_path, 'rb'))
        self.product_df = pd.read_csv(self.csv_path)
        
        # Initialize model
        self.model = tf.keras.Sequential([
            ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
            GlobalMaxPooling2D()
        ])
    
    def save_uploaded_file(self, file_path):
        """Save uploaded file to uploads directory"""
        try:
            file_name = os.path.basename(file_path)
            save_path = os.path.join(self.uploads_dir, file_name)
            with open(file_path, 'rb') as src, open(save_path, 'wb') as dst:
                dst.write(src.read())
            return save_path
        except Exception as e:
            print(f"Error saving file: {e}")
            return None
    
    def extract_features(self, img_path):
        """Feature extraction from image"""
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img = np.expand_dims(img_array, axis=0)
        preprocessed = preprocess_input(expanded_img)
        features = self.model.predict(preprocessed).flatten()
        return features / norm(features)
    
    def find_similar_items(self, features, k=6):
        """Find similar items using KNN"""
        neighbors = NearestNeighbors(n_neighbors=k, metric='euclidean')
        neighbors.fit(self.feature_list)
        distances, indices = neighbors.kneighbors([features])
        return distances, indices
    
    def get_product_info(self, filename):
        """Extract product information from CSV using filename (without extension)"""
        try:
            product_id = os.path.splitext(filename)[0]
            product_row = self.product_df[self.product_df['id'] == int(product_id)].iloc[0]
            return {
                'id': product_id,
                'name': product_row['productDisplayName'],
                'price': product_row['price_usd']
            }
        except Exception as e:
            print(f"Error getting product info for {filename}: {e}")
            return {
                'id': 'N/A',
                'name': 'Unknown Product',
                'price': 0
            }

    
    def display_recommendations(self, query_path, indices, top_k):
        """Display recommendations as dictionary with product info"""
        
        recommendations = []
        
        
        for idx in indices[0][1:top_k + 1]:  # Get top 5 recommendations
            filename = os.path.basename(self.filenames[idx])
            recommendations.append(self.get_product_info(filename))
        
        return recommendations

def rec(image_path, top_k=20):
    recommender = FashionRecommenderCLI()
    
    try:
        print(f"\nProcessing image: {image_path}")
        saved_path = recommender.save_uploaded_file(image_path)
        if not saved_path:
            print("Error saving file.")
            return None
        
        features = recommender.extract_features(saved_path)
        distances, indices = recommender.find_similar_items(features, k=top_k + 1)
        
        # Get results as dictionary
        results = recommender.display_recommendations(saved_path, indices, top_k=top_k)
        return results
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
