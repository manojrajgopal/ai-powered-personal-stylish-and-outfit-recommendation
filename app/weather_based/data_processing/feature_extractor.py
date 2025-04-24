# data_processing/feature_extractor.py

import numpy as np
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
import pickle
from tqdm import tqdm
from app_config import DATA_CONFIG, MODEL_CONFIG
import os

class FeatureExtractor:
    def __init__(self):
        self.image_features_path = DATA_CONFIG['image_features_path']
        self.feature_dim = MODEL_CONFIG['feature_dim']
        self.model = self._build_model()
        
    def _build_model(self):
        """Build feature extraction model"""
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        return base_model
    
    def extract_features(self, image_path):
        """Extract features for a single image"""
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = self.model.predict(x)
        return features.flatten()
    
    def extract_all_features(self, df):
        """Extract features for all images in dataframe"""
        if os.path.exists(self.image_features_path):
            with open(self.image_features_path, 'rb') as f:
                return pickle.load(f)
    
        features = {}
    
        # Process images in batches
        batch_size = 32
        image_paths = df['image_path'].tolist()
    
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
        
            for path in batch_paths:
                try:
                    img = image.load_img(path, target_size=(224, 224))
                    x = image.img_to_array(img)
                    x = preprocess_input(x)
                    batch_images.append(x)
                except:
                    batch_images.append(np.zeros((224, 224, 3)))
                
            batch_images = np.array(batch_images)
            batch_features = self.model.predict(batch_images)
        
            for path, feature in zip(batch_paths, batch_features):
                product_id = os.path.basename(path).split('.')[0]
                features[product_id] = feature.flatten()
    
        # Save features
        os.makedirs(os.path.dirname(self.image_features_path), exist_ok=True)
        with open(self.image_features_path, 'wb') as f:
            pickle.dump(features, f)
    
        return features