import os
import json
import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image

# 📌 Define Paths
DATA_DIR = "data/fashion-dataset/"
STYLES_CSV = os.path.join(DATA_DIR, "styles.csv")
IMAGES_CSV = os.path.join(DATA_DIR, "images.csv")
STYLES_DIR = os.path.join(DATA_DIR, "styles/")

# ✅ Step 1: Load CSV Files
styles_df = pd.read_csv(STYLES_CSV)
images_df = pd.read_csv(IMAGES_CSV)

# ✅ Step 2: Load JSON Files Efficiently
def load_json_files(styles_dir, styles_df):
    json_data_list = []
    
    for idx, row in styles_df.iterrows():
        json_path = os.path.join(styles_dir, f"{row['id']}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                json_data = json.load(file)
                json_data_list.append(json_data['data'])
    
    # Convert list of JSON objects into a DataFrame
    json_df = pd.DataFrame(json_data_list)
    return json_df

json_df = load_json_files(STYLES_DIR, styles_df)

# ✅ Step 3: Merge All Data
fashion_data_df = styles_df.merge(images_df, on="id", how="left")
fashion_data_df = fashion_data_df.merge(json_df, on="id", how="left")

print(f"Data merged successfully! Total rows: {len(fashion_data_df)}")

# ✅ Step 4: Collaborative Filtering (SVD)
def train_collaborative_filtering(ratings_file):
    ratings_df = pd.read_csv(ratings_file)
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[['user_id', 'product_id', 'rating']], reader)
    
    trainset, testset = train_test_split(data, test_size=0.2)
    algo = SVD()
    algo.fit(trainset)
    
    return algo

# Example Usage
# algo = train_collaborative_filtering("user_product_ratings.csv")

# ✅ Step 5: Content-Based Filtering (KNN)
scaler = StandardScaler()
features = fashion_data_df[['price', 'year']].fillna(0)  # Select numeric features
features_scaled = scaler.fit_transform(features)

knn = NearestNeighbors(n_neighbors=5, metric='cosine')
knn.fit(features_scaled)

def find_similar_products(product_id):
    product_idx = fashion_data_df[fashion_data_df['id'] == product_id].index[0]
    distances, indices = knn.kneighbors([features_scaled[product_idx]])
    return fashion_data_df.iloc[indices[0]]

# ✅ Step 6: Image-Based Recommendations using ResNet50
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(img_array)

def extract_image_features(image_path):
    img_array = preprocess_image(image_path)
    features = resnet.predict(img_array)
    return features.flatten()
