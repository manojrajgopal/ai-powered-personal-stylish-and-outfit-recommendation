import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_CONFIG = {
    'image_dir': "../../data/fashion-dataset/images",
    'json_dir': "../../data/fashion-dataset/styles",
    'csv_path': "../../data/fashion-dataset/fashion.csv",
    'processed_data_path': os.path.join(BASE_DIR, 'data/processed_data.pkl'),
    'image_features_path': os.path.join(BASE_DIR, 'data/image_features.pkl'),
    'model_path': os.path.join(BASE_DIR, 'models/saved/recommender_model.pkl')
}

# Model parameters
MODEL_CONFIG = {
    'image_model': 'VGG16',
    'similarity_metric': 'cosine',
    'num_recommendations': 15,
    'feature_dim': 512  # Dimension for image features
}

# Training parameters
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42
}