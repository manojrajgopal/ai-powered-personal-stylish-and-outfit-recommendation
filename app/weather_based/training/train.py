# training/train.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.data_loader import FashionDataLoader
from data_processing.feature_extractor import FeatureExtractor
from models.recommender import FashionRecommender
from app_config import DATA_CONFIG, MODEL_CONFIG

def main():
    try:
        print("Starting training process...")
        
        # Step 1: Load data
        print("\nStep 1/4: Loading data...")
        data_loader = FashionDataLoader()
        df = data_loader.load_or_process_data()
        print(f"Data shape: {df.shape}")
        
        # Step 2: Extract image features
        print("\nStep 2/4: Extracting image features...")
        feature_extractor = FeatureExtractor()
        image_features = feature_extractor.extract_all_features(df)
        print(f"Features extracted for {len(image_features)} images")
        
        # Step 3: Train model
        print("\nStep 3/4: Training recommendation model...")
        recommender = FashionRecommender()
        processed_df = recommender.preprocess_data(df, image_features)
        print(f"Processing {len(processed_df)} items")
        
        # Debug: Check sample data
        print("\nSample data:")
        print(processed_df[['gender', 'category', 'price', 'description']].head())
        
        model = recommender.train_model(processed_df)
        
        # Step 4: Completion
        print("\nStep 4/4: Training completed successfully!")
        print(f"Model saved to {DATA_CONFIG['model_path']}")
        
    except Exception as e:
        print(f"\nTraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()