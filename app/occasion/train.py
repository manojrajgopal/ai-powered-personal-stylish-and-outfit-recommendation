import numpy as np
from data_loader import ParallelFashionDataLoader
from feature_extractor import FeatureExtractor
from tqdm import tqdm
from config import BATCH_SIZE
import time

def train_system():
    
    # Initialize components
    data_loader = ParallelFashionDataLoader()
    feature_extractor = FeatureExtractor()
    
    features = []
    valid_ids = []
    
    start_time = time.time()
    for i in tqdm(range(0, len(data_loader.metadata), BATCH_SIZE)):
        batch_ids = data_loader.metadata['id'].iloc[i:i+BATCH_SIZE].tolist()
        batch_images = data_loader.load_image_batch(batch_ids)
        
        if batch_images:
            batch_features = feature_extractor.extract_features(np.array(batch_images))
            features.extend(batch_features.numpy())
            valid_ids.extend(batch_ids[:len(batch_images)])
    
    # Save results
    np.save('features.npy', np.array(features))
    data_loader.metadata.set_index('id').loc[valid_ids].to_pickle('metadata.pkl')
    
    total_time = (time.time() - start_time)/60

if __name__ == "__main__":
    train_system()