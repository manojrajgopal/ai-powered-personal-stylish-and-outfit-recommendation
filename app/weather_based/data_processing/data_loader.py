import os
import json
import pandas as pd
from tqdm import tqdm
import pickle
from app_config import DATA_CONFIG, MODEL_CONFIG

class FashionDataLoader:
    def __init__(self):
        self.image_dir = DATA_CONFIG['image_dir']
        self.json_dir = DATA_CONFIG['json_dir']
        self.csv_path = DATA_CONFIG['csv_path']
        self.processed_data_path = DATA_CONFIG['processed_data_path']
        
    def load_csv_data(self):
        """Load the main CSV file"""
        return pd.read_csv(self.csv_path)
    
    def load_json_data(self):
        """Load all JSON files"""
        json_data = {}
        for filename in tqdm(os.listdir(self.json_dir), desc="Loading JSON files"):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(self.json_dir, filename), 'r') as f:
                        data = json.load(f)
                        product_id = filename.split('.')[0]
                        json_data[product_id] = data['data']
                except:
                    continue
        return json_data
    
    def merge_data(self, df, json_data):
        """Merge CSV and JSON data"""
        enriched = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Merging data"):
            product_id = str(row['id'])
            if product_id in json_data:
                jd = json_data[product_id]
                image_path = os.path.join(self.image_dir, f"{product_id}.jpg")
                
                # Only include if image exists
                if os.path.exists(image_path):
                    enriched.append({
                        'id': product_id,
                        'gender': row['gender'],
                        'category': row['masterCategory'],
                        'subcategory': row['subCategory'],
                        'article_type': row['articleType'],
                        'color': row['baseColour'],
                        'season': row['season'],
                        'usage': row['usage'],
                        'name': row['productDisplayName'],
                        'price': row['price_usd'],
                        'image_path': image_path,
                        'age_group': jd.get('ageGroup', 'Adults'),
                        'fabric': jd.get('articleAttributes', {}).get('Fabric', ''),
                        'fit': jd.get('articleAttributes', {}).get('Fit', ''),
                        'description': jd.get('productDescriptors', {}).get('description', {}).get('value', '')
                    })
        return pd.DataFrame(enriched)
    
    def load_or_process_data(self):
        """Load processed data or create it if doesn't exist"""
        if os.path.exists(self.processed_data_path):
            return pd.read_pickle(self.processed_data_path)
        else:
            df = self.load_csv_data()
            json_data = self.load_json_data()
            enriched_df = self.merge_data(df, json_data)
            
            # Save processed data
            os.makedirs(os.path.dirname(self.processed_data_path), exist_ok=True)
            enriched_df.to_pickle(self.processed_data_path)
            return enriched_df