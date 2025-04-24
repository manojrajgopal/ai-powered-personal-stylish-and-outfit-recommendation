import os
import json
import pandas as pd
from typing import Dict, Any

class ProductMetadata:
    def __init__(self):
        # Define paths based on your existing structure
        self.images_dir = "../../data/fashion-dataset/images"
        self.csv_path = "../../data/fashion-dataset/fashion.csv"
        self.json_dir = "../../data/fashion-dataset/styles"
        
        # Load data
        self.csv_data = self._load_csv_data()
        self.json_data = self._load_json_data()
    
    def _load_csv_data(self) -> Dict[str, Dict[str, Any]]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(self.csv_path)
            return df.set_index('filename').to_dict('index')
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return {}
    
    def _load_json_data(self) -> Dict[str, Dict[str, Any]]:
        """Load data from JSON files"""
        json_data = {}
        try:
            for json_file in os.listdir(self.json_dir):
                if json_file.endswith('.json'):
                    with open(os.path.join(self.json_dir, json_file), 'r') as f:
                        data = json.load(f)
                        # Assuming JSON files contain product info with filename as key
                        json_data.update(data)
        except Exception as e:
            print(f"Error loading JSON data: {e}")
        return json_data
    
    def get_product_info(self, image_path: str) -> Dict[str, Any]:
        """Get product information by image path"""
        filename = os.path.basename(image_path)
        
        # Try CSV first
        product_info = self.csv_data.get(filename, {})
        
        # Fall back to JSON if not found in CSV
        if not product_info:
            product_info = self.json_data.get(filename, {})
        
        # Return default values if no info found
        return {
            'id': product_info.get('id', 'N/A'),
            'name': product_info.get('name', 'Unknown Product'),
            'price': product_info.get('price', 0),
            'brand': product_info.get('brand', 'Unknown Brand'),
            'category': product_info.get('category', 'Unknown Category')
        }