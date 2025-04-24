import os
import json

def validate_paths(image_dir, csv_path, json_dir):
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

def save_recommendations(recommendations, file_path):
    with open(file_path, 'w') as f:
        json.dump(recommendations, f, indent=2)