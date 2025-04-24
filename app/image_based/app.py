import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Initialize model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

def extract_features(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224,224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        return normalized_result
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return None

# Set your images directory (absolute path recommended)
images_dir = r"D:\Project\personal stylish\data\fashion-dataset\images"  # CHANGE THIS TO YOUR ACTUAL PATH

# Get all image files
filenames = []
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

for file in os.listdir(images_dir):
    if file.lower().endswith(valid_extensions):
        full_path = os.path.join(images_dir, file)
        filenames.append(full_path)

# Process images
feature_list = []
success_count = 0

for file in tqdm(filenames):
    features = extract_features(file, model)
    if features is not None:
        feature_list.append(features)
        success_count += 1

# Save results
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))