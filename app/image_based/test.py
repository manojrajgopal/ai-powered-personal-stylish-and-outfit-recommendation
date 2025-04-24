import os
import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import cv2
from product_metadata import ProductMetadata

product_metadata = ProductMetadata()

# ===== 1. SET YOUR PATHS =====
BASE_DIR = "../../data/fashion-dataset"
IMAGES_DIR = os.path.join(BASE_DIR, "images")  # Where your images are stored
EMBEDDINGS_PATH = os.path.join("embeddings.pkl")  # Path to embeddings
FILENAMES_PATH = os.path.join("filenames.pkl")  # Path to filenames

# ===== 2. LOAD DATA WITH PATH HANDLING =====
feature_list = np.array(pickle.load(open(EMBEDDINGS_PATH, 'rb')))
filenames = pickle.load(open(FILENAMES_PATH, 'rb'))

# Convert all filenames to absolute paths
filenames = [os.path.join(IMAGES_DIR, os.path.basename(f)) for f in filenames]

# ===== 3. MODEL SETUP (UNCHANGED) =====
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

# ===== 4. PROCESS SAMPLE IMAGE =====
sample_img_path = 'sample/test1.jpg'  # Keep as relative path
img = image.load_img(sample_img_path, target_size=(224,224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# ===== 5. FIND SIMILAR IMAGES =====
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)
distances, indices = neighbors.kneighbors([normalized_result])

# ===== 6. DISPLAY RESULTS WITH ERROR HANDLING =====
for idx in indices[0][1:6]:  # Skip first match (query image itself)
    img_path = filenames[idx]
    info = product_metadata.get_product_info(img_path)
    
    if not os.path.exists(img_path):
        continue
    
    temp_img = cv2.imread(img_path)
    if temp_img is None:
        continue
    
    # Create display window with product info
    display_text = (f"ID: {info['id']} | {info['name']}\n"
                   f"Price: ${info['price']:.2f} | Brand: {info['brand']}")
    
    # Create black background for text
    display_img = cv2.resize(temp_img, (512, 512))
    text_bg = np.zeros((50, display_img.shape[1], 3), dtype=np.uint8)
    display_img = np.vstack((display_img, text_bg))
    
    # Add text
    cv2.putText(display_img, display_text, (10, display_img.shape[0]-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('Product Recommendation', display_img)
    cv2.waitKey(0)