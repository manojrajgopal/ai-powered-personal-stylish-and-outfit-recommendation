import tensorflow as tf
import os

# Detect GPU availability
GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0

# Configuration
BATCH_SIZE = 128 if GPU_AVAILABLE else 32
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 8 if GPU_AVAILABLE else 4
BASE_DIR = "../../data/fashion-dataset"
STYLES_DIR = "../../data/fashion-dataset/styles"

# Export all variables
__all__ = ['GPU_AVAILABLE', 'BATCH_SIZE', 'IMAGE_SIZE', 'NUM_WORKERS', 'BASE_DIR', 'STYLES_DIR']

print(f"⚙️ Configuration loaded - GPU {'available' if GPU_AVAILABLE else 'not available'}")