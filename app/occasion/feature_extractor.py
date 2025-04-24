import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Model
from config import IMAGE_SIZE, GPU_AVAILABLE

class FeatureExtractor:
    def __init__(self):
        # Initialize base model
        self.base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(*IMAGE_SIZE, 3)
        )
        # Create feature extraction model
        x = GlobalMaxPooling2D()(self.base_model.output)
        self.model = Model(inputs=self.base_model.input, outputs=x)
        
        # Warm-up the model
        dummy_input = tf.zeros((1, *IMAGE_SIZE, 3))
        _ = self.model.predict(dummy_input, verbose=0)
    
    def extract_features(self, batch_images):
        """Process batch of images (shape: [batch_size, 224, 224, 3])"""
        if batch_images is None or len(batch_images) == 0:
            return None
        
        # Convert and preprocess
        batch_images = tf.cast(batch_images, tf.float32)
        batch_images = tf.keras.applications.efficientnet.preprocess_input(batch_images)
        
        # Extract features
        return self.model(batch_images, training=False)