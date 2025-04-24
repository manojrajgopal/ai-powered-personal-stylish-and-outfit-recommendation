import os
import pandas as pd
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from config import NUM_WORKERS, BASE_DIR, IMAGE_SIZE

class ParallelFashionDataLoader:
    def __init__(self):
        self.metadata = pd.read_csv(os.path.join(BASE_DIR, "fashion.csv"))
        self.executor = ThreadPoolExecutor(max_workers=NUM_WORKERS)
    
    def load_image_batch(self, batch_ids):
        futures = [self.executor.submit(self._load_single_image, img_id) for img_id in batch_ids]
        return [f.result() for f in futures if f.result() is not None]
    
    def _load_single_image(self, image_id):
        for ext in ['.jpg', '.jpeg', '.png']:
            img_path = os.path.join(BASE_DIR, "images", f"{image_id}{ext}")
            if os.path.exists(img_path):
                try:
                    return np.array(Image.open(img_path).convert('RGB').resize(IMAGE_SIZE))
                except:
                    continue
        return None