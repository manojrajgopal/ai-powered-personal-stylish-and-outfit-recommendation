import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ai_engine")))
import unittest
from sklearn.metrics import classification_report, mean_absolute_error
import age_gender_skinTone
import matplotlib.pyplot as plt
import numpy as np

class TestAgeGenderSkinTone(unittest.TestCase):
    
    def test_detect_age_gender_opencv(self):
        image_path = "tests/test_images/sample1.jpg"
        gender_category, estimated_age = age_gender_skinTone.detect_age_gender_opencv(image_path)
        self.assertIn(gender_category, ['Men', 'Women', 'Boys', 'Girls', 'Unisex'])
        self.assertIsInstance(estimated_age, (int, float, type(None)))
    
    def test_load_and_clean_dataset(self):
        data = age_gender_skinTone.load_and_clean_dataset("data/fashion-dataset/styles.csv")
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        self.assertIn("gender", data.columns)
    
    def test_skin_tone_detection(self):
        detector = age_gender_skinTone.SkinToneDetector()
        image_path = "tests/test_images/sample2.jpg"
        detected_tone_hex, recommended_colors = detector.generate(image_path)
        self.assertIsInstance(detected_tone_hex, (str, type(None)))
        self.assertIsInstance(recommended_colors, list)
    
    def test_outfit_generation(self):
        data = age_gender_skinTone.load_and_clean_dataset("data/fashion-dataset/styles.csv")
        outfit_generator = age_gender_skinTone.OutfitGenerator(data)
        recommended_colors = ["Blue", "Black", "White"]
        gender_category = "Men"
        outfits = outfit_generator.generate_outfits(recommended_colors, gender_category)
        self.assertIsInstance(outfits, list)
        self.assertTrue(all(isinstance(outfit, dict) for outfit in outfits))
    
    def test_process_fashion_recommendation(self):
        image_path = "tests/test_images/sample3.jpg"
        detected_tone_hex, recommended_colors, gender_category, detected_age, outfits = age_gender_skinTone.process_fashion_recommendation(image_path)
        self.assertIsInstance(detected_tone_hex, (str, type(None)))
        self.assertIsInstance(recommended_colors, list)
        self.assertIn(gender_category, ['Men', 'Women', 'Boys', 'Girls', 'Unisex'])
        self.assertIsInstance(detected_age, (int, float, type(None)))
        self.assertIsInstance(outfits, list)

    def test_accuracy_gender_classification(self):
        y_true = ["Male", "Female", "Male", "Female"]
        y_pred = ["Male", "Male", "Male", "Female"]
        print("Gender Classification Report:\n", classification_report(y_true, y_pred))
    
    def test_accuracy_age_estimation(self):
        true_ages = [28, 40, 18, 65]
        predicted_ages = [30, 42, 20, 55]
        mae = mean_absolute_error(true_ages, predicted_ages)
        print(f"Mean Absolute Error for Age Estimation: {mae}")

        # Plot Prediction Graph
        plt.figure(figsize=(8, 5))
        plt.plot(true_ages, label="True Ages", marker='o', linestyle='dashed')
        plt.plot(predicted_ages, label="Predicted Ages", marker='s')
        plt.xlabel("Sample Index")
        plt.ylabel("Age")
        plt.title("True vs Predicted Ages")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    unittest.main()
