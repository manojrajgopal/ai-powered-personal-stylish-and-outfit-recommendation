import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

class FashionModelTrainer:
    def __init__(self, styles_file, images_file):
        # Load datasets
        self.df_styles = pd.read_csv(styles_file, on_bad_lines='skip')
        self.df_images = pd.read_csv(images_file, on_bad_lines='skip')

        # Preprocess data
        self._preprocess_data()

        # Train ML models
        self._train_models()

    def _preprocess_data(self):
        # Merge datasets
        self.df_images[['id', 'image_format']] = self.df_images['filename'].str.extract(r'(\d+)\.(\w+)', expand=True)
        self.df_images = self.df_images[['id', 'link']]
        self.df_styles['id'] = self.df_styles['id'].astype(str)
        self.df_images['id'] = self.df_images['id'].astype(str)
        self.df_merged = pd.merge(self.df_styles, self.df_images, on='id', how='left')

        # Encode categorical variables
        self.label_encoders = {}
        categorical_cols = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour']
        for col in categorical_cols:
            le = LabelEncoder()
            self.df_merged[col] = le.fit_transform(self.df_merged[col].astype(str))
            self.label_encoders[col] = le

        # Normalize numerical features
        self.scaler = StandardScaler()
        numerical_cols = ['price_usd', 'discounted_price_usd']
        self.df_merged[numerical_cols] = self.scaler.fit_transform(self.df_merged[numerical_cols])

        # Text-based feature engineering
        self.tfidf = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_features = self.tfidf.fit_transform(self.df_merged['productDisplayName'].astype(str))
        self.df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=self.tfidf.get_feature_names_out())

    def _train_models(self):
        # Combine features
        X = pd.concat([self.df_merged[['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'price_usd', 'discounted_price_usd']], self.df_tfidf], axis=1)
        y = self.df_merged['articleType']  # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Random Forest Classifier
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clf.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.clf.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Save the model and preprocessing objects
        self._save_models()

    def _save_models(self):
        # Save the trained model and preprocessing objects
        joblib.dump(self.clf, 'models/fashion_recommend/fashion_model.pkl')
        joblib.dump(self.label_encoders, 'models/fashion_recommend/label_encoders.pkl')
        joblib.dump(self.scaler, 'models/fashion_recommend/scaler.pkl')
        joblib.dump(self.tfidf, 'models/fashion_recommend/tfidf.pkl')
        print("Models and preprocessing objects saved successfully.")

# Main execution
if __name__ == "__main__":
    trainer = FashionModelTrainer('data/fashion-dataset/styles.csv', 'data/fashion-dataset/images.csv')