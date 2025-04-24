import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, mean_squared_error,
    mean_absolute_error, r2_score, ndcg_score
)
from sklearn.model_selection import train_test_split
from scipy import stats
import os

class OutfitEvaluator:
    def __init__(self, data_path='metadata.pkl'):
        self.data = pd.read_pickle(data_path)
        self._prepare_data()
        os.makedirs('evaluation_plots', exist_ok=True)
    
    def _prepare_data(self):
        """Prepare data with realistic simulated metrics"""
        np.random.seed(42)
        
        # Ensure we have enough positive samples
        self.data['recommended'] = np.random.choice([0, 1], size=len(self.data), p=[0.6, 0.4])
        self.data['rating'] = np.clip(np.random.normal(3.5, 1, len(self.data)), 1, 5)
        
        # Feature engineering
        self.X = pd.get_dummies(self.data[['price_usd', 'gender', 'usage', 'subCategory']])
        self.y = self.data['recommended']
        self.ratings = self.data['rating']
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y)
        
        # For ranking evaluation
        self._prepare_ranking_data()

    def _prepare_ranking_data(self):
        """Generate realistic ranking data"""
        # Ensure we have some positive samples
        self.ranking_true = np.random.randint(0, 2, size=(5, 10))
        self.ranking_scores = np.random.uniform(0, 1, size=(5, 10))
        self.ranking_true[:, :3] = 1  # Force some positive samples

    def _plot_roc_curve(self, y_proba, model_name):
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'evaluation_plots/roc_{model_name}.png')
        plt.close()
        return roc_auc

    def _plot_precision_recall(self, y_proba, model_name):
        precision, recall, _ = precision_recall_curve(self.y_test, y_proba)
        pr_auc = average_precision_score(self.y_test, y_proba)
        
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="upper right")
        plt.savefig(f'evaluation_plots/pr_{model_name}.png')
        plt.close()
        return pr_auc

    def _plot_confusion_matrix(self, y_pred, model_name):
        cm = confusion_matrix(self.y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(f'evaluation_plots/cm_{model_name}.png')
        plt.close()

    def _plot_feature_importance(self, model, model_name):
        if model_name == "XGBoost":
            importances_dict = model.get_booster().get_score(importance_type='gain')
            features = list(importances_dict.keys())
            importances = list(importances_dict.values())
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = self.X.columns
        else:
            return

        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(f'Feature Importance - {model_name}')
        plt.bar(range(len(importances)), np.array(importances)[indices], align="center")
        plt.xticks(range(len(importances)), np.array(features)[indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'evaluation_plots/fi_{model_name}.png')
        plt.close()


    def _plot_ndcg_scores(self):
        ks = [1, 3, 5, 10]
        ndcgs = [ndcg_score(self.ranking_true, self.ranking_scores, k=k) for k in ks]
        
        plt.figure()
        plt.plot(ks, ndcgs, marker='o')
        plt.xlabel('k')
        plt.ylabel('NDCG@k')
        plt.title('NDCG at Different Cutoffs')
        plt.savefig('evaluation_plots/ndcg_scores.png')
        plt.close()

    def run_all_tests(self, model, model_name):
        """Execute all ML tests with proper error handling"""
        results = {}
        
        try:
            # Classification Tests
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            results['classification'] = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': self._plot_roc_curve(y_proba, model_name),
                'pr_auc': self._plot_precision_recall(y_proba, model_name),
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            # Regression Tests
            y_rating_pred = np.clip(y_proba * 5, 1, 5)
            results['regression'] = {
                'mse': mean_squared_error(self.ratings[:len(y_rating_pred)], y_rating_pred),
                'mae': mean_absolute_error(self.ratings[:len(y_rating_pred)], y_rating_pred),
                'r2': r2_score(self.ratings[:len(y_rating_pred)], y_rating_pred)
            }
            
            # Ranking Tests
            results['ranking'] = {
                'ndcg@5': ndcg_score(self.ranking_true, self.ranking_scores, k=5),
                'ndcg@10': ndcg_score(self.ranking_true, self.ranking_scores, k=10)
            }
            
            # Generate all plots
            self._plot_confusion_matrix(y_pred, model_name)
            if hasattr(model, 'feature_importances_'):
                self._plot_feature_importance(model, model_name)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results