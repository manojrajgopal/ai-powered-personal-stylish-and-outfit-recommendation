import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, 
    mean_squared_error, mean_absolute_error, r2_score,
    matthews_corrcoef
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics.pairwise import cosine_similarity

# Enhanced Classification Evaluation
def enhanced_classification_evaluation(y_true, y_pred, y_prob=None, classes=None, save_path='advanced_evaluation/classification'):
    os.makedirs(save_path, exist_ok=True)
    
    # Standard Metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)  # New: Robust metric
    }
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    # Save Metrics
    with open(f'{save_path}/metrics.txt', 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification Report:\n" + classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    # Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_path}/confusion_matrix.png')
    plt.close()
    
    return metrics

# Enhanced Regression Evaluation
def enhanced_regression_evaluation(y_true, y_pred, save_path='advanced_evaluation/regression'):
    os.makedirs(save_path, exist_ok=True)
    
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / np.maximum(1e-10, y_true)) * 100)  # Avoid division by zero
    
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),  # New: Interpretable %
        'r2': r2_score(y_true, y_pred)
    }
    
    # Save Metrics
    with open(f'{save_path}/metrics.txt', 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Residual Analysis
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals)
    plt.axhline(0, color='red')
    plt.title('Residual Analysis')
    plt.savefig(f'{save_path}/residuals.png')
    plt.close()
    
    return metrics

# Enhanced Recommendation Evaluation
def enhanced_recommendation_evaluation(true_items, recommended_items, k=10, save_path='advanced_evaluation/recommendation'):
    os.makedirs(save_path, exist_ok=True)
    
    # Diversity Calculation (New)
    def calculate_diversity(recommendations):
        all_items = list({item for sublist in recommendations.values() for item in sublist})
        if len(all_items) < 2:
            return 0.0
        # Simulate embeddings (replace with actual item features if available)
        embeddings = np.random.rand(len(all_items), 10)  
        sim_matrix = cosine_similarity(embeddings)
        return 1 - np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
    
    # Standard Metrics
    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    
    for user in true_items:
        if user not in recommended_items:
            continue
        relevant = set(true_items[user])
        recommended = recommended_items[user][:k]
        
        # Precision/Recall
        tp = len(set(recommended) & relevant)
        precision_scores.append(tp / len(recommended))
        recall_scores.append(tp / len(relevant)) if len(relevant) > 0 else 0
        
        # NDCG
        dcg = sum(1/np.log2(i+2) for i, item in enumerate(recommended) if item in relevant)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    
    metrics = {
        'precision@k': np.mean(precision_scores),
        'recall@k': np.mean(recall_scores),
        'ndcg@k': np.mean(ndcg_scores),
        'diversity': calculate_diversity(recommended_items)  # New
    }
    
    # Save Metrics
    with open(f'{save_path}/metrics.txt', 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    # Metric Distribution Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    sns.histplot(precision_scores, bins=10)
    plt.title('Precision@k Distribution')
    plt.subplot(132)
    sns.histplot(recall_scores, bins=10)
    plt.title('Recall@k Distribution')
    plt.subplot(133)
    sns.histplot(ndcg_scores, bins=10)
    plt.title('NDCG@k Distribution')
    plt.tight_layout()
    plt.savefig(f'{save_path}/metric_distributions.png')
    plt.close()
    
    return metrics

# Example Usage
if __name__ == "__main__":
    # Sample Data
    np.random.seed(42)
    
    # Classification Example
    y_true_class = np.random.randint(0, 3, 100)
    y_pred_class = np.random.randint(0, 3, 100)
    y_prob_class = np.random.rand(100, 3)
    classes = ['Class 0', 'Class 1', 'Class 2']
    
    print("Running Enhanced Classification Evaluation...")
    cls_metrics = enhanced_classification_evaluation(y_true_class, y_pred_class, y_prob_class, classes)
    
    # Regression Example
    y_true_reg = np.random.rand(100) * 100
    y_pred_reg = y_true_reg + np.random.normal(0, 10, 100)
    
    print("\nRunning Enhanced Regression Evaluation...")
    reg_metrics = enhanced_regression_evaluation(y_true_reg, y_pred_reg)
    
    # Recommendation Example
    true_items = {'user1': ['item1', 'item2'], 'user2': ['item3', 'item4']}
    recommended_items = {'user1': ['item1', 'item5', 'item3'], 'user2': ['item3', 'item6', 'item7']}
    
    print("\nRunning Enhanced Recommendation Evaluation...")
    rec_metrics = enhanced_recommendation_evaluation(true_items, recommended_items)
    
    print("\nEnhanced Evaluation Complete. Results saved in 'advanced_evaluation' directory.")