import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

def evaluate_model(y_true, y_pred, y_prob=None, classes=None, model=None, feature_names=None, save_path='evaluation_metrics'):
    """
    Comprehensive evaluation of the model with all major metrics and visualizations.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC and PR curves)
        classes: List of class names
        model: Trained model object (optional, for feature importance)
        feature_names: List of feature names (optional)
        save_path: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Basic metrics - ensure they return scalar values
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
    recall = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    print("\n" + "="*50)
    print("Basic Classification Metrics")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Classification report
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    # Confusion matrix
    if classes is not None:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))
        plt.close()
    
    # ROC Curve (for binary or multiclass)
    if y_prob is not None:
        if len(np.unique(y_true)) <= 2:  # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'roc_curve.png'))
            plt.close()
            
            # Precision-Recall curve
            precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_precision = average_precision_score(y_true, y_prob[:, 1])
            
            plt.figure()
            plt.plot(recall_pr, precision_pr, color='blue', lw=2,
                     label=f'Precision-Recall (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall Curve')
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'precision_recall_curve.png'))
            plt.close()
            
        else:  # Multiclass classification
            # Binarize the output
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            # Plot all ROC curves
            plt.figure(figsize=(8, 6))
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)
            
            colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
            for i, color in zip(range(n_classes), colors):
                plt.plot(fpr[i], tpr[i], color=color, lw=2,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(classes[i] if classes else str(i), roc_auc[i]))
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Multiclass ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'multiclass_roc_curve.png'))
            plt.close()
            
            # Precision-Recall curve for multiclass
            precision_pr = dict()
            recall_pr = dict()
            average_precision = dict()
            
            for i in range(n_classes):
                precision_pr[i], recall_pr[i], _ = precision_recall_curve(y_true_bin[:, i],
                                                                   y_prob[:, i])
                average_precision[i] = average_precision_score(y_true_bin[:, i], 
                                                              y_prob[:, i])
            
            # A "micro-average": quantifying score on all classes jointly
            precision_pr["micro"], recall_pr["micro"], _ = precision_recall_curve(
                y_true_bin.ravel(), y_prob.ravel())
            average_precision["micro"] = average_precision_score(y_true_bin, y_prob,
                                                               average="micro")
            
            plt.figure(figsize=(8, 6))
            f_scores = np.linspace(0.2, 0.8, num=4)
            lines = []
            labels = []
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
                plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))
            
            lines.append(l)
            labels.append('iso-f1 curves')
            
            colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
            
            for i, color in zip(range(n_classes), colors):
                l, = plt.plot(recall_pr[i], precision_pr[i], color=color, lw=2)
                lines.append(l)
                labels.append('Precision-recall for class {0} (AP = {1:0.2f})'
                              ''.format(classes[i] if classes else str(i), average_precision[i]))
            
            plt.plot(recall_pr["micro"], precision_pr["micro"], color='gold', lw=2,
                     label='micro-average Precision-recall (AP = {0:0.2f})'
                           ''.format(average_precision["micro"]))
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Multiclass Precision-Recall Curve')
            plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=10))
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'multiclass_precision_recall_curve.png'))
            plt.close()
    
    # Feature importance (if available)
    if model is not None and hasattr(model, 'feature_importances_') and feature_names is not None:
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(importances)), importances[indices],
                   color="r", align="center")
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.xlim([-1, len(importances)])
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'feature_importances.png'))
            plt.close()
        except Exception as e:
            print(f"Could not plot feature importances: {e}")
    
    # Save metrics to file
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
    }
    
    with open(os.path.join(save_path, 'metrics.txt'), 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true, y_pred, target_names=classes, zero_division=0))
    
    return metrics

def evaluate_regression(y_true, y_pred, save_path='evaluation_metrics'):
    """
    Evaluate regression performance with various metrics and visualizations.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Directory to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("\n" + "="*50)
    print("Regression Metrics")
    print("="*50)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Save metrics to file
    with open(os.path.join(save_path, 'regression_metrics.txt'), 'w') as f:
        f.write(f"Mean Squared Error (MSE): {mse:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"R-squared (R²): {r2:.4f}\n")
    
    # Residual plot
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'residual_plot.png'))
    plt.close()
    
    # Actual vs Predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'actual_vs_predicted.png'))
    plt.close()
    
    # Distribution of errors
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'residual_distribution.png'))
    plt.close()
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }

def evaluate_recommendation_system(true_items, recommended_items, k=10, save_path='evaluation_metrics'):
    """
    Evaluate recommendation system performance.
    
    Args:
        true_items: Dictionary of user_id to list of true items
        recommended_items: Dictionary of user_id to list of recommended items
        k: Top-k items to consider
        save_path: Directory to save visualizations
    """
    os.makedirs(save_path, exist_ok=True)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    ndcg_scores = []
    
    for user_id in true_items:
        if user_id not in recommended_items:
            continue
            
        true_positives = set(true_items[user_id]) & set(recommended_items[user_id][:k])
        false_positives = set(recommended_items[user_id][:k]) - set(true_items[user_id])
        false_negatives = set(true_items[user_id]) - set(recommended_items[user_id][:k])
        
        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
        recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # NDCG calculation
        dcg = 0
        idcg = 0
        for i, item in enumerate(recommended_items[user_id][:k]):
            if item in true_items[user_id]:
                dcg += 1 / np.log2(i + 2)
        
        for i in range(min(len(true_items[user_id]), k)):
            idcg += 1 / np.log2(i + 2)
            
        ndcg = dcg / idcg if idcg > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        ndcg_scores.append(ndcg)
    
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    avg_ndcg = np.mean(ndcg_scores)
    
    print("\n" + "="*50)
    print("Recommendation System Metrics")
    print("="*50)
    print(f"Average Precision@{k}: {avg_precision:.4f}")
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    print(f"Average F1@{k}: {avg_f1:.4f}")
    print(f"Average NDCG@{k}: {avg_ndcg:.4f}")
    
    # Save metrics to file
    with open(os.path.join(save_path, 'recommendation_metrics.txt'), 'w') as f:
        f.write(f"Average Precision@{k}: {avg_precision:.4f}\n")
        f.write(f"Average Recall@{k}: {avg_recall:.4f}\n")
        f.write(f"Average F1@{k}: {avg_f1:.4f}\n")
        f.write(f"Average NDCG@{k}: {avg_ndcg:.4f}\n")
    
    # Plot metrics distribution
    plt.figure(figsize=(12, 8))
    metrics = [precision_scores, recall_scores, f1_scores, ndcg_scores]
    labels = [f'Precision@{k}', f'Recall@{k}', f'F1@{k}', f'NDCG@{k}']
    
    for i, (metric, label) in enumerate(zip(metrics, labels)):
        plt.subplot(2, 2, i+1)
        sns.histplot(metric, kde=True)
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {label}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'recommendation_metrics_distribution.png'))
    plt.close()
    
    return {
        f'precision@{k}': float(avg_precision),
        f'recall@{k}': float(avg_recall),
        f'f1@{k}': float(avg_f1),
        f'ndcg@{k}': float(avg_ndcg)
    }

if __name__ == "__main__":
    # Example usage with proper data types
    np.random.seed(42)
    
    # Create evaluation directory
    eval_dir = "example_evaluation"
    os.makedirs(eval_dir, exist_ok=True)
    
    # Classification example with proper data
    print("\nRunning Classification Evaluation...")
    y_true_class = np.random.randint(0, 3, 100)
    y_pred_class = np.random.randint(0, 3, 100)
    y_prob_class = np.random.rand(100, 3)
    classes = ['Class 0', 'Class 1', 'Class 2']
    
    class_metrics = evaluate_model(
        y_true_class, 
        y_pred_class, 
        y_prob_class, 
        classes,
        save_path=os.path.join(eval_dir, 'classification')
    )
    
    # Regression example
    print("\nRunning Regression Evaluation...")
    y_true_reg = np.random.rand(100) * 100
    y_pred_reg = y_true_reg + np.random.normal(0, 10, 100)
    
    reg_metrics = evaluate_regression(
        y_true_reg, 
        y_pred_reg,
        save_path=os.path.join(eval_dir, 'regression')
    )
    
    # Recommendation system example
    print("\nRunning Recommendation System Evaluation...")
    true_items = {
        'user1': ['item1', 'item2', 'item3'],
        'user2': ['item4', 'item5'],
        'user3': ['item1', 'item6']
    }
    recommended_items = {
        'user1': ['item1', 'item7', 'item2', 'item8', 'item9'],
        'user2': ['item4', 'item10', 'item11', 'item5', 'item12'],
        'user3': ['item13', 'item1', 'item14', 'item6', 'item15']
    }
    
    rec_metrics = evaluate_recommendation_system(
        true_items, 
        recommended_items,
        save_path=os.path.join(eval_dir, 'recommendation')
    )
    
    print("\nEvaluation completed. Results saved in 'example_evaluation' directory.")