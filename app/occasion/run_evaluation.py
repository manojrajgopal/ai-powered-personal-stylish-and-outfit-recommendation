from evaluation import OutfitEvaluator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import json
import matplotlib.pyplot as plt

def main():
    evaluator = OutfitEvaluator()
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    all_results = {}
    
    for name, model in models.items():
        try:
            model.fit(evaluator.X_train, evaluator.y_train)
            all_results[name] = evaluator.run_all_tests(model, name)
        except Exception as e:
            print(f"Failed to evaluate {name}: {str(e)}")
            all_results[name] = {'error': str(e)}
    
    # Save results
    with open('evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate comparison plots
    plot_metric_comparison(all_results)

def plot_metric_comparison(results):
    """Plot comparison across all models"""
    metrics = [
        ('classification', 'accuracy'),
        ('classification', 'f1'),
        ('regression', 'r2'),
        ('ranking', 'ndcg@10')
    ]
    
    plt.figure(figsize=(15, 10))
    
    for i, (category, metric) in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = []
        labels = []
        for model_name, model_results in results.items():
            if category in model_results and metric in model_results[category]:
                values.append(model_results[category][metric])
                labels.append(model_name)
        
        if values:  # Only plot if we have values
            plt.bar(labels, values)
            plt.title(f'{metric.upper()} Comparison')
            plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('evaluation_plots/metric_comparison.png')
    plt.close()

if __name__ == '__main__':
    main()