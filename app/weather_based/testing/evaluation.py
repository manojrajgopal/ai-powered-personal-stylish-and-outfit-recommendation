import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_curve, auc, precision_recall_curve, 
    average_precision_score, confusion_matrix
)
from sklearn.manifold import TSNE
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.recommender import FashionRecommender
from app_config import DATA_CONFIG, MODEL_CONFIG

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300

class FashionEvaluator:
    def __init__(self):
        self.recommender = FashionRecommender()
        self.recommender.load_model()
        self.df = self.recommender.df.copy()
        self._clean_data()
        self.output_dir = "testing/evaluation_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def _clean_data(self):
        """Ensure consistent data types and handle missing values"""
        # String columns
        str_cols = ['season', 'gender', 'age_group', 'category', 'subcategory']
        for col in str_cols:
            if col in self.df.columns:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .replace(['nan', 'none', ''], 'unknown')
                )

        # Numeric columns
        if 'price' in self.df.columns:
            self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce').fillna(0)
            self.df['price_bin'] = pd.qcut(self.df['price'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

        # Text columns
        if 'description' in self.df.columns:
            self.df['description'] = self.df['description'].astype(str).str.strip()
            self.df['desc_length'] = self.df['description'].apply(lambda x: len(x.split()))

        # Image features
        if 'image_features' in self.df.columns:
            self.df['image_features'] = self.df['image_features'].apply(
                lambda x: x if isinstance(x, np.ndarray) 
                else np.zeros(MODEL_CONFIG['feature_dim']))

        # Add popularity indicator (mock - replace with real data)
        self.df['popularity'] = np.random.randint(1, 100, size=len(self.df))

    def _save_plot(self, fig, filename):
        """Save plot to file with standardized naming"""
        path = os.path.join(self.output_dir, f"{self.current_date}_{filename}")
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    # ================== CORE METRICS ==================
    def evaluate_recommendation_quality(self, sample_size=1000):
        """Evaluate precision, recall, F1 across different k values"""
    
        eval_df = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        results = []
        k_values = [5, 10, 15, 20]
    
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df),
                        desc="Evaluating recommendations"):
            try:
                recommendations = self.recommender.get_recommendations(
                    row['season'],
                    row['gender'],
                    row['age_group'],
                    n_recommendations=max(k_values)
                )
            
                if len(recommendations) == 0:
                    continue
                
                # Get relevant items (same category, excluding query)
                relevant_items = self.df[
                    (self.df['category'] == row['category']) & 
                    (self.df['id'] != row['id'])
                ]['id'].tolist()
            
                # Calculate metrics with explicit type conversion
                recommended_ids = recommendations['id'].tolist()
                ground_truth = np.array([1 if id in relevant_items else 0 for id in recommended_ids], dtype=np.float32)
            
                for k in k_values:
                    if len(ground_truth) >= k:
                        # Calculate metrics with zero division handling
                        prec = float(precision_score(
                            ground_truth[:k], 
                            np.ones(k, dtype=np.float32), 
                            zero_division=0
                        ))
                        rec = float(recall_score(
                            ground_truth[:k],
                            np.ones(k, dtype=np.float32),
                            zero_division=0
                        ))
                        f1 = float(f1_score(
                            ground_truth[:k],
                            np.ones(k, dtype=np.float32),
                            zero_division=0
                        ))
                    
                        results.append({
                            'k': int(k),
                            'precision': prec,
                            'recall': rec,
                            'f1': f1,
                            'category': str(row['category'])  # Ensure string type
                        })
                    
            except Exception as e:
                print(f"Skipping item {row['id']} due to error: {str(e)}")
                continue
    
        # Create DataFrame with explicit types
        metrics_df = pd.DataFrame({
            'k': [r['k'] for r in results],
            'precision': [r['precision'] for r in results],
            'recall': [r['recall'] for r in results],
            'f1': [r['f1'] for r in results]
        })
    
        # Convert to numeric and handle potential infinities
        numeric_cols = ['precision', 'recall', 'f1']
        metrics_df[numeric_cols] = metrics_df[numeric_cols].apply(
            lambda x: pd.to_numeric(x, errors='coerce')
        ).replace([np.inf, -np.inf], np.nan).fillna(0)
    
        # Plotting (fixed version)
        fig, ax = plt.subplots(figsize=(10, 6))
        for metric in numeric_cols:
            sns.lineplot(
                data=metrics_df,
                x='k',
                y=metric,
                label=metric.capitalize(),
                ax=ax
            )
        ax.set_title('Recommendation Quality vs Number of Recommendations')
        ax.set_xlabel('Number of Recommendations (k)')
        ax.set_ylabel('Score')
        ax.legend()
        self._save_plot(fig, 'quality_vs_k.png')
        plt.show()
    
        # Return aggregated results with proper numeric types
        return metrics_df.groupby('k').mean(numeric_only=True)[numeric_cols]

    # ================== DIVERSITY METRICS ==================
    def evaluate_diversity(self, sample_size=500):
        """Evaluate recommendation diversity"""
        
        eval_df = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        diversity_results = []
        
        for _, row in eval_df.iterrows():
            try:
                recs = self.recommender.get_recommendations(
                    row['season'],
                    row['gender'],
                    row['age_group'],
                    n_recommendations=20
                )
                
                if len(recs) < 2:
                    continue
                    
                # Category coverage
                unique_cats = recs['category'].nunique()
                coverage = unique_cats / self.df['category'].nunique()
                
                # Intra-list similarity
                similarities = []
                rec_items = recs.merge(self.df, on='id', how='left')
                
                for i in range(len(rec_items)):
                    for j in range(i+1, len(rec_items)):
                        sim = 1 - cosine(
                            rec_items.iloc[i]['image_features'],
                            rec_items.iloc[j]['image_features']
                        )
                        similarities.append(sim)
                
                diversity_results.append({
                    'coverage': coverage,
                    'avg_similarity': np.mean(similarities) if similarities else 0,
                    'query_category': row['category']
                })
                
            except Exception as e:
                continue
        
        div_df = pd.DataFrame(diversity_results)
        
        # Plot diversity metrics
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Category coverage
        sns.boxplot(
            data=div_df,
            x='query_category',
            y='coverage',
            ax=axes[0]
        )
        axes[0].set_title('Category Coverage by Query Category')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Intra-list similarity
        sns.boxplot(
            data=div_df,
            x='query_category',
            y='avg_similarity',
            ax=axes[1]
        )
        axes[1].set_title('Intra-List Similarity by Query Category')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.suptitle('Recommendation Diversity Metrics')
        plt.tight_layout()
        self._save_plot(fig, 'diversity_metrics.png')
        plt.show()
        
        return div_df

    # ================== COVERAGE ANALYSIS ==================
    def evaluate_catalog_coverage(self, k_values=[10, 20, 50]):
        """Evaluate what percentage of catalog is being recommended"""
    
        # Get all unique item IDs from the catalog
        all_items = set(self.df['id'].unique())
        recommended_items = set()
        sample_size = min(1000, len(self.df))
    
        # Initialize coverage results
        coverage_results = []
    
        for k in k_values:
            k_items = set()
            for _, row in self.df.sample(sample_size, random_state=42).iterrows():
                try:
                    recs = self.recommender.get_recommendations(
                        row['season'],
                        row['gender'],
                        row['age_group'],
                        n_recommendations=k
                    )
                    # Safely get IDs from recommendations
                    if not recs.empty:
                        if 'id' in recs.columns:
                            k_items.update(recs['id'].astype(str).tolist())
                        elif 'item_id' in recs.columns:  # Try common alternative column names
                            k_items.update(recs['item_id'].astype(str).tolist())
                except Exception as e:
                    print(f"Skipping recommendation for item {row['id']} due to error: {str(e)}")
                    continue
        
            coverage = len(k_items) / len(all_items) if all_items else 0
            coverage_results.append({'k': k, 'coverage': coverage})
    
    # Create and plot coverage results
        cov_df = pd.DataFrame(coverage_results)
    
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=cov_df, x='k', y='coverage', ax=ax)
        ax.set_title('Catalog Coverage by Recommendation Depth')
        ax.set_xlabel('Number of Recommendations (k)')
        ax.set_ylabel('Percentage of Catalog Recommended')
        ax.set_ylim(0, 1)
        self._save_plot(fig, 'catalog_coverage.png')
        plt.show()
    
        return cov_df

    # ================== BUSINESS METRICS ==================
    def evaluate_business_metrics(self, sample_size=500):
        """Evaluate price and popularity distributions"""
    
        # Get recommendations
        eval_df = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        recommendations = []
    
        for _, row in eval_df.iterrows():
            try:
                recs = self.recommender.get_recommendations(
                    row['season'],
                    row['gender'],
                    row['age_group'],
                    n_recommendations=20
                )
                if not recs.empty and 'id' in recs.columns:
                    recommendations.extend(recs[['id']].to_dict('records'))
            except Exception as e:
                print(f"Skipping recommendations for item {row.get('id', 'unknown')} due to error: {str(e)}")
                continue
    
        if not recommendations:
            print("No recommendations generated - skipping business metrics")
            return None
    
        rec_df = pd.DataFrame(recommendations).drop_duplicates('id')
    
        # Check if we have data to merge
        if rec_df.empty or 'id' not in self.df.columns:
            print("No valid recommendation data or missing ID column - skipping business metrics")
            return None
    
        # Ensure we don't have duplicate columns before merging
        merge_columns = ['id']
        if 'price' in self.df.columns:
            merge_columns.append('price')
        if 'popularity' in self.df.columns:
            merge_columns.append('popularity')
        if 'price_bin' in self.df.columns:
            merge_columns.append('price_bin')
    
        try:
            # Perform the merge with explicit column selection
            merged_df = rec_df.merge(
                self.df[merge_columns],
                on='id',
                how='left'
            ).drop_duplicates('id').dropna(subset=['id'])
        except Exception as e:
            print(f"Failed to merge recommendation data: {str(e)}")
            return None
    
        if merged_df.empty:
            print("No valid data after merging - skipping business metrics")
            return None
    
        results = {}
    
        # Price distribution (if available)
        if 'price' in merged_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.kdeplot(
                    data=self.df[self.df['price'].notna()], 
                    x='price', 
                    label='Full Catalog', 
                    ax=ax
                )
                sns.kdeplot(
                    data=merged_df[merged_df['price'].notna()], 
                    x='price', 
                    label='Recommended Items', 
                    ax=ax
                )
                ax.set_title('Price Distribution: Recommended vs Full Catalog')
                ax.set_xlabel('Price (USD)')
                ax.legend()
                self._save_plot(fig, 'price_distribution.png')
                plt.show()
            
                results['avg_price'] = merged_df['price'].mean()
                results['price_ratio'] = merged_df['price'].mean() / self.df['price'].mean() if self.df['price'].mean() > 0 else np.nan
            except Exception as e:
                print(f"Could not generate price distribution: {str(e)}")
    
        # Popularity distribution (if available)
        if 'popularity' in merged_df.columns:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.kdeplot(
                    data=self.df[self.df['popularity'].notna()], 
                    x='popularity', 
                    label='Full Catalog', 
                    ax=ax
                )
                sns.kdeplot(
                    data=merged_df[merged_df['popularity'].notna()], 
                    x='popularity', 
                    label='Recommended Items', 
                    ax=ax
                )
                ax.set_title('Popularity Distribution: Recommended vs Full Catalog')
                ax.set_xlabel('Popularity Score')
                ax.legend()
                self._save_plot(fig, 'popularity_distribution.png')
                plt.show()
            
                results['avg_popularity'] = merged_df['popularity'].mean()
                results['popularity_ratio'] = merged_df['popularity'].mean() / self.df['popularity'].mean() if self.df['popularity'].mean() > 0 else np.nan
            except Exception as e:
                print(f"Could not generate popularity distribution: {str(e)}")
    
        # Price quintile analysis (if available)
        if 'price_bin' in merged_df.columns:
            try:
                price_dist = merged_df['price_bin'].value_counts(normalize=True).sort_index()
            
                fig, ax = plt.subplots(figsize=(10, 6))
                price_dist.plot(kind='bar', ax=ax)
                ax.set_title('Recommended Items by Price Quintile')
                ax.set_xlabel('Price Quintile')
                ax.set_ylabel('Percentage of Recommendations')
                self._save_plot(fig, 'price_quintiles.png')
                plt.show()
            
                results['price_q_dist'] = dict(price_dist.items())
            except Exception as e:
                print(f"Could not generate price quintile analysis: {str(e)}")
    
        return results if results else None

    # ================== ADVANCED METRICS ==================
    def evaluate_advanced_metrics(self, sample_size=200):
        """ROC curves, precision-recall curves, and embedding visualization"""
        
        # ROC and Precision-Recall curves
        eval_df = self.df.sample(min(sample_size, len(self.df)), random_state=42)
        y_true = []
        y_scores = []
        
        for _, row in eval_df.iterrows():
            recs = self.recommender.get_recommendations(
                row['season'],
                row['gender'],
                row['age_group'],
                n_recommendations=20
            )
            relevant_items = self.df[
                (self.df['category'] == row['category']) & 
                (self.df['id'] != row['id'])
            ]['id'].tolist()
            
            for _, rec in recs.iterrows():
                y_true.append(1 if rec['id'] in relevant_items else 0)
                y_scores.append(np.random.random())  # Replace with actual relevance scores if available
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend()
        self._save_plot(fig, 'roc_curve.png')
        plt.show()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        self._save_plot(fig, 'precision_recall_curve.png')
        plt.show()
        
        # Embedding visualization (if features exist)
        if 'image_features' in self.df.columns:
            sample = self.df.sample(min(500, len(self.df)))
            features = np.stack(sample['image_features'].values)
            labels = sample['category'].factorize()[0]
            
            tsne = TSNE(n_components=2, random_state=42)
            embeddings = tsne.fit_transform(features)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap='tab20', alpha=0.6)
            ax.set_title('t-SNE Visualization of Item Embeddings')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            plt.colorbar(scatter, ax=ax)
            self._save_plot(fig, 'tsne_embeddings.png')
            plt.show()
        
        return {
            'roc_auc': roc_auc,
            'average_precision': avg_precision
        }

    # ================== COMPREHENSIVE EVALUATION ==================
    def run_full_evaluation(self):
        """Run all evaluations and generate comprehensive report"""
        
        results = {
            'date': self.current_date,
            'num_items': len(self.df),
            'num_categories': self.df['category'].nunique()
        }
        
        # 1. Core metrics
        print("\n1. Evaluating Recommendation Quality Metrics...")
        results['quality_metrics'] = self.evaluate_recommendation_quality()
        print("\nRecommendation Quality Metrics:")
        print(results['quality_metrics'])
        
        # 2. Diversity metrics
        print("\n2. Evaluating Recommendation Diversity...")
        results['diversity_metrics'] = self.evaluate_diversity()
        
        # 3. Coverage analysis
        print("\n3. Evaluating Catalog Coverage...")
        results['coverage_metrics'] = self.evaluate_catalog_coverage()
        
        # 4. Business metrics
        print("\n4. Evaluating Business Metrics...")
        results['business_metrics'] = self.evaluate_business_metrics()
        
        # 5. Advanced metrics
        print("\n5. Evaluating Advanced Metrics...")
        results['advanced_metrics'] = self.evaluate_advanced_metrics()
        
        # Save all results
        results_path = os.path.join(self.output_dir, f"{self.current_date}_full_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print("\nEvaluation Complete!")
        print(f"Results saved to {results_path}")
        return results


if __name__ == "__main__":
    evaluator = FashionEvaluator()
    evaluator.run_full_evaluation()