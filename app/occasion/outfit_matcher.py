import numpy as np
from sklearn.neighbors import NearestNeighbors

class OutfitRecommender:
    def __init__(self, metadata, feature_matrix):
        self.metadata = metadata
        self.feature_matrix = feature_matrix
        self.knn = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.knn.fit(feature_matrix)
        
    def _filter_by_category(self, items, category):
        return items[items['subCategory'].str.lower() == category.lower()]
    
    def recommend_outfit(self, occasion, gender='Men', top_k=5):
        # Get occasion-appropriate items
        occasion_items = self.metadata[
            (self.metadata['usage'].str.lower() == occasion.lower()) &
            (self.metadata['gender'].str.lower() == gender.lower())
        ]
        
        # Split by clothing type
        tops = self._filter_by_category(occasion_items, 'topwear')
        bottoms = self._filter_by_category(occasion_items, 'bottomwear')
        footwear = self._filter_by_category(occasion_items, 'footwear')
        accessories = self._filter_by_category(occasion_items, 'accessories')
        
        recommendations = []
        
        # Recommend complete outfits
        for _ in range(top_k):
            outfit = {}
            
            # Select random top
            if not tops.empty:
                top = tops.sample(1).iloc[0]
                outfit['top'] = top
                
                # Find matching bottoms
                if not bottoms.empty:
                    top_features = self.feature_matrix[top.name].reshape(1, -1)
                    _, indices = self.knn.kneighbors(top_features, n_neighbors=len(bottoms))
                    matching_bottom_idx = indices[0][0]
                    outfit['bottom'] = bottoms.iloc[matching_bottom_idx]
            
            # Add footwear and accessories
            if not footwear.empty:
                outfit['footwear'] = footwear.sample(1).iloc[0]
            if not accessories.empty:
                outfit['accessories'] = accessories.sample(1).iloc[0]
                
            recommendations.append(outfit)
            
        return recommendations