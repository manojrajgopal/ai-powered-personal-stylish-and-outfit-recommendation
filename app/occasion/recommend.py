import pandas as pd

class OutfitRecommender:
    def __init__(self):
        try:
            self.metadata = pd.read_pickle('app/occasion/metadata.pkl')
            
            # Ensure we have a proper index
            if not self.metadata.index.is_unique:
                self.metadata = self.metadata.reset_index(drop=True)
            
            # Clean and standardize data
            for col in ['usage', 'gender', 'subCategory', 'articleType']:
                if col in self.metadata.columns:
                    self.metadata[col] = self.metadata[col].astype(str).str.lower().str.strip()
            
            # Ensure price is numeric
            if 'price_usd' in self.metadata.columns:
                self.metadata['price_usd'] = pd.to_numeric(self.metadata['price_usd'], errors='coerce')
            
        except Exception as e:
            print(f"❌ Failed to load data: {str(e)}")
            raise

    def get_occasion_items(self, occasion=None, gender=None):
        """Get items with proper ID preservation"""
        mask = pd.Series(True, index=self.metadata.index)
        
        if occasion:
            exact_mask = self.metadata['usage'] == occasion.lower()
            mask &= exact_mask if exact_mask.any() else self.metadata['usage'].str.contains(occasion.lower(), na=False)
        
        if gender:
            mask &= self.metadata['gender'].str.contains(gender.lower(), na=False)
        
        return self.metadata[mask].copy()
    
    def get_all_items(self):
        """Get all items with proper IDs"""
        return self.metadata.copy()