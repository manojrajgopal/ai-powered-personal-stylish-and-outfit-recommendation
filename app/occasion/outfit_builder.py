from app.occasion.recommend import OutfitRecommender
import random
import re
from collections import defaultdict

class OutfitBuilder:
    def __init__(self):
        self.recommender = OutfitRecommender()
        
    def build_outfit(self, occasion, gender, top_items):
        """Generate complete outfits with proper unique IDs"""
        try:
            # Get items matching the occasion and gender
            occasion_items = self.recommender.get_occasion_items(occasion, gender)
            
            # If no items found, try more flexible matching
            if len(occasion_items) < 10:
                occasion_items = self.recommender.get_occasion_items(occasion, None)
                if len(occasion_items) < 10:
                    occasion_items = self.recommender.get_all_items()
            
            if occasion_items.empty:
                return {"error": "No fashion items available in the system"}
            
            # Pre-categorize all items
            categorized = self._categorize_items(occasion_items)
            
            outfits = []
            attempts = 0
            max_attempts = 20
            
            while len(outfits) < top_items and attempts < max_attempts:
                attempts += 1
                outfit = {}
                
                # Build outfit in logical order
                if categorized['topwear']:
                    outfit['topwear'] = self._create_item_dict(random.choice(categorized['topwear']))
                
                if categorized['bottomwear'] and 'topwear' in outfit:
                    outfit['bottomwear'] = self._create_item_dict(random.choice(categorized['bottomwear']))
                
                if categorized['footwear'] and 'bottomwear' in outfit:
                    outfit['footwear'] = self._create_item_dict(random.choice(categorized['footwear']))
                
                if categorized['accessories'] and 'footwear' in outfit:
                    outfit['accessories'] = self._create_item_dict(random.choice(categorized['accessories']))
                
                # Only add complete outfits (at least top+bottom)
                if len(outfit) >= 2:
                    outfits.append(outfit)
            
            return {"outfits": outfits} if outfits else {"error": "Couldn't create complete outfits with current filters"}
            
        except Exception as e:
            return {"error": f"System error: {str(e)}"}
    
    def _categorize_items(self, items):
        """Pre-categorize all items for faster access"""
        categorized = defaultdict(list)
        for _, item in items.iterrows():
            category = self._determine_category(item)
            if category:
                categorized[category].append(item)
        return categorized
    
    def _determine_category(self, item):
        """Determine the best category for an item"""
        category_map = {
            'topwear': ['shirt', 't-shirt', 'top', 'blouse', 'polo'],
            'bottomwear': ['trousers', 'pants', 'jeans', 'skirt'],
            'footwear': ['shoes', 'footwear', 'sneakers', 'sandals'],
            'accessories': ['accessory', 'bag', 'hat', 'tie', 'watch']
        }
        
        # Check both subCategory and articleType
        for field in ['subCategory', 'articleType']:
            if field in item:
                item_type = str(item[field]).lower()
                for category, keywords in category_map.items():
                    if any(keyword in item_type for keyword in keywords):
                        return category
        return None
    
    def _create_item_dict(self, item):
        """Create consistent item dictionary with proper unique ID"""
        return {
            'id': self._extract_unique_id(item),
            'name': item.get('productDisplayName', 'Unknown').title(),
            'price': round(float(item.get('price_usd', 0)), 2)
        }
    
    def _extract_unique_id(self, item):
        """Extract the true unique ID from the item"""
        # First try to use the pandas index as ID
        if hasattr(item, 'name') and str(item.name).isdigit():
            return str(item.name)
        
        # Then try common ID fields
        for field in ['id', 'productId', 'itemId', 'sku']:
            if field in item and str(item[field]).strip():
                return str(item[field]).strip()
        
        # Fallback to link extraction
        if 'link' in item:
            numbers = re.findall(r'\d+', str(item['link']))
            if numbers:
                return numbers[-1][:8]  # Use last number found, max 8 digits
        
        # Final fallback - random 5-digit ID
        return str(random.randint(10000, 99999))