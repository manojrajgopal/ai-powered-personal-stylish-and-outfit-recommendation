from app.occasion.outfit_builder import OutfitBuilder

ob = OutfitBuilder()

def recommend(occasion, gender, top_items):
    try:
        
        if not occasion:
            return print("error : Please specify an occasion")
            
        result = ob.build_outfit(occasion, gender or None, top_items)
        
        if 'error' in result:
            return print("Recommandation error:", result['error'])
            
        return result
    
    except Exception as e:
        return print({"error": "Internal server error"})
 