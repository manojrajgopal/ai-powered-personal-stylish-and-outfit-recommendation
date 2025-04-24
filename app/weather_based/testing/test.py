# testing/test.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.recommender import FashionRecommender
# In ALL files (train.py, test.py, recommender.py, etc.)
from app_config import DATA_CONFIG, MODEL_CONFIG

def test_recommendations():
    # Load model with new format
    recommender = FashionRecommender()
    try:
        recommender.load_model()
        
        # Test cases
        test_cases = [
            {'season': 'Summer', 'gender': 'Men', 'age_group': 'Adults-Men'},
            {'season': 'Winter', 'gender': 'Women', 'age_group': 'Adults-Women'},
        ]
        
        for case in test_cases:
            try:
                recommendations = recommender.get_recommendations(
                    case['season'],
                    case['gender'],
                    case['age_group']
                )
                
                if len(recommendations) > 0:
                    print(recommendations[['id', 'name', 'category', 'price']].head())
                else:
                    print("No recommendations found for this criteria")
                    
            except Exception as e:
                print(f"Error generating recommendations: {str(e)}")
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

if __name__ == "__main__":
    test_recommendations()