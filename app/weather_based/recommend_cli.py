import pandas as pd
from app.weather_based.models.recommender import FashionRecommender

def weather_based_recommend(season, gender):
    try:
        if not season or not gender:
            return []
        
        recommender = FashionRecommender()
        recommender.load_model()

        season = season.strip().capitalize()
        gender = gender.strip().capitalize()
        age_group = f"Adults-{gender}" if gender in ["Men", "Women"] else gender

        recommendations = recommender.get_recommendations(season, gender, age_group)

        if len(recommendations) > 0:
            # Convert DataFrame to a list of dictionaries
            recommendations_list = recommendations[['id', 'name', 'price']].to_dict('records')
            return recommendations_list
        else:
            return []
    
    except Exception as e:
        print(f"Error in weather-based recommendation: {str(e)}")
        return []