from flask import Flask, request, jsonify
from models.recommender import FashionRecommender
from app_config import DATA_CONFIG, MODEL_CONFIG
import os

app = Flask(__name__)
recommender = FashionRecommender()

# Load model at startup
@app.before_first_request
def load_model():
    if os.path.exists(DATA_CONFIG['model_path']):
        recommender.load_model()
    else:
        raise Exception("Model not found. Please train the model first.")

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    
    # Validate input
    required_fields = ['season', 'gender']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    season = data['season']
    gender = data['gender']
    age_group = data.get('age_group', 
                        'Adults-Men' if gender in ['Men', 'Women'] else 'Boys')
    n_recommendations = data.get('n_recommendations', MODEL_CONFIG['num_recommendations'])
    
    # Get recommendations
    recommendations = recommender.get_recommendations(season, gender, age_group, n_recommendations)
    
    if len(recommendations) == 0:
        return jsonify({"error": "No recommendations found for the given criteria"}), 404
    
    # Prepare response
    result = recommendations[[
        'id', 'name', 'category', 'subcategory', 
        'article_type', 'color', 'price', 'image_path'
    ]].to_dict('records')
    
    return jsonify({
        "count": len(result),
        "recommendations": result
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)