from fashion_recommender import recommend_fashion

# Example of how to call the function
filters = {
    'gender': 'Men',
    'baseColour': ['Blue', 'Black'],
    'preferredFabrics': ['Cotton'],
    'preferredStyles': ['Shirts'],
    'occasionTypes': ['Casual'],
    'styleGoals': ['Comfort'],
    'bodyType': ['Slim']
}

recommended_products = recommend_fashion(**filters)