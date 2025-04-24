import spacy
import pandas as pd
import os
import glob

# Load NLP model (English)
nlp = spacy.load("en_core_web_sm")

# Structured Attributes
color_keywords = ["Navy Blue", "Blue", "Silver", "Black", "Grey", "Green", "Purple", "White", "Beige", "Brown", "Red", "Yellow", "Gold", "Pink"]
category_keywords = ["Apparel", "Accessories", "Footwear", "Personal Care", "Sporting Goods", "Home"]
sub_category_keywords = ["Topwear", "Bottomwear", "Watches", "Shoes", "Belts", "Bags", "Innerwear", "Saree", "Jewellery", "Wallets", "Makeup", "Skin Care", "Fragrance"]
article_keywords = ["Shirts", "Jeans", "Tshirts", "Casual Shoes", "Handbags", "Kurtas", "Perfume", "Sunglasses", "Dresses", "Jackets", "Sweaters", "Leggings", "Mascara", "Earrings"]
season_keywords = ["Fall", "Summer", "Winter", "Spring"]
usage_keywords = ["Casual", "Ethnic", "Formal", "Sports", "Smart Casual", "Travel", "Party", "Home"]

def load_fashion_data(parquet_folder, sample_size=5000):
    """Loads fashion data from multiple Parquet files in a memory-efficient manner."""
    all_files = glob.glob(os.path.join(parquet_folder, "*.parquet"))
    selected_files = all_files[:min(len(all_files), sample_size // 1000)]
    
    data_frames = []
    for file in selected_files:
        df = pd.read_parquet(file, columns=["baseColour", "masterCategory", "subCategory", "articleType", "season", "usage", "productDisplayName", "discountedPrice", "description", "imageURL", "landingPageUrl"])
        data_frames.append(df.sample(frac=0.1, random_state=42))
    
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def extract_keywords(query):
    """Extracts relevant attributes from user query."""
    doc = nlp(query.lower())
    extracted_data = {
        "colors": [token.text for token in doc if token.text in [c.lower() for c in color_keywords]],
        "categories": [token.text for token in doc if token.text in [c.lower() for c in category_keywords]],
        "sub_categories": [token.text for token in doc if token.text in [c.lower() for c in sub_category_keywords]],
        "articles": [token.text for token in doc if token.text in [c.lower() for c in article_keywords]],
        "seasons": [token.text for token in doc if token.text in [c.lower() for c in season_keywords]],
        "usage": [token.text for token in doc if token.text in [c.lower() for c in usage_keywords]],
    }
    return {key: list(set(value)) for key, value in extracted_data.items() if value}

def filter_products(fashion_data, extracted_keywords):
    """Filters products based on extracted keywords."""
    if fashion_data.empty:
        return fashion_data
    
    query = pd.Series(True, index=fashion_data.index)
    for key, column in zip(["colors", "categories", "sub_categories", "articles", "seasons", "usage"],
                            ["baseColour", "masterCategory", "subCategory", "articleType", "season", "usage"]):
        if extracted_keywords.get(key):
            query &= fashion_data[column].str.lower().isin(extracted_keywords[key])
    
    return fashion_data[query]

def recommend_outfits(user_query):
    """Processes user query and recommends outfits."""
    parquet_folder = r"data/fashion-dataset/parquet/"
    fashion_data = load_fashion_data(parquet_folder)
    
    extracted_keywords = extract_keywords(user_query)
    
    if not extracted_keywords:
        print("\nNo valid attributes found in query. Please provide more details.")
        return pd.DataFrame()
    
    recommended_outfits = filter_products(fashion_data, extracted_keywords)
    
    if not recommended_outfits.empty:
        print("\nRecommended Outfits:")
        for _, outfit in recommended_outfits.iterrows():
            print(f"- {outfit['productDisplayName']} ({outfit['baseColour']})\n  Price: {outfit['discountedPrice']}\n  Description: {outfit['description']}\n  Image: {outfit['imageURL']}\n  Link: {outfit['landingPageUrl']}\n")
    else:
        print("\nNo matching outfits found.")
    
    return recommended_outfits

user_query = 'I need a blue shirts'
recommend_outfits(user_query)