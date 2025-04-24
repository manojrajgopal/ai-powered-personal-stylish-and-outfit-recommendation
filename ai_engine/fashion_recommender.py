import pandas as pd
import numpy as np
import json
import os

class FashionRecommender:
    def __init__(self, styles_file, images_file):
        # Initialize with file paths for styles and images dataset
        self.df_styles = pd.read_csv(styles_file, on_bad_lines='skip')
        self.df_images = pd.read_csv(images_file, on_bad_lines='skip')

        # Preprocess the data
        self._preprocess_data()

    def _preprocess_data(self):
        # Split 'filename' into 'id' and 'image format', assuming filename format like '12345.jpg'
        self.df_images[['id', 'image_format']] = self.df_images['filename'].str.extract(r'(\d+)\.(\w+)', expand=True)

        # Keep only 'id' and 'link' columns
        self.df_images = self.df_images[['id', 'link']]

        # Ensure both 'id' columns are of the same type (string)
        self.df_styles['id'] = self.df_styles['id'].astype(str)
        self.df_images['id'] = self.df_images['id'].astype(str)

        # Merge the two DataFrames on 'id' column
        self.df_merged = pd.merge(self.df_styles, self.df_images, on='id', how='left')
    
    
    def get_filtered_data(self, filters):
        df_filtered = self.df_merged.copy()

        def apply_conditions(df, conditions):
            """Helper function to apply conditions to the DataFrame."""
            if conditions:
                return df[np.logical_and.reduce(conditions)]
            return df

        def create_conditions(df, filters, keys):
            """Helper function to create conditions based on specified keys."""
            conditions = []
            for key in keys:
                if key in filters and filters[key]:
                    if key == 'gender':
                        df['gender'] = df['gender'].str.lower()
                        conditions.append(df['gender'] == filters[key].lower())

                    elif key == 'baseColour' and isinstance(filters[key], list):
                        df['baseColour'] = df['baseColour'].str.lower()
                        conditions.append(df['baseColour'].isin([color.lower() for color in filters[key]]))

                    elif key == 'preferredFabrics' and isinstance(filters[key], list):
                        conditions.append(df['productDisplayName'].astype(str).apply(
                            lambda x: any(fabric.lower() in x.lower() for fabric in filters[key])))

                    elif key == 'preferredStyles' and isinstance(filters[key], list):
                        conditions.append(df[['usage', 'productDisplayName', 'articleType']].astype(str).apply(
                            lambda row: any(style.lower() in ' '.join(row).lower() for style in filters[key]), axis=1))

                    elif key == 'occasionTypes' and isinstance(filters[key], list):
                        conditions.append(df[['usage', 'productDisplayName']].astype(str).apply(
                            lambda row: any(occasion.lower() in ' '.join(row).lower() for occasion in filters[key]), axis=1))

                    elif key == 'styleGoals' and isinstance(filters[key], list):
                        conditions.append(df[['usage', 'productDisplayName']].astype(str).apply(
                            lambda row: any(goal.lower() in ' '.join(row).lower() for goal in filters[key]), axis=1))

                    elif key == 'bodyType':
                        print(f"Warning: Unsupported filter applied for {key}")  # BodyType not implemented
            return conditions

        # Stage 1: Apply all user-specified filters (highest priority)
        stage1_conditions = create_conditions(df_filtered, filters, filters.keys())
        stage1_results = apply_conditions(df_filtered, stage1_conditions)

        if not stage1_results.empty:
            df_filtered = stage1_results
        else:
            # Stage 2: Relax filters - search for gender, baseColour, and preferredStyles
            stage2_keys = ['gender', 'baseColour', 'preferredStyles']
            stage2_conditions = create_conditions(self.df_merged, filters, stage2_keys)
            stage2_results = apply_conditions(self.df_merged, stage2_conditions)

            if not stage2_results.empty:
                df_filtered = stage2_results
            else:
                # Stage 3: Relax further - search for gender and baseColour only
                stage3_keys = ['gender', 'baseColour']
                stage3_conditions = create_conditions(self.df_merged, filters, stage3_keys)
                stage3_results = apply_conditions(self.df_merged, stage3_conditions)

                if not stage3_results.empty:
                    df_filtered = stage3_results
                else:
                    # Stage 4: Fallback - search for gender only
                    stage4_keys = ['gender']
                    stage4_conditions = create_conditions(self.df_merged, filters, stage4_keys)
                    stage4_results = apply_conditions(self.df_merged, stage4_conditions)

                    if not stage4_results.empty:
                        df_filtered = stage4_results
                    else:
                        # Stage 5: Final fallback - return random outfits
                        print("No matching products found. Returning random outfits.")
                        df_filtered = self.df_merged.sample(n=min(10, len(self.df_merged)), random_state=42)

        return df_filtered


    def create_category_dict(self, filtered_data):
        # Define the category order
        category_order = ["Apparel", "Accessories", "Footwear", "Personal Care", "Free Items", "Sporting Goods", "Home"]
        category_dict = {category: [] for category in category_order}

        for _, row in filtered_data.iterrows():
            category = row['masterCategory']
            if category in category_dict:
                product_details = {
                    'articleType': row['articleType'],
                    'productDisplayName': row['productDisplayName'],
                    'imageLink': row['link'],
                    'price': row['price_usd'],
                    'price_del': row['discounted_price_usd']
                }
                category_dict[category].append(product_details)
        return category_dict

    def display_products(self, category_dict):
        count = 0
        # Display products for each category in predefined order
        for category in ["Apparel", "Accessories", "Footwear", "Personal Care", "Free Items", "Sporting Goods", "Home"]:
            if category in category_dict and category_dict[category]:
                print(f"\nCategory: {category}")
                for product in category_dict[category]:  # Limit to first 5 products
                    print(f"Article Type: {product['articleType']}")
                    print(f"Product Display Name: {product['productDisplayName']}")
                    print(f"Image Link: {product['imageLink']}")
                    print("-" * 50)  # Separator for readability
                    count += 1
                print(f"Displayed {len(category_dict[category])} products for category: {category}")
        print(f"Total products displayed: {count}")

# Main execution
def recommend_fashion(gender=None, baseColour=None, preferredFabrics=None, preferredStyles=None,
                      occasionTypes=None, styleGoals=None, bodyType=None):
    # Ensure filters are passed properly, ignore empty lists
    filters = {
        'gender': gender if gender else None,
        'baseColour': baseColour if baseColour and baseColour != [''] else None,
        'preferredFabrics': preferredFabrics if preferredFabrics and preferredFabrics != [''] else None,
        'preferredStyles': preferredStyles if preferredStyles and preferredStyles != [''] else None,
        'occasionTypes': occasionTypes if occasionTypes and occasionTypes != [''] else None,
        'styleGoals': styleGoals if styleGoals and styleGoals != [''] else None,
        'bodyType': bodyType if bodyType else None
    }

    # Initialize FashionRecommender with the file paths for styles and images
    recommender = FashionRecommender('data/fashion-dataset/styles.csv', 'data/fashion-dataset/images.csv')

    # Get filtered data based on the user input
    filtered_data = recommender.get_filtered_data(filters)

    # Create a dictionary with products grouped by category
    category_dict = recommender.create_category_dict(filtered_data)

    # Display products
    recommender.display_products(category_dict)
    return category_dict
