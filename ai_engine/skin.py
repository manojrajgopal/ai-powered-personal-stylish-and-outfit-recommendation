from age_gender_skinTone import process_fashion_recommendation

# Path to an image for testing
image_path = input("Image path: ")

# Call the function and get results
detected_tone_hex, recommended_colors, gender_category, detected_age, outfits = process_fashion_recommendation(image_path)

# Print the results
print("Detected Skin Tone:", detected_tone_hex)
print("Recommended Colors:", recommended_colors)
print("Gender Category:", gender_category)
print("Detected Age:", detected_age)
print("Generated Outfits:")
for outfit in outfits:
    print(outfit)
