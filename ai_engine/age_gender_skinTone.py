import pandas as pd
import numpy as np
import random
import cv2
import logging
from PIL import Image, ImageColor
import os
from deepface import DeepFace
from markupsafe import Markup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Importing Models and set mean values
face1 = "data/age_gender-dataset/opencv_face_detector.pbtxt"
face2 = "data/age_gender-dataset/opencv_face_detector_uint8.pb"
age1 = "data/age_gender-dataset/age_deploy.prototxt"
age2 = "data/age_gender-dataset/age_net.caffemodel"
gen1 = "data/age_gender-dataset/gender_deploy.prototxt"
gen2 = "data/age_gender-dataset/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Categories
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']

def get_precise_age(agePreds):
    """Convert age prediction probabilities into an accurate numerical age using a weighted average."""
    # Define the midpoints of the given age ranges
    age_midpoints = [1, 5, 10, 17, 28, 40, 50, 70]  # Approximate midpoints for each category
    age_probabilities = agePreds[0]  # Extract prediction probabilities

    # Compute the weighted average
    estimated_age = sum(age_midpoints[i] * age_probabilities[i] for i in range(len(age_list)))
    
    return round(estimated_age, 1)  # Round to one decimal place


def detect_age_gender_opencv(image_path):
    """Detect age and gender using OpenCV's DNN models and estimate an accurate age."""
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logging.error("Error: Unable to load image.")
            return 'Unisex', None

        # Resize image
        image = cv2.resize(image, (720, 640))

        # Load models
        dnn_face = cv2.dnn.readNet(face2, face1)
        dnn_age = cv2.dnn.readNet(age2, age1)
        dnn_gen = cv2.dnn.readNet(gen2, gen1)

        # Create a blob
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], True, False)

        # Detect faces
        dnn_face.setInput(blob)
        detections = dnn_face.forward()
        
        faceBoxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                x1, y1, x2, y2 = (int(detections[0, 0, i, j] * image.shape[j % 2]) for j in range(3, 7))
                faceBoxes.append([x1, y1, x2, y2])

        if not faceBoxes:
            logging.warning("No face detected.")
            return 'Unisex', None

        # Process first detected face
        x1, y1, x2, y2 = faceBoxes[0]
        face = image[max(0, y1-15):min(y2+15, image.shape[0]-1), 
                     max(0, x1-15):min(x2+15, image.shape[1]-1)]

        # Create a blob for face
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        dnn_gen.setInput(blob)
        genderPreds = dnn_gen.forward()
        gender = gender_list[genderPreds[0].argmax()]
        print(f"Detected Gender: {gender}")

        # Age Prediction
        dnn_age.setInput(blob)
        agePreds = dnn_age.forward()

        # Get accurate numerical age
        estimated_age = get_precise_age(agePreds)
        print(f"Estimated Age: {estimated_age}")

        # Determine gender category based on estimated age
        if gender == 'Male':
            gender_category = 'Men' if estimated_age >= 15 else 'Boys'
        else:
            gender_category = 'Women' if estimated_age >= 15 else 'Girls'

        return gender_category, estimated_age

    except Exception as e:
        logging.error(f"Error detecting age and gender with OpenCV: {e}")
        return 'Unisex', None


# ---------------------------- 1. DATASET CLEANING ----------------------------
def load_and_clean_dataset(styles_path):
    """Load and clean the dataset."""
    try:
        data = pd.read_csv(styles_path, on_bad_lines="skip")
        columns_to_keep = [
            "id", "gender", "masterCategory", "subCategory",
            "articleType", "baseColour", "season", "usage"
        ]
        data = data[columns_to_keep]
        data.dropna(subset=["gender", "masterCategory", "baseColour"], inplace=True)
        data["baseColour"] = data["baseColour"].fillna("Unknown")
        data.columns = [col.strip().lower() for col in data.columns]
        data = data[data["gender"].isin(["Men", "Women"])]
        data = data[data["mastercategory"].isin(["Apparel", "Footwear", "Accessories"])]
        data = data[~data["subcategory"].isin(["Innerwear", "Loungewear and Nightwear"])]
        data = data[~data["usage"].isin([None, "Home"])]
        return data
    except Exception as e:
        logging.error(f"Error loading and cleaning dataset: {e}")
        return None

# ---------------------------- 2. SKIN TONE DETECTION ----------------------------
class SkinToneDetector:
    def __init__(self):
        # Expanded skin tone mapping with more colors
        self.skin_tone_to_color_mapping = {
            "#373028": ["Navy Blue", "Black", "Charcoal", "Burgundy", "Maroon", "Olive", "Rust", "Gold", "Cream", "Peach"],
            "#422811": ["Navy Blue", "Brown", "Khaki", "Olive", "Maroon", "Mustard", "Teal", "Tan", "Rust", "Burgundy"],
            "#513B2E": ["Cream", "Beige", "Olive", "Burgundy", "Red", "Orange", "Mustard", "Bronze", "Teal", "Peach"],
            "#6F503C": ["Beige", "Brown", "Green", "Khaki", "Cream", "Peach", "Lime Green", "Olive", "Maroon", "Rust", "Mustard"],
            "#81654F": ["Beige", "Off White", "Sea Green", "Cream", "Lavender", "Mauve", "Burgundy", "Yellow", "Lime Green"],
            "#9D7A54": ["Olive", "Khaki", "Yellow", "Sea Green", "Turquoise Blue", "Coral", "White", "Gold", "Peach"],
            "#AD8B6F": ["Coral", "Mint Green", "Lavender", "Pink", "Light Blue", "Beige", "White", "Gold"],
            "#C4A88B": ["Pastel Pink", "Light Blue", "Mint Green", "Lavender", "White", "Beige", "Gold"],
            "#E0C4A8": ["Pastel Blue", "Pastel Pink", "Lavender", "Mint Green", "White", "Beige", "Gold"],
        }
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def detect_skin_tone(self, image):
        """Detect skin tone from an image."""
        try:
            # Resize image to a manageable size
            resized_image = cv2.resize(image, (200, 200))
            avg_color = resized_image.mean(axis=0).mean(axis=0)
            avg_color_hex = "#{:02x}{:02x}{:02x}".format(int(avg_color[0]), int(avg_color[1]), int(avg_color[2]))
            avg_color_rgb = np.array(ImageColor.getrgb(avg_color_hex))
            closest_tone_hex = min(
                self.skin_tone_to_color_mapping.keys(),
                key=lambda hex_code: np.linalg.norm(avg_color_rgb - np.array(ImageColor.getrgb(hex_code)))
            )
            return closest_tone_hex, self.skin_tone_to_color_mapping.get(closest_tone_hex, [])
        except Exception as e:
            logging.error(f"Error detecting skin tone: {e}")
            return None, []

    def generate(self, image_path):
        """Generate skin tone and recommended colors from an image."""
        try:
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                logging.error("Error: Cannot open image file!")
                return None, []

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            if len(faces) == 0:
                logging.warning("No face detected. Using the entire image for skin tone detection.")
                # If no face is detected, use the entire image
                detected_tone_hex, recommended_colors = self.detect_skin_tone(img)
                return detected_tone_hex, recommended_colors

            # If faces are detected, use the first face
            for (x, y, w, h) in faces:
                face_img = img[y:y + h, x:x + w]
                detected_tone_hex, recommended_colors = self.detect_skin_tone(face_img)
                return detected_tone_hex, recommended_colors

            return None, []
        except Exception as e:
            logging.error(f"Error generating skin tone: {e}")
            return None, []

# ---------------------------- 3. GENERATING OUTFIT COMBINATIONS ----------------------------
class OutfitGenerator:
    def __init__(self, data):
        self.data = data

    def get_complementary(self, color, palette):
        """Return a complementary color if available in the palette."""
        return random.choice([c for c in palette if c != color])

    def get_analogous(self, color, palette):
        """Return an analogous color if available in the palette."""
        return random.choice([c for c in palette if c != color])

    def get_triadic(self, color, palette):
        """Return triadic colors if available in the palette."""
        return random.choice([c for c in palette if c != color])

    def get_neutral(self, palette):
        """Return a neutral color from the palette."""
        neutrals = ["Black", "White", "Beige", "Cream", "Off White", "Grey", "Charcoal"]
        return random.choice([c for c in palette if c in neutrals])

    def generate_outfits(self, recommended_colors, gender_category):
        """Generate outfit combinations with color harmony based on gender and age."""
        try:
            # Filter data based on the detected gender category
            filtered_data = self.data[self.data["gender"] == gender_category]
            filtered_data = filtered_data[filtered_data["basecolour"].isin(recommended_colors)]
        
            top_wear = filtered_data[filtered_data["subcategory"] == "Topwear"]
            bottom_wear = filtered_data[filtered_data["subcategory"] == "Bottomwear"]
            footwear = filtered_data[filtered_data["mastercategory"] == "Footwear"]
            accessories = filtered_data[filtered_data["mastercategory"] == "Accessories"]

            outfit_combinations = []
            for top in top_wear.head(20).itertuples():
                bottom_color = self.get_complementary(top.basecolour, recommended_colors)
                bottomwear_options = bottom_wear[bottom_wear["basecolour"] == bottom_color]
                if not bottomwear_options.empty:
                    bottom = bottomwear_options.sample().iloc[0]
                    footwear_color = random.choice([top.basecolour, bottom_color, self.get_neutral(recommended_colors)])
                    footwear_options = footwear[footwear["basecolour"] == footwear_color]
                    if not footwear_options.empty:
                        foot = footwear_options.sample().iloc[0]
                        accessory_color = random.choice([top.basecolour, bottom_color, footwear_color])
                        accessory_options = accessories[accessories["basecolour"] == accessory_color]
                        if not accessory_options.empty:
                            accessory = accessory_options.sample().iloc[0]
                            outfit_combinations.append({
                                "Topwear": top.id,
                                "Bottomwear": bottom.id,
                                "Footwear": foot.id,
                                "Accessory": accessory.id,
                                "Topwear Color": top.basecolour,
                                "Bottomwear Color": bottom.basecolour,
                                "Footwear Color": foot.basecolour,
                                "Accessory Color": accessory.basecolour
                            })
            return random.sample(outfit_combinations, min(len(outfit_combinations), 100))
        except Exception as e:
            logging.error(f"Error generating outfits: {e}")
            return []

# ---------------------------- 4. DISPLAYING OUTFIT IMAGES ----------------------------
def display_outfits(outfits):
    """Display generated outfits."""
    for i, outfit in enumerate(outfits):
        print(f"\n👗 Outfit {i + 1}")
        print(f"Topwear:  {outfit['Topwear Color']} (ID: {outfit['Topwear']})")
        print(f"Bottomwear: {outfit['Bottomwear Color']} (ID: {outfit['Bottomwear']})")
        print(f"Footwear:  {outfit['Footwear Color']} (ID: {outfit['Footwear']})")
        print(f"Accessory: {outfit['Accessory Color']} (ID: {outfit['Accessory']})")

# ---------------------------- RUNNING THE SYSTEM ----------------------------
def process_fashion_recommendation(image_path):
    data = load_and_clean_dataset("data/fashion-dataset/styles.csv")
    if data is None:
        logging.error("Failed to load dataset.")
        return None, None, None, None, None

    detector = SkinToneDetector()

    # Use OpenCV for gender and age detection
    gender_category, detected_age = detect_age_gender_opencv(image_path)
    detected_tone_hex, recommended_colors = detector.generate(image_path)
    if detected_tone_hex:
        print(f"Detected Skin Tone: {detected_tone_hex}")
        print(f"Recommended Colors: {recommended_colors}")
        print(f"Detected Gender and Age Category: {gender_category, detected_age}")

        outfit_generator = OutfitGenerator(data)
        outfits = outfit_generator.generate_outfits(recommended_colors, gender_category)
        display_outfits(outfits)
        
        if detected_tone_hex == "#373028":
            skin_tone = Markup("Bold, radiant, and powerful—styles that embrace the beauty of <strong>deepest ebony</strong> skin tones. 🌑✨")
            recommend_color = Markup(
                "Command attention with the bold depth of <strong>Navy Blue</strong> and <strong>Black</strong>, "
                "the timeless elegance of <strong>Charcoal</strong> and <strong>Burgundy</strong>, "
                "and the warm richness of <strong>Maroon</strong> and <strong>Olive</strong>. "
                "Add a touch of <strong>Rust</strong>, <strong>Gold</strong>, <strong>Cream</strong>, "
                "and <strong>Peach</strong> to perfect your look—because great style is all about the perfect color harmony! ✨🎨👗"
            )

        elif detected_tone_hex == "#422811":
            skin_tone = Markup("Rich, striking, and elegant—discover outfits that enhance your <strong>very deep mahogany</strong> skin tone. 🔥👗")
            recommend_color = Markup(
                "Embrace effortless style with the classic charm of <strong>Navy Blue</strong> and <strong>Brown</strong>, "
                "the earthy sophistication of <strong>Khaki</strong> and <strong>Olive</strong>, "
                "and the bold richness of <strong>Maroon</strong> and <strong>Mustard</strong>. "
                "Elevate your look with vibrant <strong>Teal</strong>, warm <strong>Tan</strong>, "
                "deep <strong>Rust</strong>, and timeless <strong>Burgundy</strong>—because fashion is all about confidence and the perfect colors! ✨🎨👗"
            )

        elif detected_tone_hex == "#513B2E":
            skin_tone = Markup("<strong>Deep brown</strong>, endlessly chic—fashion that brings out your natural glow. 🌟🖤")
            recommend_color = Markup(
                "Radiate elegance with the soft sophistication of <strong>Cream</strong> and <strong>Beige</strong>, "
                "the natural allure of <strong>Olive</strong> and <strong>Burgundy</strong>, "
                "and the fiery confidence of <strong>Red</strong> and <strong>Orange</strong>. "
                "Accentuate your style with bold <strong>Mustard</strong>, shimmering <strong>Bronze</strong>, "
                "vibrant <strong>Teal</strong>, and the delicate charm of <strong>Peach</strong>—because the right colors make every outfit unforgettable! ✨🎨👗"
            )

        elif detected_tone_hex == "#6F503C":
            skin_tone = Markup("A perfect balance of warmth and richness—styles made to complement <strong>medium brown</strong> skin. 🍫👕")
            recommend_color = Markup(
                "Step into effortless style with the timeless elegance of <strong>Beige</strong> and <strong>Brown</strong>, "
                "the earthy vibrance of <strong>Green</strong> and <strong>Khaki</strong>, "
                "and the soft sophistication of <strong>Cream</strong> and <strong>Peach</strong>. "
                "Add a pop of <strong>Lime Green</strong>, the richness of <strong>Olive</strong> and <strong>Maroon</strong>, "
                "and the bold warmth of <strong>Rust</strong> and <strong>Mustard</strong>—because fashion is all about the perfect blend of colors! ✨🎨👗"
            )

        elif detected_tone_hex == "#81654F":
            skin_tone = Markup("Sun-kissed and stylish—find outfits that enhance your gorgeous <strong>tan</strong> complexion. ☀️👗")
            recommend_color = Markup(
                "Embrace a look of pure sophistication with the timeless elegance of <strong>Beige</strong> and <strong>Off White</strong>, "
                "the refreshing charm of <strong>Sea Green</strong> and <strong>Cream</strong>, "
                "and the soft allure of <strong>Lavender</strong> and <strong>Mauve</strong>. "
                "Elevate your style with the bold richness of <strong>Burgundy</strong>, "
                "the radiant energy of <strong>Yellow</strong>, and the vibrant touch of <strong>Lime Green</strong>—because true fashion is a masterpiece of colors! ✨🎨👗"
            )

        elif detected_tone_hex == "#9D7A54":
            skin_tone = Markup("Subtle, glowing, and effortlessly chic—your <strong>light tan</strong> skin deserves fashion that shines. ✨👚")
            recommend_color = Markup(
                "Unleash your style with the earthy elegance of <strong>Olive</strong> and <strong>Khaki</strong>, "
                "the vibrant energy of <strong>Yellow</strong> and <strong>Sea Green</strong>, "
                "and the refreshing charm of <strong>Turquoise Blue</strong> and <strong>Coral</strong>. "
                "Complete your look with the timeless purity of <strong>White</strong>, "
                "the luxurious shimmer of <strong>Gold</strong>, and the soft warmth of <strong>Peach</strong>—because fashion is all about balance, boldness, and brilliance! ✨🎨👗"
            )

        elif detected_tone_hex == "#AD8B6F":
            skin_tone = Markup("A natural warmth that deserves the perfect wardrobe—flattering styles for <strong>medium fair</strong> skin. 🌸👖")
            recommend_color = Markup(
                "Step into elegance with the lively charm of <strong>Coral</strong> and <strong>Mint Green</strong>, "
                "the dreamy allure of <strong>Lavender</strong> and <strong>Pink</strong>, "
                "and the soothing grace of <strong>Light Blue</strong> and <strong>Beige</strong>. "
                "Accentuate your look with the timeless purity of <strong>White</strong> "
                "and the luxurious shimmer of <strong>Gold</strong>—because true style is a perfect blend of softness and sophistication! ✨🎨👗"
            )

        elif detected_tone_hex == "#C4A88B":
            skin_tone = Markup("Soft, delicate, and timeless—elevate your look with outfits that suit <strong>light fair</strong> skin. 🎀👗")
            recommend_color = Markup(
                "Embrace a look of pure elegance with the soft charm of <strong>Pastel Pink</strong> and <strong>Light Blue</strong>, "
                "the refreshing touch of <strong>Mint Green</strong> and <strong>Lavender</strong>, "
                "and the timeless sophistication of <strong>White</strong> and <strong>Beige</strong>. "
                "Elevate your style with the luxurious shimmer of <strong>Gold</strong>—because fashion is all about grace, balance, and a touch of brilliance! ✨🎨👗"
            )

        elif detected_tone_hex == "#E0C4A8":
            skin_tone = Markup("Classic, elegant, and effortlessly beautiful—styles that highlight <strong>very fair</strong> skin. ❄️👚")
            recommend_color = Markup(
                "Drift into effortless elegance with the dreamy hues of <strong>Pastel Blue</strong> and <strong>Pastel Pink</strong>, "
                "the soft sophistication of <strong>Lavender</strong> and <strong>Mint Green</strong>, "
                "and the timeless purity of <strong>White</strong> and <strong>Beige</strong>. "
                "Add a touch of luxury with the radiant glow of <strong>Gold</strong>—because true style is a masterpiece of delicate harmony! ✨🎨👗"
            )

        else:
            skin_tone = "Oops! AI couldn't detect your skin tone, but great style has no limits—explore, experiment, and embrace colors that make you feel amazing! 🎨✨"
            recommend_color = "Fashion is about confidence and creativity! Mix and match different shades to discover what truly represents you. Every color has its magic—find yours! ✨👗"


        if gender_category == "Men":
            gender_sentence = Markup("AI has detected you as a <strong>man</strong>—refined, confident, and effortlessly stylish, your perfect look is just a recommendation away! 🕶️✨")

        elif gender_category == "Boys":
            gender_sentence = Markup("AI recognizes you as a <strong>boy</strong>—energetic, fun, and full of life, get ready for styles that match your vibe! 🎒🔥")

        elif gender_category == "Girls":
            gender_sentence = Markup("AI detects you as a <strong>girl</strong>—bright, playful, and full of charm, your perfect outfit is waiting! 🎀🌸")

        elif gender_category == "Women":
            gender_sentence = Markup("AI has identified you as a <strong>woman</strong>—graceful, elegant, and trendsetting, your fashion journey starts here! 👗💫")

        else:
            gender_sentence = "Oops! AI couldn't detect the gender, but great style has no boundaries—explore, express, and wear what makes you feel amazing! ✨👗🕶️"

        
        if 0 <= detected_age <= 12:
            age_sentence = Markup(f"You are in the early years of life <strong>{detected_age}</strong>, where comfort and playfulness define your style! 🌟🎈")
        elif 13 <= detected_age <= 19:
            age_sentence = Markup(f"As a teenager <strong>{detected_age}</strong>, your fashion is all about self-expression and bold trends! 🔥👕")
        elif 20 <= detected_age <= 35:
            age_sentence = Markup(f"In your prime years <strong>{detected_age}</strong>, your style blends confidence, elegance, and modern trends! 🕶️✨")
        elif 36 <= detected_age <= 50:
            age_sentence = Markup(f"With experience and grace <strong>{detected_age}</strong>, your fashion reflects sophistication and timeless charm! 💼🎩")
        elif 51 <= detected_age <= 65:
            age_sentence = Markup(f"<strong>{detected_age}</strong>, Your style is a perfect mix of refinement, comfort, and effortless elegance! 👔👜")
        elif 66 <= detected_age <= 100:
            age_sentence = Markup(f"<strong>{detected_age}</strong>, A lifetime of style—your fashion exudes wisdom, grace, and timeless appeal! 👒🌟")
        else:
            age_sentence = "Oops! AI couldn't detect the age, but style has no limits—wear what makes you feel amazing! ✨👗"
        
        return skin_tone, gender_sentence, age_sentence, recommend_color, gender_category, detected_age, outfits or []
    else:
        logging.error("Skin tone detection failed.")
        return None, None, None, None, None