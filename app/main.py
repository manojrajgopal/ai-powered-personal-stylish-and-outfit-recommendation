from flask import Flask, render_template, current_app, request, jsonify, redirect, url_for, session, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from datetime import datetime, date
from werkzeug.utils import secure_filename
from app.finder import FashionFinder
from app.database import DatabaseInitializer
from ai_engine.fashion_recommender import recommend_fashion
from ai_engine.virtual_try_on import run_virtual_try_on
from ai_engine.age_gender_skinTone import process_fashion_recommendation
#from celery_setup import celery
import pymysql
import os
import google.generativeai as ai
import logging
import atexit
import time
import json
import asyncio
from threading import Thread
import requests
from app.weather_based.recommend_cli import weather_based_recommend
from app.image_based.cli_recommender import rec
from app.globals import season as global_season
from app.occasion.app import recommend

# Directory to save the images
PROFILE_PIC_FOLDER = 'app/static/uploads/profile/'
WARDROBE_IMG_FOLDER = 'app/static/uploads/wardrobe/'
VIRTUAL_TRY_ON_IMG_FOLDER = os.path.join('app/static/uploads/virtual_try_on/')

os.makedirs(PROFILE_PIC_FOLDER, exist_ok=True)
os.makedirs(WARDROBE_IMG_FOLDER, exist_ok=True)

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_LOG_SEVERITY_LEVEL"] = "ERROR"
logging.getLogger('google.generativeai').setLevel(logging.CRITICAL)
logging.getLogger('grpc').setLevel(logging.CRITICAL)

API_KEY = "" # Replace with your API key

if API_KEY is None:
    raise ValueError("API_KEY environment variable is not set!")

ai.configure(api_key=API_KEY)

# Create a new model
model = ai.GenerativeModel("gemini-1.5-pro-latest")
chat = model.start_chat()

pymysql.install_as_MySQLdb()  # This will make PyMySQL work as MySQLdb

app = Flask(__name__)
app.config['SECRET_KEY'] = 'manojrajgopal'

# Configuring the database connection
db_initializer = DatabaseInitializer()
db_initializer.initialize_database()

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/fashion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Dashboard:
    def __init__(self, app):
        self.app = app
        self.register_routes()

    def register_routes(self):
        # Define the route for the main dashboard page
        self.app.add_url_rule('/', view_func=self.main, methods=['GET', 'POST'])

    def main(self):
        if request.method == "POST":
            # Handle form submission for email subscription
            email = request.form.get("email")
            if email:
                print(f"New subscription: {email}")
        # Render the main dashboard page
        return render_template('index.html', title='Outfit Recommender')

class Login:
    def __init__(self, app, db):
        self.app = app
        self.db = db
        self.register_routes()

    def register_routes(self):
        # Define routes for login, signup, and logout
        self.app.add_url_rule('/login', methods=['GET', 'POST'], view_func=self.login)
        self.app.add_url_rule('/signup', methods=['POST'], view_func=self.signup)
        self.app.add_url_rule('/logout', methods=['GET'], view_func=self.logout)
        self.app.add_url_rule('/get_location', methods=['POST'], view_func=self.get_location)  # ✅ Add this

    def get_location(self):
        data = request.get_json()
        lat, lon = data.get('latitude'), data.get('longitude')

        if lat is None or lon is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400

        try:
            # Fetch city using OpenWeatherMap Geocoding API
            OPENWEATHERMAP_API_KEY = "" # Replace with your API key
            geo_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHERMAP_API_KEY}"
            geo_response = requests.get(geo_url).json()

            city = geo_response[0]["name"] if geo_response and len(geo_response) > 0 else "Unknown City"

            # Determine the season
            month = datetime.now().month
            if lat >= 0:  # Northern Hemisphere
                season = ["Winter", "Spring", "Summer", "Fall"][(month % 12) // 3]
            else:  # Southern Hemisphere
                season = ["Summer", "Fall", "Winter", "Spring"][(month % 12) // 3]

            from app import globals
            globals.season = season
            return jsonify({"city": city, "season": season})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_location_manual(self, lat, lon):
        try:
            # Fetch city using OpenWeatherMap Geocoding API
            OPENWEATHERMAP_API_KEY = "78542c411a2b9031ae9cb9bef906dc2f"
            geo_url = f"http://api.openweathermap.org/geo/1.0/reverse?lat={lat}&lon={lon}&limit=1&appid={OPENWEATHERMAP_API_KEY}"
            geo_response = requests.get(geo_url).json()

            city = geo_response[0]["name"] if geo_response and len(geo_response) > 0 else "Unknown City"

            # Determine the season
            month = datetime.now().month
            if lat >= 0:  # Northern Hemisphere
                season = ["Winter", "Spring", "Summer", "Fall"][(month % 12) // 3]
            else:  # Southern Hemisphere
                season = ["Summer", "Fall", "Winter", "Spring"][(month % 12) // 3]

            from app import globals
            globals.season = season
            return {"city": city, "season": season}

        except Exception as e:
            return {"city": "Unknown City", "season": "Unknown Season", "error": str(e)}

    def login(self):
        if request.method == 'POST':
            # Get username and password from the login form
            username = request.form['username'].lower()
            password = request.form['password']

            # Safely execute the query to check login credentials
            query = text('SELECT username, email, phone, name, password FROM login WHERE username=:username AND password=:password')
            result = self.db.session.execute(query, {'username': username, 'password': password})
            user = result.fetchone()  # Fetch the first matching row
            lat = request.form.get('latitude')
            lon = request.form.get('longitude')

            # Fetch city and season using get_location
            city = request.form.get('city', 'Unknown City')
            season = request.form.get('season', 'Unknown Season')

            if lat and lon:
                location_data = self.get_location_manual(float(lat), float(lon))  # Call a helper function
                city = location_data.get("city", "Unknown City")
                season = location_data.get("season", "Unknown Season")

            if user:
                # Store user information in the session
                session['user'] = {'username': user[0], 'email': user[1], 'phone': user[2], 'name': user[3], 'password': password, 'city':city}  # username = user[0], email = user[1], phone = user[2]
            
                # Now check if the user exists in the user_information table
                query_user_info = text('SELECT username FROM user_information WHERE username=:username')
                result_user_info = self.db.session.execute(query_user_info, {'username': username})
                user_info = result_user_info.fetchone()
                session['user']['city'] = city
                session['user']['season'] = season
                if not user_info:
                    # If user is not in user_information table, redirect to the quiz page
                    return redirect(url_for('quiz', name=session['user']['name'], username=session['user']['username'], phone=session['user']['phone'], email=session['user']['email']))
                else:
                    # If user exists in user_information table, redirect to the main page
                    return redirect(url_for('main'))  # Or your main dashboard page

            else:
                # If username or password is incorrect
                return render_template('login.html', message="Invalid username or password")

        # Render login page if the request method is GET
        return render_template('login.html', title='Fashion Hub')

    def signup(self):
        if request.method == 'POST':
            # Get user details from the signup form
            name = request.form['name'].title()
            username = request.form['username'].lower().strip().replace(" ", "")
            phone = request.form['phone']
            email = request.form['email'].lower()
            password = request.form['password']
            # Get latitude and longitude from the form (hidden fields from JavaScript)
            lat = request.form.get('latitude')
            lon = request.form.get('longitude')

            # Fetch city and season using get_location
            city = request.form.get('city', 'Unknown City')
            season = request.form.get('season', 'Unknown Season')

            if lat and lon:
                location_data = self.get_location_manual(float(lat), float(lon))  # Call a helper function
                city = location_data.get("city", "Unknown City")
                season = location_data.get("season", "Unknown Season")

            # Check if the username already exists in the login table
            query_check_username = text('SELECT username FROM login WHERE username=:username')
            result = self.db.session.execute(query_check_username, {'username': username})
            existing_user = result.fetchone()

            if existing_user:
                return render_template('login.html', message="Username already exists. Please choose a different username.")

            # Store user details in the session for later use
            session['user_details'] = {
                'name': name,
                'username': username,
                'phone': phone,
                'email': email,
                'password': password,
                'city': city or "Unknown City",
                'season': season or "Unknown Season"
            }

            return render_template('quiz.html', name=session['user_details']['name'], username=session['user_details']['username'], phone=session['user_details']['phone'], email=session['user_details']['email'])
        return render_template('login.html', title='Sign Up')


    def logout(self):
        # Clear the session data and log out the user
        session.pop('user', None)
        return redirect(url_for('main'))  # Redirect to the home page after logging out

class Profile:
    def __init__(self, app, db):
        self.app = app
        self.db = db
        self.register_routes()

    def register_routes(self):
        self.app.add_url_rule('/profile', methods=['GET'], view_func=self.profile)
        self.app.add_url_rule('/quiz', methods=['GET', 'POST'], view_func=self.quiz)
        self.app.add_url_rule('/dataset-images/<path:filename>', methods=['GET'], view_func=self.dataset_images)
        self.app.add_url_rule('/run_virtual_try_on', methods=['GET', 'POST'], view_func=self.virtual_try_on)
        self.app.add_url_rule('/check_status/<username>/<vton_img>/<garm_img>', methods=['GET', 'POST'], view_func=self.check_status)

    IMAGES_FOLDER = os.path.abspath(os.path.join(os.getcwd(), "data", "fashion-dataset", "images"))
    def dataset_images(self, filename):
        IMAGES_FOLDER = os.path.abspath(os.path.join(os.getcwd(), "data", "fashion-dataset", "images"))
        return send_from_directory(IMAGES_FOLDER, filename)
    
    def virtual_try_on(self):
        username = session.get('user', {}).get('username')
        if not username:
            return jsonify({"success": False, "error": "User not logged in."})

        data = request.json
          # Replace with dynamic path if needed
        img_name = db.session.execute(
                text("SELECT virtual_try_on_image FROM user_information WHERE username = :username"), 
                {"username": username}
            )
        vton_img_path = f"app/static/uploads/virtual_try_on/" + img_name.fetchone()[0]
        garm_img_path = data.get("garm_img_path").replace("/dataset-images", "data/fashion-dataset/images")

        if not vton_img_path or not garm_img_path:
            return jsonify({"success": False, "error": "Image paths missing."})

        self.vton_img_name = os.path.splitext(os.path.basename(vton_img_path))[0].replace(" ", "_")
        garm_img_name = os.path.splitext(os.path.basename(garm_img_path))[0].replace(" ", "_")

        user_folder = os.path.normpath(os.path.join("app/static/virtual_try_on", username))
        os.makedirs(user_folder, exist_ok=True)

        status_file = self.get_status_file(username, self.vton_img_name, garm_img_name)

        # Write initial status
        with open(status_file, "w") as f:
            json.dump({"status": "processing"}, f)

        def process():
            try:
                time.sleep(5)  # Simulate processing delay

                # Replace this with your actual virtual try-on logic
                success, output_img_one, output_img_two = run_virtual_try_on(username, vton_img_path, garm_img_path)

                if not success:
                    raise Exception("Virtual try-on process failed.")

                status_data = {
                    "status": "completed",
                    "img_one": output_img_one.replace("\\", "/").replace("app/static/",""),
                    "img_two": output_img_two.replace("\\", "/").replace("app/static/","")
                }

                # Write final status
                with open(status_file, "w") as f:
                    json.dump(status_data, f)

            except Exception as e:
                with open(status_file, "w") as f:
                    json.dump({"status": "failed", "error": str(e)}, f)

        # Start the thread for processing
        Thread(target=process).start()

        return jsonify({"success": True, "status": "processing", "message": "Virtual try-on started."})

    def check_status(self, username, vton_img, garm_img):
        username = session.get('user', {}).get('username')
        vton_img_name = self.vton_img_name
        garm_img_name = os.path.splitext(os.path.basename(garm_img))[0].replace(" ", "_")

        status_file = self.get_status_file(username, vton_img_name, garm_img_name)

        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                return jsonify(json.load(f))

        return jsonify({"status": "not_found"})

    def get_status_file(self, username, vton_img, garm_img):
        user_folder = os.path.normpath(os.path.join("app/static/virtual_try_on", username))
        os.makedirs(user_folder, exist_ok=True)
        return os.path.normpath(os.path.join(user_folder, f"{vton_img}_{garm_img}_status.json"))

    def profile(self):
        if 'user' in session:
            # Retrieve the username from session
            username = session['user']['username']
    
            # Query the user_information table for details based on the username
            result = db.session.execute(
                text("SELECT * FROM user_information WHERE username = :username"), 
                {"username": username}
            )
            user_info = result.fetchone()  # Fetch the first matching record
    
            if user_info:
                # Extracting data from the user_info tuple by index
                email = session['user']['email']
                phone = session['user'].get('phone', 'Not Provided')  # Default value if 'phone' is missing
                name = session['user']['name']
                city = session['user'].get('city', 'Not Detected!')
        
                # Assuming the columns are returned in this order:
                profile_pic = user_info[2]  # Update the index as per your table columns
                gender = user_info[3]
                date_of_birth = user_info[4]
                body_type = user_info[5]
                height = user_info[6]
                weight = user_info[7]
                preferred_color = user_info[8]
                preferred_fabrics = user_info[9]
                preferred_styles = user_info[10]
                occasion_types = user_info[11]
                style_goals = user_info[12]
                budget = user_info[13]
                skin_color = user_info[14]
                wardrobe_img = user_info[15]
                user_title = user_info[16]
                user_about_1 = user_info[17]
                user_about_2 = user_info[18]
                session['user']['virtual_try_on_image'] = user_info[19]
                current_date = datetime.now().date()

                if date_of_birth:
                    age = current_date.year - date_of_birth.year - ((current_date.month, current_date.day) < (date_of_birth.month, date_of_birth.day))
                    # Categorize based on age
                    if age < 18:
                        if gender.lower() == "male":
                            gender = "Boys"
                        elif gender.lower() == "female":
                            gender = "Girls"
                        else:
                            gender = "Other"
                    else:
                        if gender.lower() == "male":
                            gender = "Men"
                        elif gender.lower() == "female":
                            gender = "Women"
                else:
                    age = 0
                    gender = "Unisex"
                        
                category_dict = recommend_fashion(
                                    gender=gender,
                                    baseColour=[color.strip() for color in (preferred_color or "").split(',')],
                                    preferredFabrics=[fabrics.strip() for fabrics in (preferred_fabrics or "").split(',')],
                                    preferredStyles=[styles.strip() for styles in (preferred_styles or "").split(',')],
                                    occasionTypes=[occasion.strip() for occasion in (occasion_types or "").split(',')],
                                    styleGoals=[goal.strip() for goal in (style_goals or "").split(',')],
                                    bodyType=body_type
                                )
                
                image_path = "app/static/uploads/profile/" + profile_pic
                # Call the function and get results
                try:
                    skin_tone, gender_sentence, age_sentence, recommend_color, gender_category, detected_age, outfits = process_fashion_recommendation(image_path)
                except Exception as e:
                    print("An error occurred while processing the fashion recommendation:", e)
                    skin_tone = "Unknown"
                    gender_sentence = "Unknown"
                    age_sentence = "Unknown"
                    recommend_color = "Unknown"
                    gender_category = "Unknown"
                    detected_age = "Unknown"
                    outfits = []
                
                # Weather based recommendation
                from app.globals import season
                if season and gender:
                    try:
                        weather_recommendations = weather_based_recommend(season, gender)
                    except Exception as e:
                        weather_recommendations = []
                        print(f"Weather recommendation error: {str(e)}")
                else:
                    weather_recommendations = []
                    if not season:
                        print("No season data available (tried form, session, and location)")
                    if not gender:
                        print("No gender data available")
                
                image_wardrobe_path = "app/static/uploads/profile/" + profile_pic
                image_recommend = rec(image_wardrobe_path)
                
                occasion_types = occasion_types.split(',')
                if occasion_types:
                    all_occasion = {}
                    for occ in occasion_types:
                        if occ == 'Casual Outing':
                            occ = 'Casual'
                        reco = recommend(occ, gender, top_items=5)
                        all_occasion[occ] = reco
                            
                # Format date
                if date_of_birth:
                    date_obj = datetime.strptime(str(date_of_birth), "%Y-%m-%d")
                    f_date_of_birth = date_obj.strftime("%B %d, %Y")
                else:
                    # Handle the case where date_of_birth is None
                    f_date_of_birth = "Unknown"  # Or use a default value like "January 01, 2000"

                if gender.lower() == 'boys':
                    profile_image = 'avatar-1.png'
                elif gender.lower() == 'girls':
                    profile_image = 'avatar-2.png'
                elif gender.lower() == 'men':
                    profile_image = 'male-avatar.png'
                elif gender.lower() == 'women':
                    profile_image = 'avatar-3.png'
                else:
                    profile_image = 'avatar-4.png'

                
                
                # Fetch trove data based on the filters

                # Pass all data to the template
                return render_template(
                    'profile.html', 
                    profile_image=profile_image,
                    username=username, 
                    email=email, 
                    phone=phone, 
                    name=name,
                    city=city,
                    profile_pic=profile_pic,
                    gender=gender,
                    date_of_birth=f_date_of_birth,
                    body_type=body_type,
                    height=height,
                    weight=weight,
                    preferred_color=preferred_color,
                    preferred_fabrics=preferred_fabrics,
                    preferred_styles=preferred_styles,
                    occasion_types=occasion_types,
                    style_goals=style_goals,
                    budget=budget,
                    skin_color=skin_color,
                    wardrobe_img=wardrobe_img,
                    one_word_user=user_title,
                    paragraph_1=user_about_1,
                    paragraph_2=user_about_2,
                    category_dict=category_dict,
                    skin_tone=skin_tone,
                    recommend_color=recommend_color,
                    gender_sentence=gender_sentence,
                    age_sentence=age_sentence,
                    outfits=outfits,
                    season=season,
                    weather_recommendations=weather_recommendations,
                    image_recommend=image_recommend,
                    all_occasion=all_occasion
                )
            else:
                # If no user info is found, redirect to login or show an error
                return redirect(url_for('login'))
        else:
            # If user is not logged in, redirect to login
            return redirect(url_for('login'))

    def quiz(self):
        profile_pic_filename = None
        wardrobe_img_filename = None
        date_of_birth = None
        user_title = None
        user_about_1 = "No title available"  # Default value
        user_about_2 = "No description available"  # Default value
        gender = 'Other' # Default value
        body_type = None # Default value
        skin_color = None

        if request.method == 'POST':
            # Get quiz data from the form
            profile_pic = request.files.get('profile_pic')
            gender = request.form.get('gender') or None
            date_of_birth_str = request.form['date_of_birth'] or None
            body_type = request.form.get('body_type') or None
            height = request.form['height'] or None
            weight = request.form['weight'] or None
            preferred_color = request.form['preferred_color'] or None
            preferred_fabrics = request.form['preferred_fabrics'] or None
            preferred_styles = request.form['preferred_styles'] or None
            occasion_types = request.form['occasion_types'] or None
            style_goals = request.form['style_goals'] or None
            budget = request.form['budget'] or 0
            skin_color = request.form.get('skin_color') or None
            virtual_try_on_image = request.files.get('virtual-try-on')
            wardrobe_img = request.files.get('wardrobe_img')
        


            if date_of_birth_str:
                try:
                    date_of_birth = datetime.strptime(date_of_birth_str, "%Y-%m-%d").date()  # Convert to date format
                    current_date = date.today()
                    age = current_date.year - date_of_birth.year - ((current_date.month, current_date.day) < (date_of_birth.month, date_of_birth.day))
                except ValueError:
                    return render_template('quiz.html', title='Fashion Quiz', message="Invalid date format. Please enter a valid date.")

            # Ensure age-based gender adjustment only runs if date_of_birth is valid
            if gender == 'Male' and date_of_birth:
                gender = 'Boys' if age < 18 else 'Men'
            elif gender == 'Female' and date_of_birth:
                gender = 'Girls' if age < 18 else 'Women'


            user_details = session.get('user_details')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            
            if wardrobe_img:
                wardrobe_img_filename = f"{user_details['username']}_{timestamp}_{secure_filename(wardrobe_img.filename)}"
                wardrobe_img_path = os.path.join(WARDROBE_IMG_FOLDER, wardrobe_img_filename)
                wardrobe_img.save(wardrobe_img_path)

            if profile_pic:
                profile_pic_filename = f"{user_details['username']}_{timestamp}_{secure_filename(profile_pic.filename)}"

                profile_pic_path = os.path.join(PROFILE_PIC_FOLDER, profile_pic_filename)

                profile_pic.save(profile_pic_path)
                
            if virtual_try_on_image:
                virtual_try_on_image_filename = f"{user_details['username']}_{secure_filename(virtual_try_on_image.filename)}"
                virtual_try_on_image_path = os.path.join(VIRTUAL_TRY_ON_IMG_FOLDER, virtual_try_on_image_filename)
                virtual_try_on_image.save(virtual_try_on_image_path)

            if user_details['name'] and gender and date_of_birth_str and body_type and height and weight and preferred_color and preferred_fabrics and preferred_styles and occasion_types and style_goals and skin_color:
                prompt = f"""
                    Generate a professionally written, engaging, and personalized "About" section for a user profile in two short paragraphs (90-105 words in total). The content should impress the reader and reflect the user's unique style and preferences. Use the following details:
                    - Name: {user_details['name']}
                    - Gender: {gender}  
                    - Age: {date_of_birth_str}  
                    - Body Type: {body_type}  
                    - Height: {height}  
                    - Weight: {weight}  
                    - Preferred Colors: {preferred_color}  
                    - Preferred Fabrics: {preferred_fabrics}  
                    - Preferred Styles: {preferred_styles}  
                    - Occasion Types: {occasion_types}  
                    - Style Goals: {style_goals}  
                    - Skin Color: {skin_color}

                    Ensure the language is elegant, concise, and makes the user sound fashion-forward and confident. Avoid repetition and use positive, inspiring vocabulary.
                    """
                try:
                    response = chat. send_message (prompt)
                    paragraph = response.text
                    paragraph = paragraph.split("\n\n")
                    user_about_1 = paragraph[0]
                    user_about_2 = paragraph[1]
                except Exception as e:
                    user_about_1 = "AI Couldn't generate the title"
                    user_about_2 = "AI Couldn't generate the description"
            
                prompt = f"""
                    Using the following details about a user, provide one word that best describes the overall style or impression of the individual. Focus solely on the most fitting adjective or noun that reflects the user's fashion preferences, style goals, and persona. Do not add any special characters like asterisks or quotation marks—just return the word itself.

                    Details:
                    - Name: {user_details['name']}
                    - Gender: {gender}
                    - Age: {date_of_birth}
                    - Body Type: {body_type}
                    - Height: {height}
                    - Weight: {weight}
                    - Preferred Colors: {preferred_color}
                    - Preferred Fabrics: {preferred_fabrics}
                    - Preferred Styles: {preferred_styles}
                    - Occasion Types: {occasion_types}
                    - Style Goals: {style_goals}
                    - Skin Color: {skin_color}
                    """
                
                try:
                    response = chat.send_message(prompt)
                    clean_output = response.text.strip().replace('*', '')  # Removing any asterisks
                    user_title = clean_output
                except Exception as e:
                    user_title = "AI Couldn't generate the title"

                # Ensure budget is a valid number (float)
            try:
                budget = float(budget) if budget else None
            except ValueError:
                return render_template('quiz.html', title='Fashion Quiz', message="Please enter a valid number for the budget.")

            # Get user details from session
            user_details = session.get('user_details', {})
            if not user_details:
                return redirect(url_for('login'))  # Redirect if session data is missing


            if user_details:
                # Insert into login table
                insert_query_login = text('INSERT INTO login (name, username, phone, email, password) VALUES (:name, :username, :phone, :email, :password)')
                self.db.session.execute(insert_query_login, {
                    'name': user_details['name'],
                    'username': user_details['username'],
                    'phone': user_details['phone'],
                    'email': user_details['email'],
                    'password': user_details['password']
                })
                self.db.session.commit()

                # Insert into user_information table
                insert_query_info = text('''INSERT INTO user_information (username, profile_pic, gender, date_of_birth, body_type, height, weight, preferred_color, preferred_fabrics, preferred_styles, occasion_types, style_goals, budget, skin_color, wardrobe_img, user_title, user_about_1, user_about_2, virtual_try_on_image)
                                        VALUES (:username, :profile_pic, :gender, :date_of_birth, :body_type, :height, :weight, :preferred_color, :preferred_fabrics, :preferred_styles, :occasion_types, :style_goals, :budget, :skin_color, :wardrobe_img, :user_title, :user_about_1, :user_about_2, :virtual_try_on_image)''')
                self.db.session.execute(insert_query_info, {
                    'username': user_details['username'],
                    'profile_pic': profile_pic_filename or None,
                    'gender': gender or None,
                    'date_of_birth': date_of_birth or None,
                    'body_type': body_type or None,
                    'height': height or None,
                    'weight': weight or None,
                    'preferred_color': preferred_color or None,
                    'preferred_fabrics': preferred_fabrics or None,
                    'preferred_styles': preferred_styles or None,
                    'occasion_types': occasion_types or None,
                    'style_goals': style_goals or None,
                    'budget': budget or None,  # Ensure this is a valid number
                    'skin_color': skin_color,
                    'wardrobe_img': wardrobe_img_filename or None,
                    'user_title': user_title or None,
                    'user_about_1': user_about_1 or None,
                    'user_about_2': user_about_2 or None,
                    'virtual_try_on_image': virtual_try_on_image_filename if virtual_try_on_image_filename else None
                })
                self.db.session.commit()

                # Clear session data after the quiz
                session.pop('user_details', None)

                # Redirect to profile page after successful quiz
                return redirect(url_for('profile'))

            else:
                # If user details are missing from session, redirect to signup
                return redirect(url_for('login'))

        # If the request is GET, render the quiz page
        return render_template('quiz.html', title='Fashion Quiz')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Set limit to 16MB

# Initialize the classes
Dashboard(app)
Login(app, db)
Profile(app, db)

if __name__ == '__main__':
    app.secret_key = 'manojrajgopal'  # Ensure you have a secret key for sessions
    app.run(debug=True)