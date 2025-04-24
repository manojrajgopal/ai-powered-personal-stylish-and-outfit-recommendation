import os
import shutil
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from gradio_client import Client, handle_file

# List of API tokens for API fallback mechanism

api_tokens = [] # Add your API tokens here

# Initialize API clients once
api_clients = [Client("levihsu/OOTDiffusion", hf_token=token) for token in api_tokens]

# Define CSV file for image metadata
csv_file = "data/processed/virtal_try_on_image_data.csv"
os.makedirs("data/processed", exist_ok=True)
if not os.path.exists(csv_file):
    pd.DataFrame(columns=["username", "input_image", "garment_image", "output_image_one", "output_image_two"]).to_csv(csv_file, index=False)

# Allowed image formats
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    return os.path.splitext(filename)[0]

def check_existing_entry(username, vton_img_name, garm_img_name):
    df = pd.read_csv(csv_file)
    existing = df[(df["username"] == username) & (df["input_image"] == vton_img_name) & (df["garment_image"] == garm_img_name)]
    return existing.iloc[0] if existing.any().any() else None

def save_to_csv(username, vton_img_name, garm_img_name, output_image_one, output_image_two):
    df = pd.read_csv(csv_file) if os.stat(csv_file).st_size > 0 else pd.DataFrame(columns=["username", "input_image", "garment_image", "output_image_one", "output_image_two"])
    existing_entry = check_existing_entry(username, vton_img_name, garm_img_name)
    
    if existing_entry is None:  # Correct check
        new_entry = pd.DataFrame([[username, vton_img_name, garm_img_name, output_image_one, output_image_two]],
                                 columns=["username", "input_image", "garment_image", "output_image_one", "output_image_two"])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_file, index=False)


# Fastest API call using ThreadPoolExecutor
def call_api(client, vton_img_path, garm_img_path):
    try:
        return client.predict(
            vton_img=handle_file(vton_img_path),
            garm_img=handle_file(garm_img_path),
            n_samples=2,
            n_steps=20,
            image_scale=2,
            seed=-1,
            api_name="/process_hd"
        )
    except Exception:
        return None

async def process_images(username, vton_img_path, garm_img_path, output_img_one, output_img_two):
    os.makedirs(os.path.dirname(output_img_one), exist_ok=True)

    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=len(api_clients)) as executor:
        tasks = [loop.run_in_executor(executor, call_api, client, vton_img_path, garm_img_path) for client in api_clients]
        
        for future in asyncio.as_completed(tasks):
            result = await future
            if result:
                shutil.copy(result[0]["image"], output_img_one)
                shutil.copy(result[1]["image"], output_img_two)
                save_to_csv(username, os.path.basename(vton_img_path), os.path.basename(garm_img_path), output_img_one, output_img_two)
                return True  # Stop as soon as one API succeeds!

    return False  # If all APIs fail

def run_virtual_try_on(username, vton_img_path, garm_img_path):
    if not (allowed_file(vton_img_path) and allowed_file(garm_img_path)):
        return "Invalid file format. Only PNG, JPG, JPEG, and WEBP are allowed."

    vton_img_name = sanitize_filename(os.path.basename(vton_img_path))
    garm_img_name = sanitize_filename(os.path.basename(garm_img_path))
    user_folder = os.path.join("app/static/virtual_try_on", username)
    os.makedirs(user_folder, exist_ok=True)

    output_img_one = os.path.join(user_folder, f"{vton_img_name}_{garm_img_name}_1.jpg")
    output_img_two = os.path.join(user_folder, f"{vton_img_name}_{garm_img_name}_2.jpg")

    existing_entry = check_existing_entry(username, vton_img_name, garm_img_name)
    if existing_entry is not None:
        return f"Images already processed: {existing_entry['output_image_one']}, {existing_entry['output_image_two']}"

    success = asyncio.run(process_images(username, vton_img_path, garm_img_path, output_img_one, output_img_two))

    return True if success else False, output_img_one, output_img_two