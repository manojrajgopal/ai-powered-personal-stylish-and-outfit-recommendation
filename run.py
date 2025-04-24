from app.main import app  # Import the 'app' object from the 'main.py' file inside 'app' folder
PORT = 5000

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=False)
