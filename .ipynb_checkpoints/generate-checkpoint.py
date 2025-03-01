from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import subprocess
import os
import glob
import time

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

latest_results = None

# Static directory setup
@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html')  # public/index.html

@app.route('/generate', methods=['POST'])
def generate_csv():
    global latest_results

    data = request.json
    username = data['username']

    if not username:
        return jsonify({'error': 'Username is required'}), 400

    try:
        # Delete any previously generated images to overwrite them
        image_files = glob.glob('static/images/*.png')  # Assuming PNG images
        for file in image_files:
            os.remove(file)

        # Call the chesslytics.py script to generate the CSV and images
        result = subprocess.run(
            ["python", "chesslytics.py", username],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Generate unique image paths with cache-busting query strings
        image_paths = []
        timestamp = int(time.time())  # Get current timestamp to ensure unique filenames
        for i in range(1, 17):  # Assuming 16 images are generated
            image_paths.append(f"/static/images/image_{i}.png?t={timestamp}")

        # Return JSON response with image paths (with query strings to avoid cache)
        return jsonify({
            "images": image_paths
        })

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to Generate - Sorry!"}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)