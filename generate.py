from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Static directory setup
@app.route('/')
def serve_index():
    return send_from_directory('public', 'index.html') # public/index.html

@app.route('/generate', methods=['POST'])
def generate_csv():
    data = request.json
    username = data['username']

    try:
        # Call the chesslytics.py script to generate the CSV and images
        result = subprocess.run(
            ["python", "chesslytics.py", username],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Assuming the Python script generates images and saves them in static/images/
        image_paths = []
        for i in range(1, 4):  # For testing with one image (you can change this)
            image_paths.append(f"/static/images/image_{i}.png")

        # Return JSON response with image paths
        return jsonify({
            "images": image_paths
        })

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to generate CSV"}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)
