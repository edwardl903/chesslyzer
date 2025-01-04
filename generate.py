from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import subprocess
import os
import glob
import time
import json

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
            ["python", "testing.py", username],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Assuming the script writes the statistics JSON file to 'json/{username}_statistics.json'
        statistics_path = f'json/{username}_statistics.json'
        if os.path.exists(statistics_path):
            with open(statistics_path, 'r') as f:
                statistics = json.load(f)
            
            # Generate unique image paths with cache-busting query strings
            image_paths = []
            timestamp = int(time.time())  # Get current timestamp to ensure unique filenames
            for i in range(1, 17):  # Assuming 16 images are generated
                image_paths.append(f"/static/images/image_{i}.png?t={timestamp}")

            # Return JSON response with statistics and image paths
            return jsonify({
                "images": image_paths,
                "total_time_spent": statistics['total_time_spent'],
                "total_moves": statistics['total_moves'],
                "total_win_draw_loss": statistics['total_win_draw_loss'],
                "total_results": statistics['total_results'],
                "total_en_passant": statistics['total_en_passant'],
                "total_promotions": statistics['total_promotions'],
                "total_games": statistics['total_games'],
                "longest_winning_streak": statistics['longest_winning_streak'],
                "longest_losing_streak": statistics['longest_losing_streak'],
                "most_played_opponent": statistics['most_played_opponent'],
                "highest_rating": statistics['highest_rating'],
                "most_time_spent_day_dict" : statistics['most_time_spent_day_dict'],
                "my_openings": statistics['my_openings']
            })

        else:
            return jsonify({"error": "Failed to find generated statistics file."}), 500

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to Generate - Sorry!"}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)