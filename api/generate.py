from flask import Flask, send_from_directory, request, jsonify
from flask_cors import CORS
import subprocess
import os
import glob
import time
import json

# Get the project root directory (one level up from api/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__, 
           static_folder=os.path.join(project_root, 'static'),
           static_url_path='/static')

# Enable CORS for all routes
CORS(app)

latest_results = None

# Static directory setup
@app.route('/')
def serve_index():
    public_dir = os.path.join(project_root, 'public')
    return send_from_directory(public_dir, 'index.html')

@app.route('/generate', methods=['POST'])
def generate_csv():
    global latest_results

    data = request.json
    username = data['username']
    year = data['year']
    print(f"Received year: {year}")

    if not username: # i dont account for year, should be fine
        return jsonify({'error': 'Username is required'}), 400

    try:
        # Delete any previously generated images to overwrite them
        image_files = glob.glob(os.path.join(project_root, 'static', 'images', '*.png'))  # Assuming PNG images
        for file in image_files:
            os.remove(file)

        # Call the testing.py script to generate the CSV and images
        print(f"Running testing.py script for username: {username} for year: {year}")  # Debugging statement

        # Run the script from the project root
        script_path = os.path.join(project_root, 'tests', 'testing.py')
        result = subprocess.run(
            ["python3", script_path, username, year],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=project_root  # Run from project root
        )

        # Assuming the script writes the statistics JSON file to 'data/json/{username}_statistics.json'
        statistics_path = os.path.join(project_root, 'data', 'json', f'{username}_statistics.json')
        if os.path.exists(statistics_path):
            with open(statistics_path, 'r') as f:
                statistics = json.load(f)
            
            # Generate unique image paths with cache-busting query strings
            print("Processing Images...") 
            image_paths = []
            timestamp = int(time.time())  # Get current timestamp to ensure unique filenames
            for i in range(0, 20):
                image_path = f"/static/images/image_{i}.png"
                full_path = os.path.join(project_root, 'static', 'images', f"image_{i}.png")
                if os.path.exists(full_path):
                    image_paths.append(f"{image_path}?t={timestamp}")
                else:
                    print(f"Image {image_path} does not exist, skipping.")
            # Return JSON response with statistics and image paths
            return jsonify({
                "images": image_paths,
                "my_avatar": statistics['my_avatar'],
                "my_username": statistics['my_username'],
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
                "my_openings": statistics['my_openings'],
                "timecontrol_counts": statistics['timecontrol_counts'],
                "timeclass_counts": statistics['timeclass_counts'],
                "biggest_games": statistics['biggest_games'],
                'last_game_ratings': statistics['last_game_ratings']
            })

        else:
            return jsonify({"error": "No games found for the user in 2024"}), 500

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to Generate - Sorry! Please contact us!"}), 500

# Start the Flask app
if __name__ == "__main__":
    app.run(debug=True)