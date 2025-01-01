from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
import subprocess
import os

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

@app.route('/generate', methods=['POST'])
def generate_csv():
    data = request.json
    username = data['username']

    try:
        # Call the chesslytics.py script to generate the CSV
        result = subprocess.run(
            ["python3", "chesslytics.py", username],  # Adjust the path if necessary
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Assuming the Python script generates a CSV with the username
        csv_file = f"{username}.csv"

        if os.path.exists(csv_file):
            return send_file(csv_file, as_attachment=True, mimetype='text/csv')  # Optional: set MIME type explicitly

        return jsonify({"error": "CSV generation failed"}), 500

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to generate CSV"}), 500

if __name__ == "__main__":
    app.run(debug=True)