from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import Flask-CORS
import subprocess
import os

app = Flask(__name__)

# Enable CORS for the specific frontend origin
CORS(app, resources={r"/*": {"origins": "https://chesslyzer.vercel.app"}})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate', methods=['POST'])
def generate_csv():
    data = request.json
    username = data.get('username')

    try:
        result = subprocess.run(
            ["python", "chesslyzer.py", username],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        csv_file = f"{username}.csv"
        if os.path.exists(csv_file):
            return send_file(csv_file, as_attachment=True)

        return jsonify({"error": "CSV generation failed"}), 500

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to generate CSV"}), 500

if __name__ == "__main__":
    app.run(debug=True)
