import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_dashboard():
    data = request.json
    username = data['username']

    try:
        # Call the chesslytics.py script with the username
        result = subprocess.run(
            ["python3", "chesslytics.py", username],  # Adjust `python3` if needed
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # The CSV file path is returned by the script
        csv_file = f"{username}.csv"  # Assuming the CSV is saved with the username

        # Placeholder: You can upload the CSV to Google Sheets or another service here

        # Return Looker Studio URL (or any other response)
        #dashboard_url = f"https://lookerstudio.google.com/reporting/YOUR_REPORT_ID/page/XYZ?username={username}"


        return 0
        #return jsonify({"dashboard_url": dashboard_url})

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return jsonify({"error": "Failed to generate CSV"}), 500

if __name__ == "__main__":
    app.run(debug=True)