# Chesslytics - A Chess Analytics Platform

## Setup and Installation

1. **Create a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows, use `venv\Scripts\activate`

2. **Install Required Python Packages:**

pip install -r requirements.txt

or

pip install python-chess
pip install sklearn
pip install flask
pip install flask-cors
pip install matplotlib
pip install numpy
pip install pandas
pip install seaborn
pip install requests
pip install gunicorn==20.0.4
pip install pandas google-cloud-bigquery
pip install google
pip install pyarrow
pip install google-auth




Flask: Web server framework.
Matplotlib: For generating visualizations.
NumPy: For numerical operations.
Pandas: For data processing and manipulation.
Scikit-learn: For machine learning and statistical operations.
Seaborn: For advanced visualizations.
Requests: For API calls to Chess.com.
Gunicorn: For running the Flask server in production.
Flask-CORS: For handling cross-origin requests.
Python-Chess: For chess game analysis and move generation.

3. **Install Stockfish for Evaluation: (OPTIONAL)**

brew install stockfish

4. **Run the Flask Server Locally:**

python generate.py



5. **Project Architecture**

Backend (generate.py):
The Flask server is responsible for handling requests from the frontend. It runs the testing.py script, which interacts with three main Python scripts: data_processing.py, my_stats.py, and visualizations.py.

Data Processing (data_processing.py):
This script fetches data from the Chess.com API, processes it, and extracts relevant statistics. It cleans and prepares the data, then outputs it as a JSON file and a CSV with numerous columns used later for analysis.

Statistics (my_stats.py):
This script reads the processed CSV file created by data_processing.py and generates all the statistics shown on the website. It outputs the statistics to a JSON file.

Visualizations (visualizations.py):
This script uses libraries such as matplotlib and seaborn to generate graphs and charts, which are saved as images and served via the web application.

Frontend (public/index.html & static/styles.css):
The index.html file contains the structure and content of the homepage, and it links to the static files (images, CSS, etc.). The styles are defined in static/styles.css, and images (including background and graphs) are stored in static/images.

6. **Development**
jupyter-lab
Use pip-compile requirements.in to generate the requirements.txt file.
heroku push origin main

$ heroku login
$ heroku git:clone -a chesslyzer-app
$ cd chesslyzer-app    
$ git add .
$ git commit -am "make it better"
$ git push heroku main

7. **My Future Iterations**
Add Filter Buttons

Create buttons to filter live games and game modes (e.g., Blitz, Rapid, Bullet, Daily).
Improve Visualization Speed

Speed up visualizations by optimizing SQL queries and loading only necessary data.
Fix Styles

Update the design to improve the look and feel.
Make It Mobile-Friendly

Ensure the app works well on mobile devices with responsive design.

8. **Working on**
Currently changing testing.py andd adding bigquery_uploader.py to incorporate GCP
Added to the testing.py, added to gcp folder
added to requirements.txt

for heroku
credentials_info = json.loads(os.environ['GOOGLE_CREDENTIALS'])
credentials = service_account.Credentials.from_service_account_info(credentials_info)
client = bigquery.Client(credentials=credentials, project=credentials.project_id)

replacing this
client = bigquery.Client.from_service_account_json(credentials_path)
# pip install looker-sdk google-auth google-auth-oauthlib google-auth-httplib2

#right now im appending to bigquery in table, jlee2327 shows up with my games
❌ Error: Please install the 'db-dtypes' package to use this function.