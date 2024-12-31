# chess_scraper.py
# Description: This script scrapes chess games from the chess.com game archive page.
# Author: Edward
# Date: 2024-08-16

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import time
import re
import csv

USERNAME = "jlee2327"

def setup_driver():
    """Sets up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--disable-javascript")
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    return webdriver.Chrome(options=chrome_options)

def extract_game_info(row):
    """Extracts relevant game information from a table row."""
    columns = row.find_all('td')
    if len(columns) <= 1:
        return None

    opponent_info = columns[1].text.strip()
    player_name, player_rating, opponent_name, opponent_rating, player_color, opponent_color = parse_player_info(opponent_info)

    accuracies = columns[3].text.strip().split()
    player_accuracy, opponent_accuracy = extract_accuracies(accuracies, player_color)

    result_text = columns[2].text.strip()
#     print(columns[2].text.strip())
    game_result = determine_game_result(result_text, player_color)

    game_link = extract_game_link(row)

    if not game_link:
        return None  # Skip rows without a game link

    player_country, opponent_country = extract_countries(row)

    game_data = {
        'Game_Type': columns[0].text.strip(),
        'Result': game_result,
        'Accuracies': columns[3].text.strip(),
        'Total_Moves': columns[4].text.strip(),
        'Date': columns[5].text.strip(),
        'Link': game_link,
        'Player_Details': {
            player_name: {
                'currentRating': player_rating,
                'country': player_country,
                'username': USERNAME,
                'color': player_color,
                'accuracy': player_accuracy,
            },
            opponent_name: {
                'currentRating': opponent_rating,
                'country': opponent_country,
                'username': opponent_name,
                'color': opponent_color,
                'accuracy': opponent_accuracy,
            }
        }
    }
    return game_data

def parse_player_info(opponent_info):
    """Parses the player and opponent information from a text string."""
    cleaned_info = ' '.join(opponent_info.split())
    pattern = r'(.+?)\((\d{4})\).*?(.+?)\((\d{4})\)'

    match = re.match(pattern, cleaned_info)
    if match:
        player_name, player_rating, opponent_name, opponent_rating = match.group(1).strip(), match.group(2).strip(), match.group(3).strip(), match.group(4).strip()
        player_color, opponent_color = ('White', 'Black') if USERNAME in player_name else ('Black', 'White')
    else:
        player_name, player_rating, opponent_name, opponent_rating = 'Me', 'N/A', 'Opponent', 'N/A'
        player_color, opponent_color = 'Unknown', 'Unknown'

    return player_name, player_rating, opponent_name, opponent_rating, player_color, opponent_color

def extract_accuracies(accuracies, player_color):
    """Extracts accuracies for the player and opponent based on player color."""
    if len(accuracies) == 2:
        return (accuracies[0], accuracies[1]) if player_color == 'White' else (accuracies[1], accuracies[0])
    return 'N/A', 'N/A'

def determine_game_result(result_text, player_color):
    """
    Determines the game result based on the result text and player color.

    Args:
        result_text (str): The result of the game, which may include line breaks (e.g., '1\n0', '0\n1', '½\n½').
        player_color (str): The color of the player ('White' or 'Black').

    Returns:
        str: 'Win', 'Loss', or 'Draw' based on the result.
    """
    # Debugging: Print the raw result_text
#     print(f"Debugging Result_Text: {result_text}")
    
    # Clean the result_text by removing any whitespace and newlines
    cleaned_result = result_text.strip().replace('\n', '')
    
    # Debugging: Print the cleaned result_text
#     print(f"Cleaned Result_Text: {cleaned_result}")
    
    if cleaned_result == '½½':
        return 'Draw'
    elif (cleaned_result == '10' and player_color == 'White') or (cleaned_result == '01' and player_color == 'Black'):
        return 'Win'
    else:
        return 'Loss'
    
def extract_game_link(row):
    """Extracts the game link from a table row."""
    game_link_element = row.find('a', {'class': 'archive-games-background-link'})
    return game_link_element['href'] if game_link_element else 'N/A'

def extract_countries(row):
    """Extracts the countries for the player and opponent."""
    country_divs = row.find_all('div', {'data-cy': 'user-country-flag'})
    if len(country_divs) >= 2:
        player_country = country_divs[0].get('v-tooltip', 'Unknown')
        opponent_country = country_divs[1].get('v-tooltip', 'Unknown')
    else:
        player_country = opponent_country = 'Unknown'
    return player_country, opponent_country

def scrape_games_from_page(driver, page_url, scraped_game_links):
    """Scrapes all games from a given page URL."""
    driver.get(page_url)
    time.sleep(3)  # Wait for the page to load
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    
    games_table = soup.find('table', {'class': 'archive-games-table'})
    games = []

    if games_table:
        tbody = games_table.find('tbody')
        rows = tbody.find_all('tr')

        for row in rows:
            game_data = extract_game_info(row)
            if game_data:
                if game_data['Link'] not in scraped_game_links:
                    games.append(game_data)
                    scraped_game_links.add(game_data['Link'])
    
    return games

def save_games_to_csv(games, filename='chess_games.csv'):
    """Saves the scraped games to a CSV file."""
    headers = [
        'Game_Type', 'Result', 'Accuracies', 'Total_Moves', 'Date', 'Link',
        'Player1_Name', 'Player1_Rating', 'Player1_Country', 'Player1_Username', 'Player1_Color', 'Player1_Accuracy',
        'Player2_Name', 'Player2_Rating', 'Player2_Country', 'Player2_Username', 'Player2_Color', 'Player2_Accuracy'
    ]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for game in games:
            player1, player2 = list(game['Player_Details'].keys())
            
            row = [
                game['Game_Type'], game['Result'], game['Accuracies'], game['Total_Moves'], game['Date'], game['Link'],
                player1, game['Player_Details'][player1]['currentRating'], game['Player_Details'][player1]['country'], game['Player_Details'][player1]['username'], game['Player_Details'][player1]['color'], game['Player_Details'][player1]['accuracy'],
                player2, game['Player_Details'][player2]['currentRating'], game['Player_Details'][player2]['country'], game['Player_Details'][player2]['username'], game['Player_Details'][player2]['color'], game['Player_Details'][player2]['accuracy']
            ]
            
            writer.writerow(row)
    
    print(f"Data saved to {filename}")

def scrape_chess_games():
    """Main function to scrape chess games and save them as JSON and CSV."""
    driver = setup_driver()
    base_url = f"https://www.chess.com/games/archive/{USERNAME}?page="
    all_games = []
    scraped_game_links = set()

    for page_num in range(1, 100):  # Adjust range as needed for the number of pages
        page_url = base_url + str(page_num)
        print(f"Scraping page {page_num}: {page_url}")
        games = scrape_games_from_page(driver, page_url, scraped_game_links)
        if not games:
            print("No more new games found. Stopping.")
            break  # Stop if no new games are found (end of pages or duplicates)
        all_games.extend(games)
        time.sleep(3)  # Avoid hammering the server too hard

    with open(f'{USERNAME}_chess_games.json', 'w') as f:
        json.dump(all_games, f, indent=4)
    
    save_games_to_csv(all_games)
    driver.quit()
    print(f"Scraping completed. Total unique games scraped: {len(all_games)}")


from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials
import json

# Path to your service account JSON key file
SERVICE_ACCOUNT_FILE = 'path/to/your/service-account-key.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Spreadsheet ID and range
SPREADSHEET_ID = 'your_spreadsheet_id'
RANGE_NAME = 'Sheet1!A1'

def save_json_to_google_sheet(json_data):
    """
    Saves JSON data to a Google Spreadsheet.
    """
    # Authenticate with the Google Sheets API
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('sheets', 'v4', credentials=creds)

    # Prepare the data for insertion
    headers = [
        'Game_Type', 'Result', 'Accuracies', 'Total_Moves', 'Date', 'Link',
        'Player1_Name', 'Player1_Rating', 'Player1_Country', 'Player1_Username', 'Player1_Color', 'Player1_Accuracy',
        'Player2_Name', 'Player2_Rating', 'Player2_Country', 'Player2_Username', 'Player2_Color', 'Player2_Accuracy'
    ]
    
    values = [headers]  # Start with headers
    
    for game in json_data:
        player1, player2 = list(game['Player_Details'].keys())
        row = [
            game['Game_Type'], game['Result'], game['Accuracies'], game['Total_Moves'], game['Date'], game['Link'],
            player1, game['Player_Details'][player1]['currentRating'], game['Player_Details'][player1]['country'], game['Player_Details'][player1]['username'], game['Player_Details'][player1]['color'], game['Player_Details'][player1]['accuracy'],
            player2, game['Player_Details'][player2]['currentRating'], game['Player_Details'][player2]['country'], game['Player_Details'][player2]['username'], game['Player_Details'][player2]['color'], game['Player_Details'][player2]['accuracy']
        ]
        values.append(row)

    # Write data to Google Sheet
    body = {'values': values}
    service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=RANGE_NAME,
        valueInputOption='RAW',
        body=body
    ).execute()

    print("Data successfully saved to Google Spreadsheet.")

if __name__ == "__main__":
    scrape_chess_games()
    # Load the JSON data from file
    with open(f'{USERNAME}_chess_games.json', 'r') as f:
        json_data = json.load(f)
    save_json_to_google_sheet(json_data)  # Save to Google Spreadsheet
