import os
output_dir = 'static/images/'
os.makedirs(output_dir, exist_ok=True)

import re
import requests

def fetch_player_data(username):
    url = f"https://api.chess.com/pub/player/{username}"
    headers = {
        "User-Agent": "MyChessApp/1.0 (https://example.com)"  # ensure my request is legit 
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.text}")
        return None
    
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Failed to parse JSON. Status code: {response.status_code}, Content: {response.text}")
        return None

#my_data = fetch_player_data("emazing19")
#print(my_data.keys())

#i only want my avatar
#what do i want? 1. their country so I can strip it, if titled, is_streamer
## Future Note: I want to extract your pfp and be able to host a wesbite showing your pfp

def fetch_monthly_games(username, year, month):
    url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02}"
    headers = {
        "User-Agent": "MyChessApp/1.0 (https://example.com)"
    }
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        print(f"Error: Received status code {response.status_code}")
        print(f"Response content: {response.text}")
        return None
    
    try:
        data = response.json()
        return data["games"]
    except requests.exceptions.JSONDecodeError:
        print(f"Failed to parse JSON. Status code: {response.status_code}, Content: {response.text}")
        return None


def fetch_all_games(username):
    player_data = fetch_player_data(username)
    if not player_data or "joined" not in player_data:
        print("Could not determine player's join date.")
        return []

    join_date = datetime.fromtimestamp(player_data["joined"])
    current_date = datetime.now()

    all_games = []
    year, month = join_date.year, join_date.month

    while (year, month) <= (current_date.year, current_date.month):
        games = fetch_monthly_games(username, year, month)
        all_games.extend(games)

        # Increment month
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return all_games

def fetch_all_games_for_current_year(username, game_filter=None):
    player_data = fetch_player_data(username)
    if not player_data or "joined" not in player_data:
        print("Could not determine player's join date.")
        return []
        
    current_date = datetime.now()
    all_games = []
    
    year = current_date.year
    month = 1

    while month <= current_date.month:
        games = fetch_monthly_games(username, year, month)
        if games is None:
            print(f"No games found for {username} in {year}-{month:02d}.")
            games = []  
        if game_filter:
            games = [game for game in games if game_filter(game)]
        all_games.extend(games)
        month += 1

    return all_games

def fetch_all_games_for_last_year(username, game_filter=None):
    player_data = fetch_player_data(username)
    if not player_data or "joined" not in player_data:
        print("Could not determine player's join date.")
        return []

    current_date = datetime.now()
    last_year = current_date.year - 1
    all_games = []
    year = last_year
    month = 1

    while month <= 12:
        games = fetch_monthly_games(username, year, month)
        if games is None:
            print(f"No games found for {username} in {year}-{month:02d}.")
            games = []
        if game_filter:
            games = [game for game in games if game_filter(game)]
        all_games.extend(games)

        # Increment month
        month += 1

    return all_games


import chess
import chess.engine

def evaluate_positions(game_moves, engine_path, depth=5, time_limit=0.1):
    """
    Evaluates the positions of a chess game after every move using a chess engine.
    
    Args:
        game_moves (list): A list of moves for a single chess game in standard algebraic notation.
        engine_path (str): Path to the chess engine executable (e.g., '/path/to/stockfish').
        depth (int): Depth of evaluation for the chess engine.
        time_limit (float): Time limit for each move evaluation in seconds (optional, default 2.0 seconds).
    
    Returns:
        list: A list of evaluations after each move (in centipawns).
    """
    board = chess.Board()
    
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        evaluations = []
        
        for move in game_moves:
            try:
                board.push_san(move)
            except ValueError as e:
                print(f"Invalid move {move}: {e}")
                break
            
            info = engine.analyse(board, chess.engine.Limit(time=time_limit))  # Set a time limit for evaluation
            
            # Get the PovScore (from the point of view of the current side to move)
            pov_score = info["score"]
            
            # Access the relative score from the PovScore object
            relative_score = pov_score.relative
            
            # Check if the relative score indicates a checkmate situation
            if relative_score.is_mate():
                eval_cp = f"M in {relative_score.mate()}"
            else:
                # For non-mate situations, use the centipawn evaluation
                eval_cp = relative_score.score(mate_score=10000)  # Centipawn score or checkmate
            
            evaluations.append(eval_cp)
    
    return evaluations



import re
import pandas as pd
from datetime import datetime

import re

def extract_moves_from_pgn(pgn):
    """
    Extracts only chess moves from a PGN string, ignoring move numbers, annotations, and timestamps.
    Ensures that only valid moves containing letters (indicating pieces or files) are included.
    
    Args:
        pgn (str): The PGN string containing metadata and moves.
    
    Returns:
        list: A list of chess moves in sequence, including any pawn promotions (e.g., '=Q', '=N').
    """
    # Remove PGN metadata (lines that start with '[')
    pgn_clean = re.sub(r'\[.*?\]\n', '', pgn)

    # Split the PGN into individual words (moves, annotations, timestamps)
    pgn_clean = pgn_clean.split()
    moves = []

    for word in pgn_clean:
        # Skip anything that looks like a move number (e.g., '1.', '2.', etc.)
        if word.isdigit() or '.' in word:
            continue
        
        # Skip annotations (like '{[%clk 0:02:54.6]}')
        if word.startswith('{'):
            continue
    
        if word in ['O-O', 'O-O-O']:
            moves.append(word)
        # Handle pawn promotions (e.g., d8=Q, e7=R)
        elif '=' in word:
            moves.append(word)
        # Handle regular moves (e.g., Nc6, exd5), ensure it contains at least one letter
        elif any(c.isalpha() for c in word):
            moves.append(word)

    return moves


def extract_pgn_metadata(pgn):
    """
    Extract metadata from a PGN string.
    
    Args:
        pgn (str): The PGN string containing metadata and moves.
    
    Returns:
        dict: A dictionary containing the metadata fields and their values.
    """
    metadata = {}
    metadata_lines = re.findall(r'\[(\w+)\s+"(.*?)"\]', pgn)
    for key, value in metadata_lines:
        metadata[key] = value
    return metadata

def calculate_time_difference(start_time, end_time):
    """
    Calculate the time difference between two times in HH:MM:SS format.
    
    Args:
        start_time (str): Start time in HH:MM:SS format.
        end_time (str): End time in HH:MM:SS format.
    
    Returns:
        int: Time difference in seconds.
    """
    try:
        fmt = "%H:%M:%S"
        start = datetime.strptime(start_time, fmt)
        end = datetime.strptime(end_time, fmt)
        delta = (end - start).seconds
        return delta
    except Exception:
        return None

'''def extract_moves_and_timestamps(game_data):
    # Regular expression to match move numbers (multi-digit supported) and timestamps
    matches = re.findall(r'(\d+)\. \S+ \{\[%clk ([0-9:]+(?:\.[0-9]+)?)\]\}', game_data)
    # Return the last two pairs (move number, timestamp), if they exist
    return matches[:] if len(matches) >= 2 else matches'''
def extract_moves_and_timestamps(game_data):
    matches = re.findall(r'(\d+)\. \S+ \{\[%clk ([0-9:]+(?:\.[0-9]+)?)\]\}|(\d+)\.\.\. \S+ \{\[%clk ([0-9:]+(?:\.[0-9]+)?)\]\}', game_data)
    
    # Extract only move numbers and timestamps
    white_moves = [(move[0], move[1]) for move in matches if move[0] != '']
    black_moves = [(move[2], move[3]) for move in matches if move[2] != '']
    return white_moves, black_moves

def format_single_time_control(time_control):
    """
    Formats the time control into a user-friendly string.
    Handles cases with very short time controls like 1-second games or unlimited time.
    
    Args:
        time_control (str): Time control string in the format "base+increment" (e.g., "60+1").
    
    Returns:
        str: Formatted time control string.
    """
    if not time_control or not isinstance(time_control, str):
        return "Invalid time control"
    
    if time_control == "-":
        return "Unlimited"
    
    if '/' in time_control:
        # Handle Chess.com's daily format like "1/259200"
        parts = time_control.split('/')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            days = int(parts[0]) / (int(parts[1]) / 86400)  # Convert fraction to days
            return f"{days:.1f} day(s) per move"
        else:
            return f"Unrecognized format: {time_control}"
    
    try:
        if '+' in time_control:
            base, increment = map(int, time_control.split('+'))
        else:
            base, increment = int(time_control), 0
    except ValueError:
        return f"Unrecognized format: {time_control}"

    # Convert base time to minutes and seconds
    minutes = base // 60
    seconds = base % 60
    
    if minutes > 0:
        base_str = f"{minutes}m {seconds}s" if seconds > 0 else f"{minutes}m"
    else:
        base_str = f"{seconds}s"
    
    increment_str = f"{increment}s" if increment > 0 else "0s"

    return f"{base_str} + {increment_str}"
    

def convert_to_est(time_str):
    # Convert string time to a datetime object (using the date 1900-01-01 as it's irrelevant here)
    time = pd.to_datetime(time_str, format='%H:%M:%S')
    
    # Subtract 5 hours to convert from UTC to EST
    est_time = time - pd.Timedelta(hours=5)
    
    # Return the time in 'HH:MM:SS' format
    return est_time.strftime('%H:%M:%S')
    
def fetch_and_process_game_data(username, engine_path="/opt/homebrew/bin/stockfish"):
    all_games = fetch_all_games_for_last_year(username) ## change to current year if necessary
    game_data = []

    for game in all_games:

        if 'pgn' not in game:
            continue  # Skip this game if 'pgn' is missing (these are bughouse games or weird ass games)
            
        pgn = game['pgn']

        time_control = format_single_time_control(game['time_control'])
        
        metadata = extract_pgn_metadata(pgn)

        date = metadata.get('Date', None)
        
        start_time = metadata.get('StartTime', None)
        end_time = metadata.get('EndTime', None)

        start_time_est = convert_to_est(start_time)
        end_time_est = convert_to_est(end_time)
            
        link = metadata.get('Link', None)
        eco = metadata.get('ECOUrl', None)
        time_spent = calculate_time_difference(start_time, end_time)
        moves = extract_moves_from_pgn(pgn) 
        w_metamoves, b_metamoves = extract_moves_and_timestamps(pgn)
        
        data = {
            'url': game['url'],
            'pgn': pgn,
            'moves': moves,
            'date': date,
            'start_time_est': start_time_est,
            'end_time_est': end_time_est,
            'time_spent': time_spent,
            'link': link,
            'time_control': time_control,
            'time_class': game['time_class'],
            'game_type': game['rules'],
            'rated': game['rated'],
            'tcn': game['tcn'],
            'uuid': game['uuid'],
            'initial_setup': game['initial_setup'],
            'fen': game['fen'], #this is the position
            'rules': game['rules'],
            'white_username': game['white']['username'],
            'white_rating': game['white']['rating'],
            'white_result': game['white']['result'],
            'white_metamoves': w_metamoves,
            'black_username': game['black']['username'],
            'black_rating': game['black']['rating'],
            'black_result': game['black']['result'],
            'black_metamoves': b_metamoves,
            'eco': eco
        }
        game_data.append(data)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(game_data)
    
    return df



import pandas as pd
import numpy as np

def clean_dataframe(df, username):
    """
    Cleans and processes a chess DataFrame.

    Parameters:
        df (pd.DataFrame): The original DataFrame to clean.
        username (str): The username of the player to analyze.

    Returns:
        pd.DataFrame: The cleaned and processed DataFrame.
    """
    cleaned_df = df.copy()

    # Clean and process the 'eco' column
    cleaned_df['eco'] = cleaned_df['eco'].str.split('/openings/').str[1]

    def truncate_eco(name):
        if name is None:
            return 'No Opening'
        keywords = ["Defense", "Gambit", "Opening", "Game", "Attack", "System"]
        first_position = len(name)
        keyword_found = None
        for keyword in keywords:
            pos = name.find(keyword)
            if pos != -1 and pos < first_position:  # Find the earliest occurrence
                first_position = pos
                keyword_found = keyword
        # If a keyword was found, truncate up to the end of the keyword
        return name[:first_position + len(keyword_found)] if keyword_found else name
    
    # Define the function to update 'my_opening' based on color and 'Defense' presence
    def update_opening(row):
        if row['my_color'] == 'white':
            # If white, assign the ECO if it doesn't contain 'Defense'
            if 'Defense' not in row['eco']:
                return row['eco']
        elif row['my_color'] == 'black':
            # If black, assign the ECO if it contains 'Defense'
            if 'Defense' in row['eco']:
                return row['eco']
        return 'N/A'  # Otherwise, keep it as NaN

    cleaned_df['eco'] = cleaned_df['eco'].apply(truncate_eco)


    columns_to_initialize = [
        'my_username', 'my_rating', 'my_result', 'my_color', 'my_win_or_lose', 'my_metamoves',
        'opp_username', 'opp_rating', 'opp_result', 'opp_metamoves' #do i have to initialize my_color
    ]
    for column in columns_to_initialize:
        cleaned_df[column] = None

    for index, row in cleaned_df.iterrows():
        if row['white_username'].lower() == username.lower():
            cleaned_df.at[index, 'my_username'] = row['white_username']
            cleaned_df.at[index, 'my_rating'] = row['white_rating']
            cleaned_df.at[index, 'my_result'] = row['white_result']
            cleaned_df.at[index, 'my_color'] = 'white'
            cleaned_df.at[index, 'my_metamoves'] = row['white_metamoves']
            cleaned_df.at[index, 'opp_username'] = row['black_username']
            cleaned_df.at[index, 'opp_rating'] = row['black_rating']
            cleaned_df.at[index, 'opp_result'] = row['black_result']
            cleaned_df.at[index, 'opp_metamoves'] = row['black_metamoves']
            cleaned_df.at[index, 'my_win_or_lose'] = 'win' if row['white_result'] == 'win' else 'draw' if row['white_result'] in ['draw', 'stalemate', 'repetition', 'insufficient', 'timevsinsufficient', 'agreed'] else 'lose'

        elif row['black_username'].lower() == username.lower():
            cleaned_df.at[index, 'my_username'] = row['black_username']
            cleaned_df.at[index, 'my_rating'] = row['black_rating']
            cleaned_df.at[index, 'my_result'] = row['black_result']
            cleaned_df.at[index, 'my_color'] = 'black'
            cleaned_df.at[index, 'my_metamoves'] = row['black_metamoves']
            cleaned_df.at[index, 'opp_username'] = row['white_username']
            cleaned_df.at[index, 'opp_rating'] = row['white_rating']
            cleaned_df.at[index, 'opp_result'] = row['white_result']
            cleaned_df.at[index, 'opp_metamoves'] = row['white_metamoves']
            cleaned_df.at[index, 'my_win_or_lose'] = 'win' if row['black_result'] == 'win' else 'draw' if row['black_result'] in ['draw', 'stalemate', 'repetition', 'insufficient', 'timevsinsufficient', 'agreed'] else 'lose'
        else:
            print(f"BIG FUCKING Error: username '{username}' not found in either white or black columns at index {index}")
            print(f"White Username: {row['white_username']}, Black Username: {row['black_username']}, Game Link: {row['link']}")

    cleaned_df['my_opening'] = cleaned_df.apply(update_opening, axis=1)

    def time_string_to_seconds_with_fraction(time_str):
        # Split the string into hours, minutes, and seconds
        h, m, s = map(float, time_str.split(':'))
        # Calculate total seconds, including fractional seconds
        return h * 3600 + m * 60 + s

    def convert_time_class_to_seconds(time_class):
        """
        Converts a time control string into total seconds.
        Handles formats like '5+5' (base + increment), '5m + 0s', '8m 20s + 0s', and Chess.com's daily formats like '1/259200'.
        
        Args:
            time_class (str): The time control string.
        
        Returns:
            float or str: Total seconds for the base time or "Weird" if unrecognized.
        """
        if '+' in time_class:
            try:
                # Split by the '+' symbol into base time and increment time
                base_part, increment_part = time_class.split('+')

                # Handle base part (minutes and/or seconds)
                base_minutes = 0
                base_seconds = 0
                if 'm' in base_part and 's' in base_part:
                    # Handle case where both minutes and seconds are in the base part (e.g., '8m 20s')
                    parts = base_part.split()
                    for part in parts:
                        if 'm' in part:
                            base_minutes = int(part.replace('m', '').strip())
                        elif 's' in part:
                            base_seconds = int(part.replace('s', '').strip())
                    base_seconds += base_minutes * 60  # Convert minutes to seconds
                elif 'm' in base_part:
                    base_minutes = int(base_part.replace('m', '').strip())
                    base_seconds = base_minutes * 60
                elif 's' in base_part:
                    base_seconds = int(base_part.replace('s', '').strip())
                else:
                    base_seconds = int(base_part.strip())

                # Handle increment part (minutes or seconds)
                increment_seconds = 0
                if 'm' in increment_part:
                    increment_minutes = int(increment_part.replace('m', '').strip())
                    increment_seconds = increment_minutes * 60
                elif 's' in increment_part:
                    increment_seconds = int(increment_part.replace('s', '').strip())
                else:
                    increment_seconds = int(increment_part.strip())

                # Return total time in seconds
                return base_seconds + increment_seconds
            except ValueError:
                return "Weird"
        
        # Handle Chess.com's daily format like '1/259200'
        if '/' in time_class:
            parts = time_class.split('/')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                base_days = int(parts[0]) / (int(parts[1]) / 86400)  # Convert fraction to days in seconds
                return base_days * 86400
            else:
                return "Weird"
        
        # Return "Weird" if no valid format is matched
        return "Weird"


    # Calculate time-related columns
    cleaned_df['my_time_left'] = np.where(
        cleaned_df['my_result'].isin(['timeout', 'timevsinsufficient']), 0, 
        cleaned_df['my_metamoves'].apply(lambda x: time_string_to_seconds_with_fraction(x[-1][1]) if x and len(x[-1]) > 1 else 0) #account for empty 
    )

    cleaned_df['opp_time_left'] = np.where(
        cleaned_df['opp_result'].isin(['timeout', 'timevsinsufficient']), 0, 
        cleaned_df['opp_metamoves'].apply(lambda x: time_string_to_seconds_with_fraction(x[-1][1]) if x and len(x[-1]) > 1 else 0) #account for empty 
    )
    
    cleaned_df['my_num_moves'] = cleaned_df['my_metamoves'].apply(lambda x: len(x) if x else 0)


        # Add a safe calculation for ratios, only if time_control is valid
    def safe_time_left_ratio(row, col_name):
        
        time_control_seconds = convert_time_class_to_seconds(row['time_control'])
        if time_control_seconds == "Weird" or time_control_seconds == 0:
            print(f"Printing Row: {row['link']} and its time_control {row['time_control']} and also print time_control_seconds {time_control_seconds}")
            return np.nan  # Return NaN for invalid or unrecognized time controls
        return row[col_name] / time_control_seconds
    
    cleaned_df['my_time_left_ratio'] = cleaned_df.apply(
        lambda row: safe_time_left_ratio(row, 'my_time_left'),
        axis=1
    )
    
    cleaned_df['opp_time_left_ratio'] = cleaned_df.apply(
        lambda row: safe_time_left_ratio(row, 'opp_time_left'),
        axis=1
    )


    # Split moves based on player color
    def split_moves(row):
        moves = row['moves']
        if row['my_color'] == 'white':
            my_moves = moves[::2]  # Take odd positions (0-indexed: 0, 2, 4, ...)
            opp_moves = moves[1::2]  # Take even positions (1-indexed: 1, 3, 5, ...)
        else:
            my_moves = moves[1::2]  # Take even positions (1-indexed: 1, 3, 5, ...)
            opp_moves = moves[::2]  # Take odd positions (0-indexed: 0, 2, 4, ...)
        return my_moves, opp_moves

    # Track castling
    def track_castling(my_moves):
        if 'O-O-O' in my_moves:
            return 'queenside'
        elif 'O-O' in my_moves:
            return 'kingside'
        else:
            return 'none'

    # Apply the functions to the dataframe
    cleaned_df[['my_moves', 'opp_moves']] = cleaned_df.apply(
        lambda row: pd.Series(split_moves(row)), axis=1
    )
    cleaned_df['my_castling'] = cleaned_df['my_moves'].apply(track_castling)
    cleaned_df['opp_castling'] = cleaned_df['opp_moves'].apply(track_castling)

    def count_promotions(my_moves):
        """Counts the number of promotions in the given list of moves."""
        return sum('=' in move for move in my_moves)

    def count_en_passant(my_moves):
        """Counts the number of en passant occurrences in the given list of moves."""
        return sum('e.p.' in move for move in my_moves)

    cleaned_df['en_passant_count'] = cleaned_df['my_moves'].apply(count_en_passant)
    cleaned_df['promotion_count'] = cleaned_df['my_moves'].apply(count_promotions)

    def get_country(username):
        # Would be cool if I found out all the opponents that you played that were streamers, titled, or country
        # cleaned_df['country'] = cleaned_df['opp_username'].apply(get_country) takes 162 seconds lol way too long

        """Gets the country of the opponent"""
        opp_data = fetch_player_data(username)
        #print(opp_data['country'])
        return opp_data['country']
    
    return cleaned_df




