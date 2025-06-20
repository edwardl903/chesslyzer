import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data.processor import *
from stats.analysis import *
from visualizations.plots import *
from data.uploader import *



import time
import sys
import json
import os
import pandas as pd
from datetime import datetime

pd.set_option('display.max_rows', None)  # No limit on rows

def drop_columns(df):
    if 'pgn' in df.columns:
        df = df.drop(columns=['pgn'])
        print(df.columns)
    return df
    

def main(username, year):
    print("Chesslyzer booting...!!!")
    
    output_dir = 'static/images/'
    os.makedirs(output_dir, exist_ok=True)

    # Start overall timer
    overall_start = time.time()

    # Time fetch_and_process_game_data function
    start_time = time.time()
    df = fetch_and_process_game_data(username, year)
    fetch_process_duration = time.time() - start_time
    print(f"Fetching and processing data took {fetch_process_duration:.2f} seconds")

    # Time clean_dataframe function
    start_time = time.time()
    metadata_df = clean_dataframe(df, username)
    clean_duration = time.time() - start_time
    print(f"Cleaning data took {clean_duration:.2f} seconds")

    print(f"Columns: {metadata_df.columns}")
    if metadata_df.empty:
        return []  # Return an empty DataFrame

    start_time = time.time()
    chess_df = metadata_df[(metadata_df['rules'] == 'chess') & (metadata_df['time_class'].isin(['blitz', 'rapid', 'bullet', 'daily']))]
    filter_duration = time.time() - start_time
    print(f"Filtering data took {filter_duration:.2f} seconds")

    print(chess_df.isna().sum())

    start_time = time.time()
    chess_df = chess_df.dropna()
    dropna_duration = time.time() - start_time
    print(f"Dropping NaN values took {dropna_duration:.2f} seconds")

    start_time = time.time()
    statistics = total_statistics(chess_df)
    statistics_duration = time.time() - start_time
    print(f"Calculating statistics took {statistics_duration:.2f} seconds")

    # Assuming 'statistics' is a dictionary
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    json_dir = os.path.join(project_root, 'data', 'json')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f'{username}_statistics.json')
    with open(json_path, 'w') as f:
        json.dump(statistics, f, indent=4)

    start_time = time.time()
    call_visualizations(chess_df, output_dir)
    visualization_duration = time.time() - start_time
    print(f"Generating visualizations took {visualization_duration:.2f} seconds")

    start_time = time.time()
    final_df = drop_columns(chess_df)
    csv_dir = os.path.join(project_root, 'data', 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'{username}.csv')
    final_df.to_csv(csv_path, index=False)
    saving_duration = time.time() - start_time
    print(f"Saving CSV file took {saving_duration:.2f} seconds")

    # Example for Excel output
    excel_path = os.path.join(project_root, 'data', f'{username}.xlsx')
    final_df.to_excel(excel_path, index=False)

    # Example for DataFrame JSON output
    df_json_dir = os.path.join(project_root, 'data', 'json')
    os.makedirs(df_json_dir, exist_ok=True)
    df_json_path = os.path.join(df_json_dir, f'{username}.df.json')
    final_df.to_json(df_json_path, orient='records', indent=4)

    # Upload to BigQuery

    nested_cols = ['white_metamoves', 'black_metamoves', 'my_metamoves', 'opp_metamoves']
    final_df = final_df.drop(columns=nested_cols, errors='ignore')

    important_columns = [
    'url', 'uuid', 'date', 'timestamp',
    'time_control', 'time_class', 'game_type', 'rated', 'eco', 'my_opening',
    'my_username', 'my_rating', 'my_result', 'my_color',
    'opp_username', 'opp_rating', 'opp_result',
    'my_win_or_lose', 'rating_diff',
    'my_time_left', 'opp_time_left', 'my_time_left_ratio', 'opp_time_left_ratio', 'time_spent',
    'my_moves', 'opp_moves', 'moves', 'my_num_moves',
    'en_passant_count', 'promotion_count', 'my_castling', 'opp_castling',
    'month', 'weekday', 'hour', 'day_of_week'
    ]

    bigquery_df = final_df[important_columns]

    # Add deduplication logic to prevent duplicates when same user generates data multiple times
    bigquery_df['unique_id'] = bigquery_df['uuid']  # uuid is already unique per game
    bigquery_df['username_year'] = f"{username}_{year}"  # Add username_year for filtering
    
    # Add upload timestamp
    bigquery_df['uploaded_at'] = datetime.now()
    
    # Drop duplicates based on uuid (same game shouldn't be uploaded twice)
    bigquery_df = bigquery_df.drop_duplicates(subset=['uuid'], keep='last')
    
    print(f"📊 Uploading {len(bigquery_df)} unique games for {username} ({year})")

    sanitized_username = ''.join(c if c.isalnum() or c == '_' else '_' for c in username)
    # table_id = f"crucial-decoder-462021-m4.test_table.{sanitized_username}_{year}"
    table_id = "crucial-decoder-462021-m4.test1.megachessdataset"
    credentials_path = "gcp/service_account.json"
    # Use MERGE for efficient upserts without duplicates
    merge_to_bigquery_table(table_id, bigquery_df, unique_columns=['uuid'], credentials_path=credentials_path)

    end_time = time.time()
    total_duration = end_time - overall_start
    print(f"Total execution time: {total_duration:.2f} seconds")



# Boilerplate to run main when executed directly (for testing or debugging)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        username = sys.argv[1]  # Get username from command line arguments
        year = sys.argv[2]
        main(username, year)
    else:
        print("Usage: python testing.py <username> <year>")
