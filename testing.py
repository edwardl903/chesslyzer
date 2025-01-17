from data_processing import *
from my_stats import *
from visualizations import *

import time
import sys
import json
import os
import pandas as pd

pd.set_option('display.max_rows', None)  # No limit on rows

def drop_columns(df):
    if 'pgn' in df.columns:
        df = df.drop(columns=['pgn'])
        print(df.columns)
    return df
    

def main(username):
    print("Chesslyzer booting...!!!")
    
    output_dir = 'static/images/'
    os.makedirs(output_dir, exist_ok=True)

    # Start overall timer
    overall_start = time.time()

    # Time fetch_and_process_game_data function
    start_time = time.time()
    df = fetch_and_process_game_data(username)
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
    with open(f'json/{username}_statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)

    start_time = time.time()
    call_visualizations(chess_df, output_dir)
    visualization_duration = time.time() - start_time
    print(f"Generating visualizations took {visualization_duration:.2f} seconds")

    start_time = time.time()
    final_df = drop_columns(chess_df)
    final_df.to_csv(f'csv/{username}.csv', index=False)
    saving_duration = time.time() - start_time
    print(f"Saving CSV file took {saving_duration:.2f} seconds")

    end_time = time.time()
    total_duration = end_time - overall_start
    print(f"Total execution time: {total_duration:.2f} seconds")

# Boilerplate to run main when executed directly (for testing or debugging)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        username = sys.argv[1]  # Get username from command line arguments
        main(username)
    else:
        print("Usage: python chesslytics.py <username>")
