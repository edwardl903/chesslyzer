from data_processing import *
from my_stats import *

import time
import sys
import json

pd.set_option('display.max_rows', None)  # No limit on rows
def drop_columns(df):
    if 'pgn' in df.columns:
        df = df.drop(columns=['pgn'])
        print(df.columns)
    return df
    
def main(username):
    print("Chesslyzer booting...!!!")
    
    start_time = time.time()
    df = fetch_and_process_game_data(username)
    metadata_df = clean_dataframe(df, username)
    print(f"Columns: {metadata_df.columns}")
    #print(metadata_df['time_class'].unique())
    chess_df = metadata_df[(metadata_df['rules'] == 'chess') & (metadata_df['time_class'].isin(['blitz', 'rapid', 'bullet', 'daily']))]
    print(chess_df.isna().sum()) #lol kills the daily games, bug atm
    # Drop rows with NaN values (if you want to remove them)
    chess_df = chess_df.dropna()

    statistics = total_statistics(chess_df)

    
    # Concatenate all dataframes along the columns (axis=1)
    #final_stats_df = pd.concat([stats_df, more_stats_df, flag_stats_df], axis=1)
    #print("Statistics combined into one dataframe:")
    #print(final_stats_df)

    
    # Assuming 'statistics' is a dictionary
    with open(f'json/{username}_statistics.json', 'w') as f:
        json.dump(statistics, f, indent=4)
    #for stat, value in flag_statistics.items():
        #print(f"{stat}: {value}")
    #for stat, value in more_statistics.items():
        #print(f"{stat}: {value}")

    final_df = drop_columns(chess_df)
    final_df.to_csv(f'csv/{username}.csv', index=False)

        
    testing_df = final_df.copy()
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Execution time: {duration:.2f} seconds")

# Boilerplate to run main when executed directly (for testing or debugging)
if __name__ == "__main__":
    if len(sys.argv) > 1:
        username = sys.argv[1]  # Get username from command line arguments
        main(username)
    else:
        print("Usage: python chesslytics.py <username>")
        #username = "joebruin"
        #main(username)