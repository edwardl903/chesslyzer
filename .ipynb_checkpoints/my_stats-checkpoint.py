def total_statistics(cleaned_df):
    # Count the occurrences of each type of castling
    castling_counts = cleaned_df['my_castling'].value_counts()

    # Store the counts in variables
    kingside_castles = castling_counts.get('kingside', 0)
    queenside_castles = castling_counts.get('queenside', 0)
    no_castles = castling_counts.get('none', 0)

    # Count the result occurrences
    result_counts = cleaned_df['my_result'].value_counts()
    total_games = len(cleaned_df)
    total_win = result_counts.get('win', 0)
    total_checkmated = result_counts.get('checkmated', 0)
    total_timeout = result_counts.get('timeout', 0)
    total_timevsinsufficient = result_counts.get('timevsinsufficient', 0)
    total_resigned = result_counts.get('resigned', 0)
    total_stalemate = result_counts.get('stalemate', 0)
    total_repetition = result_counts.get('repetition', 0)
    total_abandoned = result_counts.get('abandoned', 0)
    total_insufficient = result_counts.get('insufficient', 0)

    total_draw = total_stalemate + total_repetition + total_timevsinsufficient + total_insufficient
    total_loss = total_checkmated + total_timeout + total_resigned + total_abandoned
    total_moves = cleaned_df['my_num_moves'].sum()
    total_time_spent = cleaned_df['time_spent'].sum()
    #total_thinking_time_spent = cleaned_df['']

    # Sum the total occurrences of en passant and promotions
    total_en_passant = cleaned_df['en_passant_count'].sum()
    total_promotions = cleaned_df['promotion_count'].sum()

    # Prepare the statistics dictionary with total counts only
    statistics = {
        'kingside_castles': kingside_castles,
        'queenside_castles': queenside_castles,
        'no_castles': no_castles,
        'total_time_spent': total_time_spent,
        'total_moves': total_moves,
        'total_win': total_win,
        'total_draw': total_draw,
        'total_loss': total_loss,
        'total_checkmated': total_checkmated,
        'total_timeout': total_timeout,
        'total_timevsinsufficient': total_timevsinsufficient,
        'total_resigned': total_resigned,
        'total_stalemate': total_stalemate,
        'total_repetition': total_repetition,
        'total_abandoned': total_abandoned,
        'total_insufficient': total_insufficient,
        'total_en_passant': total_en_passant,
        'total_promotions': total_promotions,
        'total_games': total_games
    }

    return statistics



import pandas as pd
from itertools import groupby

# Function to calculate win rates by opening
def calculate_win_rates(df):
    win_rates = (
        df.groupby('eco')['my_win_or_lose']
        .apply(lambda x: (x == 'win').sum() / len(x))
        .reset_index(name='win_rate')
    )
    return win_rates

# Function to count games played by opening
def count_games_played(df):
    games_played = df['eco'].value_counts().reset_index()
    games_played.columns = ['eco', 'Games_Played']
    return games_played

# Function to merge win rates and games played
def merge_opening_stats(win_rates, games_played):
    openings_stats = win_rates.merge(games_played, on='eco')
    return openings_stats

# Function to calculate win/loss/draw percentages
def calculate_win_loss_draw_percentages(df):
    total_games = len(df)
    win_percentage = (df['my_win_or_lose'] == 'win').mean() * 100
    loss_percentage = (df['my_win_or_lose'] == 'lose').mean() * 100
    draw_percentage = (df['my_win_or_lose'] == 'draw').mean() * 100
    return win_percentage, loss_percentage, draw_percentage

# Function to calculate Elo progression
def calculate_elo_progression(df):
    df['date'] = pd.to_datetime(df['date'])
    elo_progression = df.groupby('date')['my_rating'].mean()
    return elo_progression

# Function to calculate the longest streak
def longest_streak(results, target):
    streak = max((len(list(g)) for k, g in groupby(results) if k == target), default=0)
    return streak

# Function to find the quickest checkmate
def find_quickest_checkmate(df):
    df['num_moves'] = df['moves'].apply(len)
    quickest_win = df.loc[(df['my_win_or_lose'] == 'win') & (df['num_moves'] == df['num_moves'].min())]
    return quickest_win

# Function to find best game by rating difference
def find_best_game_by_rating(df):
    df['rating_diff'] = df['my_rating'] - df['opp_rating']
    best_game = df.loc[(df['my_win_or_lose'] == 'win') & (df['rating_diff'] == df['rating_diff'].max())]
    return best_game

# Function to calculate total time spent
def calculate_total_time_spent(df):
    total_time_spent = df['time_spent'].sum() / 3600  # Convert seconds to hours
    return total_time_spent

# Function to calculate most played opponent
def most_played_opponent(df):
    most_played_opponent = df['opp_username'].value_counts().reset_index()
    most_played_opponent.columns = ['Opponent', 'Games_Played']
    return most_played_opponent



# Function to collect statistics into a dictionary
def collect_statistics(df):
    stats = {}

    # Opening stats
    win_rates = calculate_win_rates(df)
    games_played = count_games_played(df)
    openings_stats = merge_opening_stats(win_rates, games_played)
    most_played_openings = openings_stats.sort_values(by='Games_Played', ascending=False)
    stats['most_played_openings'] = most_played_openings.head()

    # Win/Loss/Draw percentages
    win_percentage, loss_percentage, draw_percentage = calculate_win_loss_draw_percentages(df)
    stats['win_loss_draw_percentages'] = {
        'win_percentage': win_percentage,
        'loss_percentage': loss_percentage,
        'draw_percentage': draw_percentage
    }

    # Streaks
    winning_streak = longest_streak(df['my_win_or_lose'], 'win')
    losing_streak = longest_streak(df['my_win_or_lose'], 'lose')
    stats['streaks'] = {
        'winning_streak': winning_streak,
        'losing_streak': losing_streak
    }

    # Quickest checkmate
    quickest_win = find_quickest_checkmate(df)
    stats['quickest_checkmate'] = quickest_win[['date', 'moves', 'my_rating', 'opp_rating', 'my_num_moves', 'link']]

    # Total time spent
    total_time_spent = calculate_total_time_spent(df)
    stats['total_time_spent'] = total_time_spent

    # Best game by rating difference
    best_game = find_best_game_by_rating(df)
    stats['best_game_by_rating'] = best_game[['date', 'moves', 'my_rating', 'opp_rating', 'my_num_moves', 'link']]

    # Most played opponent
    most_played_opponent_df = most_played_opponent(df)
    stats['most_played_opponent'] = most_played_opponent_df.head()

    return stats


def get_flag_statistics(cleaned_df):
    # Filter for wins where 'my_win_or_lose' is 'win' (games where you flagged the opponent)
    my_filtered_wins = cleaned_df[
        (cleaned_df['my_win_or_lose'] == 'win') &
        (cleaned_df['my_time_left'] >= 0.1) &
        (cleaned_df['my_time_left'] <= 5)
    ]

    top_10_flagged_games = my_filtered_wins.sort_values(by='opp_time_left').head(10)

    opp_filtered_wins = cleaned_df[
        (cleaned_df['my_win_or_lose'] == 'win') &
        (cleaned_df['opp_time_left'] >= 0.1) &
        (cleaned_df['opp_time_left'] <= 5)
    ]

    # Sort by 'my_time_left' (ascending) to get the games where you were flagged the most
    top_10_get_flagged_games = opp_filtered_wins.sort_values(by='my_time_left').head(10)

    # Group by relevant columns and get counts for both filtered dataframes
    my_flag_counts = my_filtered_wins.groupby(['my_time_left', 'my_win_or_lose', 'link']).size().sort_index()
    opp_flag_counts = opp_filtered_wins.groupby(['my_time_left', 'my_win_or_lose', 'link']).size().sort_index()

    # Store statistics in a dictionary
    flag_statistics = {
        "my_total_flag_wins": len(my_filtered_wins),
        "my_total_flag_losses": len(opp_filtered_wins),
        "top_10_flagged_games": top_10_flagged_games[['link', 'opp_time_left', 'my_time_left', 'my_win_or_lose', 'my_color']].to_dict(orient='records'),
        "top_10_get_flagged_games": top_10_get_flagged_games[['link', 'opp_time_left', 'my_time_left', 'my_win_or_lose', 'my_color']].to_dict(orient='records'),
        #"my_flag_counts": my_flag_counts.to_dict(),
        #"opp_flag_counts": opp_flag_counts.to_dict()
    }

    return flag_statistics