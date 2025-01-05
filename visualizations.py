
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler


def plot_game_statistics(cleaned_df, output_dir):
    """
    This function processes the cleaned dataframe and generates three plots: 
    - Frequency of games played over months
    - Frequency of games played by weekday
    - Frequency of games played by hour of day
    """
    # Ensure that 'timestamp' column is a datetime object
    cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['date'])

    # Extract the month and year from the 'timestamp' column
    cleaned_df['month'] = cleaned_df['timestamp'].dt.to_period('M')

    # Extract the weekday (0 = Monday, 6 = Sunday)
    cleaned_df['weekday'] = cleaned_df['timestamp'].dt.day_name()

    # Extract the hour of the day from 'start_time_est'
    cleaned_df['hour'] = cleaned_df['start_time_est'].str.split(':').str[0].astype(int)

    # 1. Count the frequency of games played each month
    monthly_counts = cleaned_df['month'].value_counts().sort_index()

    # 2. Count the frequency of games played by weekday
    weekday_counts = cleaned_df['weekday'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # 3. Count the frequency of games played by hour
    hourly_counts = cleaned_df['hour'].value_counts().sort_index()

    # Plotting the frequency of games over months
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_counts.index.astype(str), y=monthly_counts.values)
    plt.title('Frequency of Games Played Over Months')
    plt.xlabel('Month')
    plt.ylabel('Frequency of Games')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.savefig(f"{output_dir}image_1.png")
    #plt.show()

    # Plotting the frequency of games by weekday
    plt.figure(figsize=(10, 6))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values)
    plt.title('Frequency of Games Played by Weekday')
    plt.xlabel('Weekday')
    plt.ylabel('Frequency of Games')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}image_2.png")
    #plt.show()

    # Plotting the frequency of games by hour
    plt.figure(figsize=(10, 6))
    sns.barplot(x=hourly_counts.index.astype(str), y=hourly_counts.values)
    plt.title('Frequency of Games Played by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Frequency of Games')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}image_3.png")
    #plt.show()




# In[ ]:





# In[14]:


def plot_opening_statistics(cleaned_df, output_dir):
    # Group by 'eco' and 'my_win_or_lose' to count occurrences

    print("Checking for NaN values...")
    print(f"NaN values in 'my_time_left': {cleaned_df['my_time_left'].isna().sum()}")
    print(f"NaN values in 'my_win_or_lose': {cleaned_df['my_win_or_lose'].isna().sum()}")
    print(f"NaN values in 'time_class': {cleaned_df['time_class'].isna().sum()}")
    
    # Print unique values for each column to inspect the data
    '''print("\nUnique values in 'my_time_left':")
    print(cleaned_df['my_time_left'].unique())
    
    print("\nUnique values in 'my_win_or_lose':")
    print(cleaned_df['my_win_or_lose'].unique())
    
    print("\nUnique values in 'time_class':")
    print(cleaned_df['time_class'].unique())'''

    
    opening_stats = cleaned_df.groupby(['eco', 'my_win_or_lose']).size().unstack(fill_value=0)

    # Calculate total games per opening
    opening_stats['total'] = opening_stats.sum(axis=1)

    # Sort by the total number of games played and keep only the top 10
    top_10_openings = opening_stats.sort_values(by='total', ascending=False).head(10)

    # Normalize the stats to get win, draw, and loss rates
    top_10_openings_normalized = top_10_openings.div(top_10_openings['total'], axis=0).drop(columns=['total'])

    # Ensure 'win', 'draw', and 'lose' columns exist
    for col in ['win', 'draw', 'lose']:
        if col not in top_10_openings_normalized.columns:
            top_10_openings_normalized[col] = 0

    # Plot the stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for win, draw, and lose
    color_map = {'win': 'green', 'draw': 'gray', 'lose': 'red'}

    # Plot each segment
    bottom = None
    for outcome in ['win', 'draw', 'lose']:
        if bottom is None:
            bottom = top_10_openings_normalized[outcome]
            ax.bar(top_10_openings.index, top_10_openings_normalized[outcome], label=outcome.capitalize(), color=color_map[outcome])
        else:
            ax.bar(top_10_openings.index, top_10_openings_normalized[outcome], bottom=bottom, label=outcome.capitalize(), color=color_map[outcome])
            bottom += top_10_openings_normalized[outcome]

    # Add game counts as labels above the bars
    for idx, total_games in enumerate(top_10_openings['total']):
        ax.text(idx, 1.02, f'{int(total_games)} games', ha='center', fontsize=10, color='black')

    # Customize the plot
    plt.title('Win/Draw/Loss Rates for Top 10 Most Common Openings (with Game Counts)', pad=50)
    plt.xlabel('Opening')
    plt.ylabel('Rate')
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Rotate x-axis labels for readability
    plt.legend(title='Outcome', loc='upper right')
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(f"{output_dir}image_4.png")
    #plt.show()



# In[15]:



def plot_time_control_statistics(cleaned_df, output_dir):
    # Get unique time_controls
    time_controls = cleaned_df['time_control'].unique()

    # Set up the number of columns (e.g., 2 columns per row)
    n_cols = 2  # Set number of columns to 2 (you can adjust this)
    n_rows = (len(time_controls) + n_cols - 1) // n_cols  # Calculate number of rows dynamically

    # Set up the subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 8))
    axes = axes.flatten()  # Flatten the axes array to make indexing easier

    # Define custom colors: Green for win, Red for loss, Gray for draw
    color_map = {'win': 'green', 'lose': 'red', 'draw': 'gray'}

    # Loop through each time_control and plot the pie chart
    for i, time_control in enumerate(time_controls):
        # Filter the DataFrame for the current time_control
        time_control_data = cleaned_df[cleaned_df['time_control'] == time_control]

        # Count the occurrences of 'my_win_or_lose' for that time_control
        win_loss_counts = time_control_data['my_win_or_lose'].value_counts()


        # Ensure 'win', 'lose', and 'draw' are included even if missing
        for result in ['win', 'lose', 'draw']:
            if result not in win_loss_counts:
                win_loss_counts[result] = 0

        # Sort by expected order
        win_loss_counts = win_loss_counts[['win', 'lose', 'draw']]

        # Skip empty data (all counts zero)
        if win_loss_counts.sum() == 0:
            print(f"Debugging: Skipping {time_control} as all counts are zero.")
            continue

        # Define colors based on outcomes
        colors = [color_map[result] for result in win_loss_counts.index]

        # Create labels with both count and percentage
        total_games = win_loss_counts.sum()
        labels = [f'{result} - {count} games' for result, count in zip(win_loss_counts.index, win_loss_counts)]
        label_text = f'Total: {total_games} games'

        # Plot a pie chart for the current time_control
        ax = axes[i]  # Use the i-th axis in the flattened axes array
        wedges, texts, autotexts = ax.pie(
            win_loss_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors
        )

        # Set title and aspect ratio
        ax.set_title(f'{time_control} Win/Loss/Draw Distribution')
        ax.axis('equal')  # Ensures pie chart is circular

        # Add total games label below the pie chart
        ax.text(0, -1.4, label_text, ha='center', va='center', fontsize=12, color='black')

    # Hide unused subplots (for empty time_controls)
    for j in range(len(time_controls), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save and display the pie charts
    print("Debugging: Saving and displaying the pie charts.")
    #plt.savefig(f"{output_dir}image_5.png")
    #plt.show()





# In[16]:



def plot_time_class_statistics(cleaned_df, output_dir):
    # Get unique time_classes
    time_classes = cleaned_df['time_class'].unique()

    # Set up the number of rows and columns for the subplots
    n_cols = len(time_classes)  # Create one subplot per time_class
    n_rows = 1  # All in one row
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, 8))  # Adjust figsize as needed

    # Define new color palette
    win_color = '#66b3ff'  # Soft blue for win
    lose_color = '#ff6666'  # Soft red for loss
    draw_color = '#ffcc99'  # Soft orange for draw

    # Loop through each time_class and plot the pie chart
    for i, time_class in enumerate(time_classes):
        # Filter the DataFrame for the current time_class
        time_class_data = cleaned_df[cleaned_df['time_class'] == time_class]
        
        # Count the occurrences of 'my_win_or_lose' for that time_class (including 'draw')
        win_loss_counts = time_class_data['my_win_or_lose'].value_counts()

        # Make sure all outcomes (win, loss, draw) are represented
        outcomes = ['win', 'lose', 'draw']
        win_loss_counts = win_loss_counts.reindex(outcomes, fill_value=0)

        # Define custom colors (subtle, elegant shades)
        colors = [win_color, lose_color, draw_color]
        
        # Create labels with both count and percentage
        total_games = len(time_class_data)
        labels = [f'{result} - {count} games' for result, count in zip(win_loss_counts.index, win_loss_counts)]
        # Add total games count at the bottom of the chart
        label_text = f'Total: {total_games} games'
        
        # Plot a pie chart for the current time_class
        ax = axes[i] if n_cols > 1 else axes  # Adjust axes indexing based on the number of columns
        wedges, texts, autotexts = ax.pie(win_loss_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        
        # Set title  
        ax.set_title(f'{time_class} Win/Loss/Draw Distribution')
        
        # Adjust the label text position for the pie chart
        ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
        
        # Add total games label below the pie chart
        ax.text(0, -1.4, label_text, ha='center', va='center', fontsize=12, color='black')

    # Adjust layout for better spacing between charts
    plt.tight_layout()

    # Save the pie charts as an image
    plt.savefig(f"{output_dir}image_5.png")
    #plt.show()


# In[ ]:





# In[ ]:





# In[17]:


def plot_game_outcome_by_hour(cleaned_df, output_dir):
    # Filter rows for wins, losses, and draws
    wins_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].copy()
    losses_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'lose'].copy()
    draws_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'draw'].copy()

    # Convert 'start_time_est' to datetime and extract the hour
    wins_data['start_time_est'] = pd.to_datetime(wins_data['start_time_est'], format='%H:%M:%S')
    losses_data['start_time_est'] = pd.to_datetime(losses_data['start_time_est'], format='%H:%M:%S')
    draws_data['start_time_est'] = pd.to_datetime(draws_data['start_time_est'], format='%H:%M:%S')

    # Extract the hour from the 'start_time_est' column
    wins_data['hour'] = wins_data['start_time_est'].dt.hour
    losses_data['hour'] = losses_data['start_time_est'].dt.hour
    draws_data['hour'] = draws_data['start_time_est'].dt.hour

    # Group by hour and count the wins, losses, and draws
    hourly_wins = wins_data.groupby('hour').size()
    hourly_losses = losses_data.groupby('hour').size()
    hourly_draws = draws_data.groupby('hour').size()

    # Combine the wins, losses, and draws into a single DataFrame for easier plotting
    hourly_counts = pd.DataFrame({
        'wins': hourly_wins,
        'losses': hourly_losses,
        'draws': hourly_draws
    }).fillna(0)  # Fill missing values with 0 (some hours may have no wins, losses, or draws)

    # Plot the side-by-side bar chart with improved color scheme
    ax = hourly_counts.plot(kind='bar', width=0.8, color=['#2ca02c', '#ff7f0e', '#1f77b4'], position=0)

    # Customize the plot for better visual appeal
    plt.title('Wins, Losses, and Draws by Hour of the Day', fontsize=16)
    plt.xlabel('Hour of the Day', fontsize=12)
    plt.ylabel('Number of Wins, Losses, and Draws', fontsize=12)
    plt.xticks(range(0, 24), fontsize=10)  # Show every hour on the x-axis
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    #plt.savefig(f"{output_dir}GameOutcome_HourofDay1.png")
    #plt.show()

    # Calculate total games and win percentages for each hour
    hourly_totals = hourly_wins.add(hourly_losses, fill_value=0).add(hourly_draws, fill_value=0)
    hourly_win_percentage = (hourly_wins / hourly_totals) * 100

    # Create the plot with dual axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the win percentage on the first axis as a bar chart
    ax1.bar(hourly_win_percentage.index, hourly_win_percentage, width=0.4, color='#1f77b4', label='Win Percentage')
    ax1.set_xlabel('Hour of the Day (EST)', fontsize=12)
    ax1.set_ylabel('Win Percentage (%)', color='#1f77b4', fontsize=12)
    ax1.set_ylim(0, 100)  # Percentage ranges from 0 to 100
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_xticks(range(0, 24))  # Show every hour on the x-axis
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Create a second axis to plot the total games (record count) as a line chart
    ax2 = ax1.twinx()
    ax2.plot(hourly_totals.index, hourly_totals, color='#ff6347', marker='o', label='Total Games', linewidth=2)
    ax2.set_ylabel('Total Games (Record Count)', color='#ff6347', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#ff6347')

    # Add a title and display the plot
    plt.title('Win Percentage, Total Games, and Draws by Hour of the Day', fontsize=16)
    fig.tight_layout()  # Ensure everything fits without overlap

    # Show the plot
    plt.savefig(f"{output_dir}image_6.png")
    #plt.show()


# In[18]:


def plot_game_outcome_by_day(cleaned_df, output_dir):
    # Ensure 'date' is in datetime format
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Filter rows for wins, losses, and draws
    wins_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].copy()
    losses_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'lose'].copy()
    draws_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'draw'].copy()

    # Extract the day of the week from the 'date' column
    wins_data['day_of_week'] = wins_data['date'].dt.day_name()  # Get day name (e.g., 'Monday')
    losses_data['day_of_week'] = losses_data['date'].dt.day_name()  # Get day name (e.g., 'Monday')
    draws_data['day_of_week'] = draws_data['date'].dt.day_name()  # Get day name (e.g., 'Monday')

    # Group by day of the week and count the wins, losses, and draws
    hourly_wins = wins_data.groupby('day_of_week').size()
    hourly_losses = losses_data.groupby('day_of_week').size()
    hourly_draws = draws_data.groupby('day_of_week').size()

    # Reorder days of the week to ensure proper ordering in the plot
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Create a DataFrame combining wins, losses, and draws
    hourly_counts = pd.DataFrame({
        'wins': hourly_wins,
        'losses': hourly_losses,
        'draws': hourly_draws
    }).fillna(0)  # Fill missing values with 0 (some days may have no wins, losses, or draws)

    # Reorder the rows based on the days of the week to ensure the correct order
    hourly_counts = hourly_counts.reindex(days_order)

    # Plot the side-by-side bar chart with improved color scheme
    ax = hourly_counts.plot(kind='bar', width=0.8, color=['#2ca02c', '#ff7f0e', '#1f77b4'], position=0)

    # Customize the plot for better visual appeal
    plt.title('Wins, Losses, and Draws by Day of the Week', fontsize=16)
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Number of Wins, Losses, and Draws', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels for readability
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_7.png")
    #plt.show()


# In[19]:


def plot_game_outcome_by_month(cleaned_df, output_dir):
    # Ensure 'date' is in datetime format
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Filter rows for wins, losses, and draws
    wins_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].copy()
    losses_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'lose'].copy()
    draws_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'draw'].copy()

    # Extract the month from the 'date' column
    wins_data['month'] = wins_data['date'].dt.month
    losses_data['month'] = losses_data['date'].dt.month
    draws_data['month'] = draws_data['date'].dt.month

    # Group by month and count the wins, losses, and draws
    monthly_wins = wins_data.groupby('month').size()
    monthly_losses = losses_data.groupby('month').size()
    monthly_draws = draws_data.groupby('month').size()

    # Create a DataFrame combining wins, losses, and draws
    monthly_counts = pd.DataFrame({
        'wins': monthly_wins,
        'losses': monthly_losses,
        'draws': monthly_draws
    }).fillna(0)  # Fill missing values with 0 (some months may have no wins, losses, or draws)

    # Plot the side-by-side bar chart with improved color scheme
    ax = monthly_counts.plot(kind='bar', width=0.8, color=['#2ca02c', '#ff7f0e', '#1f77b4'], position=0)

    # Customize the plot for better visual appeal
    plt.title('Wins, Losses, and Draws by Month', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Number of Wins, Losses, and Draws', fontsize=12)
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=10)  # Month names
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_8.png")
    #plt.show()


# In[ ]:





# In[20]:


def plot_win_percentage_vs_opp_rating(cleaned_df, output_dir):
    # Create bins for 'opp_rating' from 0 to 2500, with intervals of 50
    rating_bins = list(range(0, 2550, 50))  # Bins from 0 to 2500, with step 50
    rating_labels = [f'{i}-{i+50}' for i in range(0, 2500, 50)]  # Create labels for each bin range

    # Assuming 'opp_rating' is available in the dataframe and represents the opponent's rating
    cleaned_df['opp_rating_range'] = pd.cut(cleaned_df['opp_rating'], bins=rating_bins, labels=rating_labels)

    # Calculate wins and losses by opponent rating range and time control type
    win_counts = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].groupby(['opp_rating_range', 'time_class']).size()
    loss_counts = cleaned_df[cleaned_df['my_win_or_lose'] == 'lose'].groupby(['opp_rating_range', 'time_class']).size()

    # Combine win and loss counts
    game_counts = win_counts.add(loss_counts, fill_value=0)

    # Calculate win percentage
    win_percentage = win_counts / game_counts * 100
    win_percentage = win_percentage.reset_index(name='win_percentage')

    # Create the scatterplot
    plt.figure(figsize=(14, 8))
    sns.scatterplot(x='opp_rating_range', y='win_percentage', hue='time_class', data=win_percentage, palette='coolwarm', s=100)
    plt.title('Win Percentage vs. Opponent Rating Range by Time Class')
    plt.xlabel('Opponent Rating Range')
    plt.ylabel('Win Percentage (%)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_9.png")
    #plt.show()


# In[21]:

def plot_moves_vs_game_outcome(cleaned_df, output_dir):
    # Filter rows where 'my_num_moves', 'time_spent', and 'my_win_or_lose' exist
    filtered_data = cleaned_df[['my_num_moves', 'time_spent', 'my_win_or_lose']].dropna()

    # Set plot size
    plt.figure(figsize=(10, 6))

    # Create scatter plot
    sns.scatterplot(x='my_num_moves', y='time_spent', hue='my_win_or_lose', data=filtered_data, palette='coolwarm', s=100)

    # Customize the plot
    plt.title('Moves vs. Game Outcome')
    plt.xlabel('Number of Moves Made')
    plt.ylabel('Time Spent (seconds)')
    plt.legend(title='Game Outcome')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_10.png")  # Save the plot
    #plt.show()



# In[22]:

def plot_win_rate_vs_num_moves(cleaned_df, output_dir):
    # Filter rows where 'my_num_moves', 'time_spent', and 'my_win_or_lose' exist
    filtered_data = cleaned_df[['my_num_moves', 'time_spent', 'my_win_or_lose']].dropna()

    # Create bins for moves (group by 5-move intervals)
    bins = range(0, filtered_data['my_num_moves'].max() + 5, 5)
    filtered_data['moves_bin'] = pd.cut(filtered_data['my_num_moves'], bins=bins)

    # Convert 'moves_bin' intervals to the lower bound of the bin for plotting
    filtered_data['moves_bin_lower'] = filtered_data['moves_bin'].apply(lambda x: x.left)

    # Group by the move bins and calculate win rate (win_count / total_games)
    win_rate_data = filtered_data.groupby('moves_bin_lower').agg(
        win_count=('my_win_or_lose', lambda x: (x == 'win').sum()),
        total_games=('my_win_or_lose', 'count')
    ).reset_index()

    # Calculate win rate as percentage
    win_rate_data['win_percentage'] = (win_rate_data['win_count'] / win_rate_data['total_games']) * 100

    # Set plot size
    plt.figure(figsize=(10, 6))

    # Plot win rate as dots and connect them with a line
    sns.scatterplot(x='moves_bin_lower', y='win_percentage', data=win_rate_data, color='green', s=100, label='Win Rate')
    plt.plot(win_rate_data['moves_bin_lower'], win_rate_data['win_percentage'], color='green', linestyle='-', linewidth=2)

    # Add shading below the curve
    plt.fill_between(win_rate_data['moves_bin_lower'], win_rate_data['win_percentage'], color='green', alpha=0.3)

    # Customize the plot
    plt.title('Win Rate vs. Number of Moves Made (Binned)')
    plt.xlabel('Number of Moves (Binned)')
    plt.ylabel('Win Rate (%)')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Display the plot
    plt.savefig(f"{output_dir}image_11.png")  # Save the plot
    #plt.show()



# In[23]:


def plot_scaled_rating_difference_by_outcome(cleaned_df, output_dir):
    # Ensure necessary columns exist and are numeric
    if 'my_rating' not in cleaned_df or 'opp_rating' not in cleaned_df or 'my_win_or_lose' not in cleaned_df:
        print("Error: Missing one of the necessary columns.")
        return
    
    # Create a new column for rating difference
    cleaned_df['rating_difference'] = cleaned_df['my_rating'] - cleaned_df['opp_rating']
    
    # Filter out rows where rating difference or my_win_or_lose is missing
    filtered_data = cleaned_df[['rating_difference', 'my_win_or_lose']].dropna()
    
    # Check if filtered data is empty
    if filtered_data.empty:
        print("Error: No valid data available after dropping missing values.")
        return

    # Standardize the rating_difference column
    scaler = StandardScaler()
    filtered_data['scaled_rating_difference'] = scaler.fit_transform(filtered_data[['rating_difference']])
    
    # Set plot size
    plt.figure(figsize=(10, 6))
    
    # Create a boxplot for scaled rating difference by game outcome with wider boxes
    sns.boxplot(x='my_win_or_lose', y='scaled_rating_difference', data=filtered_data, palette='Set2', width=0.6)
    
    # Customize the plot
    plt.title('Scaled Rating Difference vs. Game Outcome')
    plt.xlabel('Game Outcome')
    plt.ylabel('Scaled Rating Difference (Player - Opponent)')
    
    # Adjust the y-axis limits to emphasize a specific range
    plt.ylim(-3, 3)  # You can change these values to whatever makes sense for your data
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure output directory exists
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_12.png")  # Save the plot
    #plt.show()




# In[24]:


def plot_game_outcome_by_time_left(cleaned_df, output_dir):
        # Ensure 'time_left' is sorted
        # Check for NaN values and print them
    print("Checking for NaN values...")
    print(f"NaN values in 'my_time_left': {cleaned_df['my_time_left'].isna().sum()}")
    print(f"NaN values in 'my_win_or_lose': {cleaned_df['my_win_or_lose'].isna().sum()}")
    print(f"NaN values in 'time_class': {cleaned_df['time_class'].isna().sum()}")
    
    # Print unique values for each column to inspect the data
    # print("\nUnique values in 'my_time_left':")
    # print(cleaned_df['my_time_left'].unique())
    
    # print("\nUnique values in 'my_win_or_lose':")
    # print(cleaned_df['my_win_or_lose'].unique())
    
    # print("\nUnique values in 'time_class':")
    # print(cleaned_df['time_class'].unique())

    
    cleaned_df = cleaned_df.sort_values('my_time_left')

    # Define time controls
    time_controls = ['bullet', 'blitz', 'rapid']

    # Plot for each time control
    for time_control in time_controls:
        # Filter the data for the current time control
        filtered_df = cleaned_df[cleaned_df['time_class'] == time_control]

        # Group by 'time_left' and calculate outcome counts
        time_stats = filtered_df.groupby(['my_time_left', 'my_win_or_lose']).size().unstack(fill_value=0)

        # Calculate total games for each time_left
        time_stats['total'] = time_stats.sum(axis=1)

        # Calculate percentages for each outcome
        time_stats_normalized = time_stats.div(time_stats['total'], axis=0).drop(columns=['total'])

        # Smooth the data for better visualization (optional)
        time_left = time_stats_normalized.index
        # Check if the required columns exist before accessing them
        if 'win' in time_stats_normalized.columns:
            win_percent = time_stats_normalized['win'].rolling(window=30, min_periods=1).mean()
        else:
            win_percent = pd.Series([0] * len(time_stats_normalized))  # Default to 0 if no 'win' data
        
        # Handle 'draw' and 'lose' similarly
        if 'draw' in time_stats_normalized.columns:
            draw_percent = time_stats_normalized['draw'].rolling(window=30, min_periods=1).mean()
        else:
            draw_percent = pd.Series([0] * len(time_stats_normalized))
        
        if 'lose' in time_stats_normalized.columns:
            lose_percent = time_stats_normalized['lose'].rolling(window=30, min_periods=1).mean()
        else:
            lose_percent = pd.Series([0] * len(time_stats_normalized))

        # Plot the line chart
        plt.figure(figsize=(14, 8))
        plt.plot(time_left, win_percent, label='Win Percentage', color='green', linewidth=2)
        plt.plot(time_left, draw_percent, label='Draw Percentage', color='gray', linewidth=2)
        plt.plot(time_left, lose_percent, label='Loss Percentage', color='red', linewidth=2)

        # Customize the plot
        plt.title(f'Win/Draw/Loss Percentages by Time Left ({time_control.capitalize()} Games)')
        plt.xlabel('Time Left (Seconds)')
        plt.ylabel('Percentage')
        plt.legend(title='Outcome', loc='best')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}image_13.png")  # Save the plot
        #plt.show()



# In[25]:


# How many Queen Promotions you have (and other promotions?)
# How many en passants you have done
# Win and Loss ratio being white vs black
# Biggest Upset and Biggest Reverse Upset
# Brilliant Moves Finder


# In[26]:


# Are you a fair loser? how do u lose


# In[27]:


#cleaned_df.columns


# In[28]:



def plot_win_ratio_heatmap_by_castling(cleaned_df, output_dir):
    # Filter the data for relevant columns and drop missing values
    filtered_data = cleaned_df[['my_castling', 'opp_castling', 'my_win_or_lose']].dropna()

    # Create a new column to represent win (1) and loss (0), ignoring draws
    filtered_data['win'] = filtered_data['my_win_or_lose'].apply(lambda x: 1 if x == 'win' else 0)

    # Create a pivot table to calculate win ratio (wins / total games) for each castling combination
    win_ratio_data = filtered_data.groupby(['my_castling', 'opp_castling']).agg(
        win_ratio=('win', 'mean'),
        total_games=('win', 'size')
    ).reset_index()

    # Filter to include only combinations with a certain number of games (e.g., at least 10 games)
    win_ratio_data = win_ratio_data[win_ratio_data['total_games'] >= 10]

    # Define all possible castling values
    castling_values = ['kingside', 'queenside', 'none']

    # Reindex the data to ensure all combinations are represented, filling missing values with NaN
    heatmap_data = win_ratio_data.pivot(index='my_castling', columns='opp_castling', values='win_ratio')
    heatmap_data = heatmap_data.reindex(castling_values, axis=0).reindex(castling_values, axis=1)

    # Fill NaN values with 0 (or use any other placeholder you'd like, such as 'NaN' or 'No Data')
    heatmap_data = heatmap_data.fillna(0)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, cbar_kws={'label': 'Win Ratio'})

    plt.title('Win Ratio Heatmap Based on Castling Frequencies')
    plt.xlabel('Opponent Castling Frequency')
    plt.ylabel('My Castling Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_14.png")  # Save the plot

    #plt.show()


# In[29]:


def plot_game_count_heatmap_by_castling(cleaned_df, output_dir):
    # Filter the data for relevant columns and drop missing values
    filtered_data = cleaned_df[['my_castling', 'opp_castling', 'my_win_or_lose']].dropna()

    # Create a pivot table to count the number of games for each castling combination
    game_count_data = filtered_data.groupby(['my_castling', 'opp_castling']).size().reset_index(name='game_count')

    # Define all possible castling values
    castling_values = ['kingside', 'queenside', 'none']

    # Reindex the data to ensure all combinations are represented, filling missing values with 0
    heatmap_data = game_count_data.pivot(index='my_castling', columns='opp_castling', values='game_count')
    heatmap_data = heatmap_data.reindex(castling_values, axis=0).reindex(castling_values, axis=1)

    # Fill NaN values with 0 (or use any other placeholder you'd like, such as 'NaN' or 'No Data')
    heatmap_data = heatmap_data.fillna(0)

    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='Blues', fmt='d', linewidths=.5, cbar_kws={'label': 'Number of Games'})

    plt.title('Game Count Heatmap Based on Castling Frequencies')
    plt.xlabel('Opponent Castling Frequency')
    plt.ylabel('My Castling Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_15.png")  # Save the plot

    #plt.show()


# In[30]:


def plot_rating_progression_over_time(cleaned_df, output_dir):
    """
    This function generates a plot of rating progression over time for different time classes.
    It filters the data for non-null ratings and plots the rating for each time class.
    """
    # Ensure 'date' is in datetime format
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Filter rows where 'my_rating' exists
    rating_data = cleaned_df[~cleaned_df['my_rating'].isna()]

    # Get unique time_classes
    time_classes = rating_data['time_class'].unique()

    # Plot rating over time for each time_class
    plt.figure(figsize=(12, 6))

    for time_class in time_classes:
        # Filter data for the current time_class
        time_class_data = rating_data[rating_data['time_class'] == time_class]
        
        # Plot rating over time for the current time_class
        plt.plot(time_class_data['date'], time_class_data['my_rating'], marker='o', linestyle='-', label=time_class)

    # Customize the plot
    plt.title('Rating Over Time for Different Time Controls')
    plt.xlabel('Date')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend(title='Time Class')

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_16.png")  # Save the plot
    #plt.show()

def call_visualizations(cleaned_df, output_dir):
    print("Visualzations have been called")

    # plot_rating_progression_over_time(cleaned_df, output_dir)
    plot_opening_statistics(cleaned_df, output_dir)
    print("Visualizations are over")
    #plot_time_class_statistics(cleaned_df, output_dir)
    #plot_scaled_rating_difference_by_outcome(cleaned_df, output_dir)
    #plot_game_outcome_by_hour(cleaned_df, output_dir)
    #plot_game_outcome_by_day(cleaned_df, output_dir)
    # plot_game_outcome_by_month(cleaned_df, output_dir)
    # plot_win_percentage_vs_opp_rating(cleaned_df, output_dir)
    # plot_moves_vs_game_outcome(cleaned_df, output_dir)
    # plot_win_rate_vs_num_moves(cleaned_df, output_dir)
    # plot_scaled_rating_difference_by_outcome(cleaned_df, output_dir)
    # plot_game_outcome_by_time_left(cleaned_df, output_dir)
    # plot_win_ratio_heatmap_by_castling(cleaned_df, output_dir)
    # plot_game_count_heatmap_by_castling(cleaned_df, output_dir)
