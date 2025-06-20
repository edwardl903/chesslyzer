
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
    print(cleaned_df['start_time_est'].dtype)
    cleaned_df['start_time_est'] = pd.to_datetime(cleaned_df['start_time_est'], format='%H:%M:%S')
    cleaned_df['hour'] = cleaned_df['start_time_est'].dt.hour

    # 1. Count the frequency of games played each month
    monthly_counts = cleaned_df['month'].value_counts().sort_index()
    monthly_counts.index = monthly_counts.index.strftime('%b') #indexes it to jan, feb , etc


    # 2. Count the frequency of games played by weekday
    weekday_counts = cleaned_df['weekday'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # 3. Count the frequency of games played by hour
    hourly_counts = cleaned_df['hour'].value_counts().sort_index()



    # Plotting the frequency of games over months
    plt.figure(figsize=(10, 6))
    sns.barplot(x=monthly_counts.index.astype(str), y=monthly_counts.values, palette="flare")
    plt.title('Frequency of Games Played Over Months', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Frequency of Games')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.savefig(f"{output_dir}image_1.png")
    #plt.show()

    # Plotting the frequency of games by weekday
    plt.figure(figsize=(10, 6))
    sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette="flare")
    plt.title('Frequency of Games Played by Weekday', fontsize=16)
    plt.xlabel('Weekday')
    plt.ylabel('Frequency of Games')
    plt.xticks(rotation=45)
    plt.savefig(f"{output_dir}image_2.png")
    #plt.show()

    # Plotting the frequency of games by hour
    plt.figure(figsize=(10, 6))
    sns.barplot(x=hourly_counts.index.astype(str), y=hourly_counts.values, palette="flare")
    plt.title('Frequency of Games Played by Hour', fontsize=16)
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
    plt.title('Win/Draw/Loss Rates for Top 10 Most Common Openings (with Game Counts)', pad=20, fontsize=16)
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
 # Filter rows for wins
    wins_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].copy()

    # Convert 'start_time_est' to datetime and extract the hour
    cleaned_df['start_time_est'] = pd.to_datetime(cleaned_df['start_time_est'], format='%H:%M:%S')
    wins_data['start_time_est'] = pd.to_datetime(wins_data['start_time_est'], format='%H:%M:%S')

    # Extract the hour from the 'start_time_est' column
    cleaned_df['hour'] = cleaned_df['start_time_est'].dt.hour
    wins_data['hour'] = wins_data['start_time_est'].dt.hour

    # Group by hour and count total games and wins
    hourly_totals = cleaned_df.groupby('hour').size()
    hourly_wins = wins_data.groupby('hour').size()

    # Calculate win percentage
    hourly_win_percentage = (hourly_wins / hourly_totals) * 100
    hourly_win_percentage = hourly_win_percentage.fillna(0)  # Replace NaN with 0% for hours without wins

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the total games as a bar chart
    ax1.bar(hourly_totals.index, hourly_totals, color='#ff6347', width=0.4, label='Total Games')
    ax1.set_xlabel('Hour of the Day (EST)', fontsize=12)
    ax1.set_ylabel('Total Games (Record Count)', color='#ff6347', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#ff6347')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.set_xticks(range(0, 24))

    # Create a second axis to plot the win percentage as a line chart
    ax2 = ax1.twinx()
    ax2.plot(hourly_win_percentage.index, hourly_win_percentage, color='#1f77b4', marker='o', label='Win Percentage', linewidth=2)
    ax2.set_ylabel('Win Percentage (%)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0, 100)  # Percentage ranges from 0 to 100

    # Add a title and legend
    plt.title('Win Percentage and Total Games by Hour of the Day', fontsize=16)
    fig.tight_layout()  # Ensure everything fits without overlap

    # Save the plot
    plt.savefig(f"{output_dir}image_6.png")
    #plt.show()



# In[18]:


def plot_game_outcome_by_day(cleaned_df, output_dir):
    # Ensure 'date' is in datetime format
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Filter rows for wins
    wins_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].copy()

    # Extract the day of the week from the 'date' column
    cleaned_df['day_of_week'] = cleaned_df['date'].dt.day_name()
    wins_data['day_of_week'] = wins_data['date'].dt.day_name()

    # Group by day of the week and count total games and wins
    daily_totals = cleaned_df.groupby('day_of_week').size()
    daily_wins = wins_data.groupby('day_of_week').size()

    # Calculate win percentage
    daily_win_percentage = (daily_wins / daily_totals) * 100
    daily_win_percentage = daily_win_percentage.fillna(0)  # Replace NaN with 0% for days without wins

    # Reorder days of the week for proper ordering in the plot
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_totals = daily_totals.reindex(days_order)
    daily_win_percentage = daily_win_percentage.reindex(days_order)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the total games as a bar chart
    ax1.bar(daily_totals.index, daily_totals, color='#ff6347', width=0.4, label='Total Games')
    ax1.set_xlabel('Day of the Week', fontsize=12)
    ax1.set_ylabel('Total Games (Record Count)', color='#ff6347', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#ff6347')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Create a second axis to plot the win percentage as a line chart
    ax2 = ax1.twinx()
    ax2.plot(daily_win_percentage.index, daily_win_percentage, color='#1f77b4', marker='o', label='Win Percentage', linewidth=2)
    ax2.set_ylabel('Win Percentage (%)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0, 100)  # Percentage ranges from 0 to 100

    # Add a title and legend
    plt.title('Win Percentage and Total Games by Day of the Week', fontsize=16)
    fig.tight_layout()  # Ensure everything fits without overlap

    # Save the plot
    plt.savefig(f"{output_dir}image_7.png")
    #plt.show()

# In[19]:


def plot_game_outcome_by_month(cleaned_df, output_dir):
    # Ensure 'date' is in datetime format
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Filter rows for wins
    wins_data = cleaned_df[cleaned_df['my_win_or_lose'] == 'win'].copy()

    # Extract the month from the 'date' column
    cleaned_df['month'] = cleaned_df['date'].dt.month
    wins_data['month'] = wins_data['date'].dt.month

    # Group by month and count total games and wins
    monthly_totals = cleaned_df.groupby('month').size()
    monthly_wins = wins_data.groupby('month').size()

    # Calculate win percentage
    monthly_win_percentage = (monthly_wins / monthly_totals) * 100
    monthly_win_percentage = monthly_win_percentage.fillna(0)  # Replace NaN with 0% for months without wins

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot the total games as a bar chart
    ax1.bar(monthly_totals.index, monthly_totals, color='#ff6347', width=0.4, label='Total Games')
    ax1.set_xlabel('Month', fontsize=12)
    ax1.set_ylabel('Total Games (Record Count)', color='#ff6347', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#ff6347')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    # Create a second axis to plot the win percentage as a line chart
    ax2 = ax1.twinx()
    ax2.plot(monthly_win_percentage.index, monthly_win_percentage, color='#1f77b4', marker='o', label='Win Percentage', linewidth=2)
    ax2.set_ylabel('Win Percentage (%)', color='#1f77b4', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#1f77b4')
    ax2.set_ylim(0, 100)  # Percentage ranges from 0 to 100

    # Add a title and legend
    plt.title('Win Percentage and Total Games by Month', fontsize=16)
    fig.tight_layout()  # Ensure everything fits without overlap

    # Save the plot
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
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='opp_rating_range', y='win_percentage', hue='time_class', data=win_percentage, palette='coolwarm', s=100)
    plt.title('Win Percentage vs. Opponent Rating Range by Time Class', fontsize=16)
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
    plt.title('Moves vs. Game Outcome', fontsize=16)
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
    plt.title('Win Rate vs. Number of Moves Made (Binned)', fontsize=16)
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
    plt.title('Scaled Rating Difference vs. Game Outcome', fontsize=16)
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
        plt.figure(figsize=(10, 6))
        plt.plot(time_left, win_percent, label='Win Percentage', color='green', linewidth=2)
        plt.plot(time_left, draw_percent, label='Draw Percentage', color='gray', linewidth=2)
        plt.plot(time_left, lose_percent, label='Loss Percentage', color='red', linewidth=2)

        # Customize the plot
        plt.title(f'Win/Draw/Loss Percentages by Time Left ({time_control.capitalize()} Games)', fontsize=16)
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

    plt.title('Win Ratio Heatmap Based on Castling Frequencies', fontsize=16)
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
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        cmap='Blues', 
        fmt='.0f',  # Display as whole numbers
        linewidths=.5, 
        cbar_kws={'label': 'Number of Games'}
    )
    plt.title('Game Count Heatmap Based on Castling Frequencies', fontsize=16)
    plt.xlabel('Opponent Castling Frequency')
    plt.ylabel('My Castling Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_15.png")  # Save the plot

    #plt.show()


# In[30]:

def plot_rating_progression_over_time(cleaned_df, output_dir):
    """
    This function generates a plot of rating progression over time for different time classes,
    averaged monthly. It filters the data for non-null ratings and plots the average monthly rating
    for each time class.
    """
    # Ensure 'date' is in datetime format
    cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])

    # Filter rows where 'my_rating' exists and is numeric
    rating_data = cleaned_df[~cleaned_df['my_rating'].isna() & cleaned_df['my_rating'].apply(lambda x: isinstance(x, (int, float)))]

    # Extract the month from the 'date' column for grouping
    rating_data['month'] = rating_data['date'].dt.month

    # Get unique time_classes
    time_classes = rating_data['time_class'].unique()

    # Prepare the plot
    plt.figure(figsize=(10, 6))

    for time_class in time_classes:
        # Filter data for the current time class
        time_class_data = rating_data[rating_data['time_class'] == time_class]

        # Ensure that only numeric values are being processed
        time_class_data['my_rating'] = pd.to_numeric(time_class_data['my_rating'], errors='coerce')

        # Group by month and calculate the mean rating
        monthly_avg = time_class_data.groupby('month')['my_rating'].mean()

        # Plot the monthly average rating over time for the current time class
        plt.plot(monthly_avg.index, monthly_avg, marker='o', linestyle='-', label=time_class)

    # Customize the plot
    plt.title('Monthly Average Rating Over Time for Different Time Controls', fontsize=16)
    plt.xlabel('Month')
    plt.ylabel('Average Rating')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True)
    plt.legend(title='Time Class')

     # Adding a border to the plot
    


    # Display the plot
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}image_0.png")  # Save the plot
    #plt.show()


def plot_game_count_vs_time_spent_by_type(cleaned_df, output_dir):
    """
    Plot game count vs time spent (in seconds), categorized by game type (Blitz, Rapid, Bullet),
    with time spent divided into 1-minute bins.

    Parameters:
        cleaned_df (DataFrame): A cleaned DataFrame with 'time_spent' (in seconds) and 'game_type' columns.
        output_dir (str): Directory to save the output image.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Ensure the 'time_spent' column is numeric (in seconds)
    cleaned_df['time_spent'] = pd.to_numeric(cleaned_df['time_spent'], errors='coerce')
    cleaned_df = cleaned_df.dropna(subset=['time_spent'])  # Drop rows with missing 'time_spent'

    # Define time bins in seconds (e.g., 0-60 seconds, 60-120 seconds, etc.)
    bins = [0, 60, 120, 180, 240, 300, 600, 900, 1200, 1800, 2400, 3000]
    labels = ['0-1 min', '1-2 min', '2-3 min', '3-4 min', '4-5 min', '5-10 min', '10-15 min', '15-20 min', '20-30 min', '30-40 min', '40-50 min']

    # Create a new column with the binned time spent values (in seconds)
    cleaned_df['time_spent_bins'] = pd.cut(cleaned_df['time_spent'], bins=bins, labels=labels, right=False)

    # Filter for specific game types
    time_class = ['blitz', 'rapid', 'bullet']
    filtered_df = cleaned_df[cleaned_df['time_class'].isin(time_class)]

    # Group by time spent bin and game type, then count the games
    game_counts = filtered_df.groupby(['time_spent_bins', 'time_class']).size().unstack(fill_value=0)

    # Ensure all time classes are present in the DataFrame
    for game_type in time_class:
        if game_type not in game_counts.columns:
            game_counts[game_type] = 0

    # Sort the columns to ensure consistent order
    game_counts = game_counts[time_class]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot bars for each game type
    bar_width = 0.25  # Width of each bar
    x_positions = game_counts.index
    x_indices = range(len(x_positions))

    ax.bar([x - bar_width for x in x_indices], game_counts['blitz'], width=bar_width, label='Blitz', color='#1f77b4')
    ax.bar(x_indices, game_counts['rapid'], width=bar_width, label='Rapid', color='#ff7f0e')
    ax.bar([x + bar_width for x in x_indices], game_counts['bullet'], width=bar_width, label='Bullet', color='#2ca02c')

    # Customize the plot
    ax.set_title('Game Count vs Time Spent by Game Type', fontsize=16)
    ax.set_xlabel('Time Spent per Game (Binned in Minutes)', fontsize=12)
    ax.set_ylabel('Number of Games Played', fontsize=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels(x_positions, fontsize=10)

    # Set y-tick font size
    ax.tick_params(axis='y', labelsize=10)

    # Add grid and legend
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Game Type', fontsize=10)

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"{output_dir}/image_16.png")
    #plt.show()


def call_visualizations(cleaned_df, output_dir):
    print("Visualzations have been called")

    plot_game_statistics(cleaned_df, output_dir)
    plot_rating_progression_over_time(cleaned_df, output_dir)
    plot_opening_statistics(cleaned_df, output_dir)
    print("Visualizations are over")
    plot_time_class_statistics(cleaned_df, output_dir)
    plot_scaled_rating_difference_by_outcome(cleaned_df, output_dir)
    plot_game_outcome_by_hour(cleaned_df, output_dir)
    plot_game_outcome_by_day(cleaned_df, output_dir)
    plot_game_outcome_by_month(cleaned_df, output_dir)
    plot_win_percentage_vs_opp_rating(cleaned_df, output_dir)
    plot_moves_vs_game_outcome(cleaned_df, output_dir)
    plot_win_rate_vs_num_moves(cleaned_df, output_dir)
    plot_scaled_rating_difference_by_outcome(cleaned_df, output_dir)
    plot_game_outcome_by_time_left(cleaned_df, output_dir)
    plot_win_ratio_heatmap_by_castling(cleaned_df, output_dir)
    plot_game_count_heatmap_by_castling(cleaned_df, output_dir)
    plot_game_count_vs_time_spent_by_type(cleaned_df, output_dir)
