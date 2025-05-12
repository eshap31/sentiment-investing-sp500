import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Load the Fear & Greed data
fear_greed_df = pd.read_csv('Data/fear-greed-2011-2023.csv')

# Convert date column to datetime format
fear_greed_df['Date'] = pd.to_datetime(fear_greed_df['Date'])

# Set the date as the index
fear_greed_df = fear_greed_df.set_index('Date')
fear_greed_df = fear_greed_df.rename(columns={'Fear Greed': 'fear_greed_value'})

# Sort by date
fear_greed_df = fear_greed_df.sort_index()

# Download S&P 500 data for the same period
start_date = fear_greed_df.index.min()
end_date = fear_greed_df.index.max()
spy_data = yf.download('SPY', start=start_date, end=end_date)

# Combine the datasets
combined_data = spy_data.copy()
combined_data['fear_greed_value'] = fear_greed_df['fear_greed_value']
combined_data = combined_data.dropna()

# Plot 1: Fear & Greed Index over time
plt.figure(figsize=(14, 7), num='Fear & Greed Time Series')
plt.plot(fear_greed_df.index, fear_greed_df['fear_greed_value'], color='blue', linewidth=1.5)
plt.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Extreme Fear Threshold (25)')
plt.axhline(y=75, color='g', linestyle='--', alpha=0.7, label='Extreme Greed Threshold (75)')
plt.fill_between(fear_greed_df.index, 0, 25, color='red', alpha=0.1, label='Extreme Fear Zone')
plt.fill_between(fear_greed_df.index, 75, 100, color='green', alpha=0.1, label='Extreme Greed Zone')
plt.title('CNN Fear & Greed Index (2011-2023)', fontsize=16)
plt.ylabel('Index Value', fontsize=12)
plt.ylim(0, 100)
plt.legend()
plt.tight_layout()
plt.show(block=False)  # This makes the plot non-blocking

# Plot 2: Distribution of Fear & Greed values
plt.figure(figsize=(10, 6), num='Fear & Greed Distribution')
sns.histplot(fear_greed_df['fear_greed_value'], bins=20, kde=True)
plt.title('Distribution of Fear & Greed Index Values', fontsize=16)
plt.xlabel('Index Value', fontsize=12)
plt.axvline(x=25, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=75, color='g', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show(block=False)  # This makes the plot non-blocking

# Analyze time spent in each sentiment zone
def get_sentiment_label(value):
    if value <= 24:
        return 'Extreme Fear'
    elif value <= 49:
        return 'Fear'
    elif value == 50:
        return 'Neutral'
    elif value <= 74:
        return 'Greed'
    else:
        return 'Extreme Greed'

fear_greed_df['sentiment'] = fear_greed_df['fear_greed_value'].apply(get_sentiment_label)
sentiment_counts = fear_greed_df['sentiment'].value_counts()

# Plot 3: Sentiment distribution
plt.figure(figsize=(10, 6), num='Sentiment Distribution')
sentiment_counts.plot(kind='bar', color=['firebrick', 'lightcoral', 'gray', 'lightgreen', 'forestgreen'])
plt.title('Distribution of Market Sentiment States', fontsize=16)
plt.ylabel('Number of Days', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=False)  # This makes the plot non-blocking

# Calculate percentage of time in each sentiment state
sentiment_percentages = (sentiment_counts / len(fear_greed_df) * 100).round(2)
print("Percentage of time in each sentiment state:")
for sentiment, percentage in sentiment_percentages.items():
    print(f"{sentiment}: {percentage}%")

# Plot 4: SPY price alongside Fear & Greed
plt.figure(figsize=(14, 10), num='SPY vs Fear & Greed')

# Plot SPY price
ax1 = plt.subplot(2, 1, 1)
ax1.plot(combined_data.index, combined_data['Close'], color='black', linewidth=1.5)
ax1.set_title('S&P 500 ETF (SPY) Price', fontsize=14)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot Fear & Greed Index
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(combined_data.index, combined_data['fear_greed_value'], color='blue', linewidth=1.5)
ax2.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Extreme Fear (25)')
ax2.axhline(y=75, color='g', linestyle='--', alpha=0.7, label='Extreme Greed (75)')
ax2.fill_between(combined_data.index, 0, 25, color='red', alpha=0.1)
ax2.fill_between(combined_data.index, 75, 100, color='green', alpha=0.1)
ax2.set_title('CNN Fear & Greed Index', fontsize=14)
ax2.set_ylabel('Index Value', fontsize=12)
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()  # This is the final plot, so we use a blocking show()

# Keep the program running until all plot windows are closed
plt.pause(0.1)  # Small pause to ensure all plots are displayed
input("Press Enter to close all plots and exit...")