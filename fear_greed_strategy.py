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

# Fix the MultiIndex structure - we need a flat DataFrame
if isinstance(spy_data.columns, pd.MultiIndex):
    # Drop the ticker level if it's a MultiIndex
    spy_data.columns = [col[0] for col in spy_data.columns]

# Print debugging info
print("SPY data columns after fixing:", spy_data.columns)

# Combine the datasets
combined_data = pd.DataFrame(index=spy_data.index)
combined_data['Close'] = spy_data['Close']
combined_data['fear_greed_value'] = fear_greed_df['fear_greed_value']

# Forward fill any missing Fear & Greed values
combined_data['fear_greed_value'] = combined_data['fear_greed_value'].ffill()

# Drop any remaining rows with NaN values
combined_data = combined_data.dropna()

# Print debugging info
print("Combined data after processing:")
print(combined_data.head())


# Create sentiment labels
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


combined_data['sentiment'] = combined_data['fear_greed_value'].apply(get_sentiment_label)

# Initialize backtesting variables
initial_capital = 15000.0
cash = initial_capital
shares_owned = 0.0
buy_amount = 250.0
sell_amount = 250.0

# Create lists to store values for each date
cash_values = []
shares_values = []
position_values = []
total_values = []
trade_records = []

# Simple loop through each row
for idx, row in combined_data.iterrows():
    price = row['Close']
    sentiment = row['sentiment']

    # Trading logic
    trade = ""
    if sentiment == 'Extreme Fear' and cash >= buy_amount:
        shares_to_buy = buy_amount / price
        shares_owned += shares_to_buy
        cash -= buy_amount
        trade = f"BUY {shares_to_buy:.2f} shares at ${price:.2f}"
    elif sentiment == 'Extreme Greed' and shares_owned > 0:
        shares_value = shares_owned * price
        if shares_value <= sell_amount:
            # Sell all remaining shares
            cash += shares_value
            trade = f"SELL ALL {shares_owned:.2f} shares at ${price:.2f}"
            shares_owned = 0
        else:
            # Sell $250 worth of shares
            shares_to_sell = sell_amount / price
            shares_owned -= shares_to_sell
            cash += sell_amount
            trade = f"SELL {shares_to_sell:.2f} shares at ${price:.2f}"

    # Calculate position value
    position_value = shares_owned * price
    total_value = cash + position_value

    # Store values
    cash_values.append(cash)
    shares_values.append(shares_owned)
    position_values.append(position_value)
    total_values.append(total_value)
    trade_records.append(trade)

# Assign the calculated values back to the DataFrame
combined_data['cash'] = cash_values
combined_data['shares_owned'] = shares_values
combined_data['position_value'] = position_values
combined_data['total_portfolio_value'] = total_values
combined_data['trades'] = trade_records

# Create benchmark (buy and hold SPY)
benchmark_shares = initial_capital / combined_data['Close'].iloc[0]
combined_data['benchmark_value'] = benchmark_shares * combined_data['Close']

# Calculate strategy performance metrics
start_value = initial_capital
end_value = combined_data['total_portfolio_value'].iloc[-1]
total_return = (end_value / start_value - 1) * 100
annual_return = ((end_value / start_value) ** (252 / len(combined_data)) - 1) * 100
max_drawdown = (combined_data['total_portfolio_value'] / combined_data[
    'total_portfolio_value'].cummax() - 1).min() * 100
daily_returns = combined_data['total_portfolio_value'].pct_change().dropna()
sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0

# Calculate benchmark performance metrics
benchmark_end_value = combined_data['benchmark_value'].iloc[-1]
benchmark_total_return = (benchmark_end_value / start_value - 1) * 100
benchmark_annual_return = ((benchmark_end_value / start_value) ** (252 / len(combined_data)) - 1) * 100
benchmark_max_drawdown = (combined_data['benchmark_value'] / combined_data['benchmark_value'].cummax() - 1).min() * 100
benchmark_daily_returns = combined_data['benchmark_value'].pct_change().dropna()
benchmark_sharpe_ratio = (benchmark_daily_returns.mean() / benchmark_daily_returns.std()) * np.sqrt(
    252) if benchmark_daily_returns.std() != 0 else 0

# Print performance summary
print(f"Backtest Period: {combined_data.index.min().date()} to {combined_data.index.max().date()}")
print(f"\nStarting Capital: ${initial_capital:,.2f}")
print(f"Ending Portfolio Value: ${end_value:,.2f}")
print(f"Final Cash: ${cash:,.2f}")
print(f"Final Shares Owned: {shares_owned:.2f}")
print(f"Final Position Value: ${position_value:,.2f}")
print(f"\nTotal Return: {total_return:.2f}%")
print(f"Annualized Return: {annual_return:.2f}%")
print(f"Maximum Drawdown: {max_drawdown:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

print(f"\nBenchmark (Buy & Hold) Performance:")
print(f"Ending Value: ${benchmark_end_value:,.2f}")
print(f"Total Return: {benchmark_total_return:.2f}%")
print(f"Annualized Return: {benchmark_annual_return:.2f}%")
print(f"Maximum Drawdown: {benchmark_max_drawdown:.2f}%")
print(f"Sharpe Ratio: {benchmark_sharpe_ratio:.2f}")

# Count trades
buy_trades = [t for t in trade_records if 'BUY' in t]
sell_trades = [t for t in trade_records if 'SELL' in t]
print(f"\nTotal Buy Trades: {len(buy_trades)}")
print(f"Total Sell Trades: {len(sell_trades)}")

# Calculate days spent in each sentiment category
sentiment_counts = combined_data['sentiment'].value_counts()
print("\nDays in Each Sentiment Category:")
for sentiment, days in sentiment_counts.items():
    print(f"{sentiment}: {days} days ({days / len(combined_data) * 100:.2f}%)")

# Plot portfolio value vs benchmark
plt.figure(figsize=(14, 7), num="Portfolio Performance")
plt.plot(combined_data.index, combined_data['total_portfolio_value'], label='Fear & Greed Strategy', color='blue',
         linewidth=1.5)
plt.plot(combined_data.index, combined_data['benchmark_value'], label='Buy & Hold SPY', color='gray', linewidth=1.5,
         alpha=0.7)
plt.title('Fear & Greed Strategy vs Buy & Hold SPY', fontsize=16)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show(block=False)

# Plot equity curve with buy/sell markers
plt.figure(figsize=(14, 10), num="Equity Curve with Trades")

# Plot equity curve
plt.subplot(2, 1, 1)
plt.plot(combined_data.index, combined_data['total_portfolio_value'], color='blue', linewidth=1.5)
plt.plot(combined_data.index, combined_data['benchmark_value'], color='gray', linewidth=1.5, alpha=0.7,
         label='Buy & Hold SPY')

# Add buy markers
buy_indices = [i for i, t in enumerate(trade_records) if 'BUY' in t]
if buy_indices:
    buy_dates = [combined_data.index[i] for i in buy_indices]
    buy_values = [combined_data['total_portfolio_value'].iloc[i] for i in buy_indices]
    plt.scatter(buy_dates, buy_values, marker='^', color='green', s=50, label='Buy')

# Add sell markers
sell_indices = [i for i, t in enumerate(trade_records) if 'SELL' in t]
if sell_indices:
    sell_dates = [combined_data.index[i] for i in sell_indices]
    sell_values = [combined_data['total_portfolio_value'].iloc[i] for i in sell_indices]
    plt.scatter(sell_dates, sell_values, marker='v', color='red', s=50, label='Sell')

plt.title('Portfolio Equity Curve with Buy/Sell Signals', fontsize=16)
plt.ylabel('Portfolio Value ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Fear & Greed Index
plt.subplot(2, 1, 2)
plt.plot(combined_data.index, combined_data['fear_greed_value'], color='blue', linewidth=1.5)
plt.axhline(y=25, color='r', linestyle='--', alpha=0.7, label='Extreme Fear (25)')
plt.axhline(y=75, color='g', linestyle='--', alpha=0.7, label='Extreme Greed (75)')
plt.fill_between(combined_data.index, 0, 25, color='red', alpha=0.1)
plt.fill_between(combined_data.index, 75, 100, color='green', alpha=0.1)
plt.title('CNN Fear & Greed Index', fontsize=14)
plt.ylabel('Index Value', fontsize=12)
plt.ylim(0, 100)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show(block=False)

# Plot drawdowns
combined_data['strategy_drawdown'] = combined_data['total_portfolio_value'] / combined_data[
    'total_portfolio_value'].cummax() - 1
combined_data['benchmark_drawdown'] = combined_data['benchmark_value'] / combined_data['benchmark_value'].cummax() - 1

plt.figure(figsize=(14, 7), num="Drawdown Comparison")
plt.plot(combined_data.index, combined_data['strategy_drawdown'] * 100, label='Fear & Greed Strategy', color='blue',
         linewidth=1.5)
plt.plot(combined_data.index, combined_data['benchmark_drawdown'] * 100, label='Buy & Hold SPY', color='gray',
         linewidth=1.5, alpha=0.7)
plt.title('Drawdown Comparison: Fear & Greed Strategy vs Buy & Hold SPY', fontsize=16)
plt.ylabel('Drawdown (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Keep the program running until all plot windows are closed
plt.pause(0.1)  # Small pause to ensure all plots are displayed
input("Press Enter to close all plots and exit...")