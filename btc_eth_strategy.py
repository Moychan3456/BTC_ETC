import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from tabulate import tabulate
import random
from datetime import date, timedelta
import pytz
import talib
import mplfinance as mpf
import ccxt

# === CONFIG ===
end_date = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
start_date = end_date - timedelta(days=2000)
interval = '1h'
resample_to = '4h'


# === BACKTESTING PARAMETERS ===
capital = 100000.00
risk_per_trade_percent = 0.5
RISK_REWARD_RATIO = 4.5
timezone = 'Asia/Kolkata'

# === REAL-WORLD TRADING COSTS ===
COMMISSION_PER_LOT = 7.00
SLIPPAGE_PIPS = (0.5, 1.5)

# === THE FOCUSED PORTFOLIO ===
tickers = [
    'BTC/USDT', 'ETH/USDT'
]

# === DATA FETCHING FUNCTION ===
def fetch_yfinance_data(instrument, start_date, end_date, interval):
    print(f"➡️ Downloading historical data for {instrument} from ccxt...")

    try:
        exchange = ccxt.binance()
        
        # Map your yfinance interval to ccxt timeframe
        interval_map = {
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        timeframe = interval_map.get(interval, '1h')

        # Convert start/end to ccxt milliseconds
        since = int(start_date.timestamp() * 1000)
        all_data = []

        while True:
            candles = exchange.fetch_ohlcv(instrument, timeframe, since=since, limit=1500)
            if not candles:
                break
            all_data.extend(candles)
            
            # Stop if we’ve reached the end_date
            last_ts = candles[-1][0]
            if last_ts >= int(end_date.timestamp() * 1000):
                break
            
            since = last_ts + (exchange.parse_timeframe(timeframe) * 1000)

        if not all_data:
            print(f"❌ No data for {instrument}")
            return pd.DataFrame()

        df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df.set_index('timestamp', inplace=True)
        df.index = df.index.tz_convert(timezone)
        
        # Keep Adj Close column for compatibility
        df['Adj Close'] = df['Close']
        return df

    except Exception as e:
        print(f"🚫 Error fetching data for {instrument}: {e}")
        return pd.DataFrame()


# === STRATEGY HELPERS ===
def is_bullish_candle(candle):
    return candle['Close'] > candle['Open']

# === BACKTEST LOGIC (MAIN FUNCTION) ===
def run_backtest_for_ticker(ticker_symbol):
    print(f"\n=======================================================")
    print(f"      ✅  ANALYZING: {ticker_symbol} from {start_date.date()} to {end_date.date()}")
    print(f"=======================================================")

    df = fetch_yfinance_data(ticker_symbol, start_date, end_date, interval)

    if df.empty:
        return None

    trades = []

    print(f"➡️ Resampling and running backtest logic...")
    df_4h = df.resample(resample_to).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }).dropna()

    df_4h['MA'] = df_4h['Close'].rolling(window=200).mean()
    df_4h['RSI'] = talib.RSI(df_4h['Close'], timeperiod=14)
    # Calculate the slope of the MA
    df_4h['MA_Slope'] = df_4h['MA'].diff()

    if len(df_4h) < 201:
        print(f"⚠️ Not enough data for 200-period MA for {ticker_symbol}. Skipping.")
        return None

    risk_amount_per_trade = (capital * risk_per_trade_percent / 100)
    
    # Store trade entry signals for visualization
    entry_signals = pd.Series(index=df_4h.index, dtype=float)

    for i in range(201, len(df_4h) - 1):
        prev_candle = df_4h.iloc[i - 1]
        curr_candle = df_4h.iloc[i]

        # --- NEW & IMPROVED MARKET-STATE FILTER ---
        # We now require the price to be above the MA AND the MA itself must be trending up (positive slope).
        # We also require RSI to be above a more conservative level of 60.
        market_is_bullish = (
            curr_candle['Close'] > curr_candle['MA'] and
            df_4h['MA_Slope'].iloc[i] > 0 and
            curr_candle['RSI'] > 45
        )
        
        if market_is_bullish:
            # --- The entry conditions are now just the bullish candles ---
            if (is_bullish_candle(prev_candle) and is_bullish_candle(curr_candle)):
    
                # This is the point in time where the trade decision is made.
                # We now model the entry on the NEXT candle's open.
                entry_candle_data = df_4h.iloc[i + 1]
                entry_time = df_4h.index[i + 1]
                
                slippage_factor = 10000
                if 'JPY' in ticker_symbol:
                    slippage_factor = 100
                slippage_amount = random.uniform(SLIPPAGE_PIPS[0], SLIPPAGE_PIPS[1]) / slippage_factor
                
                entry_price = entry_candle_data['Open'] + slippage_amount
                stop_loss = curr_candle['Low']
                
                if entry_price <= stop_loss:
                    # The trade would be a loss immediately, but we can't trade on that.
                    # Skip this signal as it's invalid.
                    continue
                
                risk_per_unit = entry_price - stop_loss
                target = entry_price + (risk_per_unit * RISK_REWARD_RATIO)
                
                if risk_per_unit == 0:
                    continue

                position_size = risk_amount_per_trade / risk_per_unit
                lots = position_size / 100000
                commission_cost = lots * COMMISSION_PER_LOT
                
                in_trade = True
                
                # Store the entry price for plotting
                entry_signals.loc[entry_time] = entry_price

                # --- FIX FOR LOOKAHEAD BIAS ON STOP LOSS/TARGET ---
                # Check the entry candle itself for a stop-out or target hit.
                if entry_candle_data['Low'] <= stop_loss:
                    exit_price = stop_loss
                    pnl = -risk_amount_per_trade
                    pnl_after_costs = pnl - commission_cost
                    trades.append({
                        'Ticker': ticker_symbol,
                        'EntryTime': entry_time, 'ExitTime': df_4h.index[i+1],
                        'EntryPrice': entry_price, 'ExitPrice': exit_price,
                        'Outcome': 'loss', 'PnL': pnl_after_costs
                    })
                    in_trade = False
                elif entry_candle_data['High'] >= target:
                    exit_price = target
                    pnl = risk_amount_per_trade * RISK_REWARD_RATIO
                    pnl_after_costs = pnl - commission_cost
                    trades.append({
                        'Ticker': ticker_symbol,
                        'EntryTime': entry_time, 'ExitTime': df_4h.index[i+1],
                        'EntryPrice': entry_price, 'ExitPrice': exit_price,
                        'Outcome': 'win', 'PnL': pnl_after_costs
                    })
                    in_trade = False

                if in_trade:
                    # Now, if the trade is still live, check subsequent candles
                    for j in range(i + 2, len(df_4h)):
                        if df_4h['Low'].iloc[j] <= stop_loss:
                            exit_price = stop_loss
                            pnl = -risk_amount_per_trade
                            pnl_after_costs = pnl - commission_cost
                            trades.append({
                                'Ticker': ticker_symbol,
                                'EntryTime': entry_time, 'ExitTime': df_4h.index[j],
                                'EntryPrice': entry_price, 'ExitPrice': exit_price,
                                'Outcome': 'loss', 'PnL': pnl_after_costs
                            })
                            in_trade = False
                            break
                        if df_4h['High'].iloc[j] >= target:
                            exit_price = target
                            pnl = risk_amount_per_trade * RISK_REWARD_RATIO
                            pnl_after_costs = pnl - commission_cost
                            trades.append({
                                'Ticker': ticker_symbol,
                                'EntryTime': entry_time, 'ExitTime': df_4h.index[j],
                                'EntryPrice': entry_price, 'ExitPrice': exit_price,
                                'Outcome': 'win', 'PnL': pnl_after_costs
                            })
                            in_trade = False
                            break
                
                if in_trade:
                    # The trade is still open at the end of the backtest period
                    pnl = (df_4h['Close'].iloc[-1] - entry_price) * position_size
                    pnl_after_costs = pnl - commission_cost
                    trades.append({
                        'Ticker': ticker_symbol,
                        'EntryTime': entry_time, 'ExitTime': df_4h.index[-1],
                        'EntryPrice': entry_price, 'ExitPrice': df_4h['Close'].iloc[-1],
                        'Outcome': 'pending', 'PnL': pnl_after_costs
                    })

    trades_df = pd.DataFrame(trades)
    print(f"✅ Backtest for {ticker_symbol} complete.")
    
    # === PLOT THE CHART FOR BTC-USD and ETH-USD ===
    if ticker_symbol in ['BTC/USDT', 'ETH/USDT']:
        print(f"➡️ Generating chart for {ticker_symbol}...")
        
        # Correctly use the full series with NaNs for plotting
        buy_signals = entry_signals

        # Create addplots for the MA and RSI
        ap = [
            mpf.make_addplot(df_4h['MA'], color='blue', panel=0, title='250 MA'),
            mpf.make_addplot(df_4h['RSI'], color='purple', panel=1, ylabel='RSI'),
            mpf.make_addplot(buy_signals, type='scatter', marker='^', markersize=200, color='green', panel=0)
        ]

        # Plot the chart
        mpf.plot(
            df_4h,
            type='candle',
            style='yahoo',
            title=f'{ticker_symbol} Candlestick Chart with Trading Signals',
            addplot=ap,
            volume=False,
            mav=(200),
            tight_layout=True,
            figsize=(15, 10)
        )

    # === PRINT INDIVIDUAL TICKER RESULTS ===
    if not trades_df.empty:
        total_trades = len(trades_df)
        wins = trades_df[trades_df['Outcome'] == 'win']
        losses = trades_df[trades_df['Outcome'] == 'loss']
        
        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        net_pl = trades_df['PnL'].sum()
        
        trades_df['ExitTime'] = pd.to_datetime(trades_df['ExitTime'])
        daily_returns = trades_df.set_index('ExitTime')['PnL'].resample('D').sum()
        std_dev_returns = daily_returns.std()
        sharpe_ratio = (daily_returns.mean() / std_dev_returns) * np.sqrt(252) if std_dev_returns != 0 else float('inf')

        # Calculate Max Drawdown for Ticker
        equity = capital + trades_df['PnL'].cumsum()
        drawdown = (equity.cummax() - equity) / equity.cummax()
        max_drawdown = drawdown.max() * 100 if not drawdown.empty else 0

        print(f"\n--- 📈 Performance for {ticker_symbol} ---")
        ticker_summary = {
            'Ticker': [ticker_symbol],
            'Total Trades': [total_trades],
            'Winning Trades': [len(wins)],
            'Losing Trades': [len(losses)],
            'Win Rate': [win_rate],
            'Net P/L': [net_pl],
            'Sharpe Ratio': [sharpe_ratio],
            'Max Drawdown (%)': [max_drawdown]
        }
        summary_df = pd.DataFrame(ticker_summary)
        print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))
    
    return trades_df

# === MAIN EXECUTION LOOP FOR THE PORTFOLIO ===
all_trades_df = pd.DataFrame()

for ticker in tickers:
    results = run_backtest_for_ticker(ticker)
    if results is not None and not results.empty:
        all_trades_df = pd.concat([all_trades_df, results], ignore_index=True)

# === PORTFOLIO-LEVEL METRICS ===
if not all_trades_df.empty:
    total_trades = len(all_trades_df)
    wins = all_trades_df[all_trades_df['Outcome'] == 'win']
    losses = all_trades_df[all_trades_df['Outcome'] == 'loss']
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    net_pl = all_trades_df['PnL'].sum()
    
    daily_returns = all_trades_df.set_index('ExitTime')['PnL'].resample('D').sum()
    std_dev_returns = daily_returns.std()
    sharpe_ratio = (daily_returns.mean() / std_dev_returns) * np.sqrt(252) if std_dev_returns != 0 else float('inf')
    
    # Calculate Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (daily_returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else float('inf')
    
    # Calculate Max Drawdown
    all_trades_df = all_trades_df.sort_values(by='ExitTime')
    equity = capital + all_trades_df['PnL'].cumsum()
    drawdown = (equity.cummax() - equity) / equity.cummax()
    max_drawdown = drawdown.max() * 100
    
    # Calculate Calmar Ratio
    final_equity = capital + net_pl
    cagr = ((final_equity / capital) ** (1 / 2)) - 1 if capital > 0 else 0
    calmar_ratio = cagr / (max_drawdown / 100) if max_drawdown != 0 else float('inf')
    
    # Calculate Max Profit per Trade
    max_profit_trade = all_trades_df['PnL'].max() if not all_trades_df.empty else 0
    
    print("\n=== 📊 Overall Portfolio Performance ===")
    portfolio_summary = {
        'Total Trades': [total_trades],
        'Winning Trades': [len(wins)],
        'Losing Trades': [len(losses)],
        'Win Rate': [win_rate],
        'Net P/L': [net_pl],
        'Profit Factor': [all_trades_df[all_trades_df['PnL'] > 0]['PnL'].sum() / abs(all_trades_df[all_trades_df['PnL'] < 0]['PnL'].sum()) if not all_trades_df[all_trades_df['PnL'] < 0].empty else float('inf')],
        'Sharpe Ratio': [sharpe_ratio],
        'Sortino Ratio': [sortino_ratio],
        'Max Drawdown (%)': [max_drawdown],
        'Calmar Ratio': [calmar_ratio],
        'Max Profit (Single Trade)': [max_profit_trade]
    }
    summary_df = pd.DataFrame(portfolio_summary)
    print(tabulate(summary_df, headers='keys', tablefmt='fancy_grid', floatfmt=".2f"))