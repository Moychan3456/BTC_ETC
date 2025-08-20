# Quantitative Trading Strategy – BTC & ETH Backtest (2015–2025)

## 📌 Project Overview  
This project is a **systematic trading backtest** applied to **BTC/USDT and ETH/USDT** using Python.  
The goal was to test a rules-based breakout strategy with realistic market assumptions (slippage, commission, position sizing).  

The backtest resamples **1H data to 4H** intervals and applies multi-factor entry rules, portfolio allocation, and full equity curve tracking.

> ⚠️ **Note:** This project is still a **work in progress**. The 4H strategy is functional, but I am currently refining it by integrating **multi-timeframe signals (daily & weekly)** and additional risk filters to make it closer to my live trading approach.

---

## ⚙️ Methodology
- **Entry Rules:**  
  - Double bullish candle confirmation  
  - Price above **250-period Moving Average (MA)**  
  - RSI filter (>45)  
  - ATR & ADX regime filters  

- **Execution Modeling:**  
  - Realistic slippage (0.5–1.5 pips)  
  - Commission: $7/lot  
  - ATR-adjusted position sizing  
  - Dynamic allocation between BTC & ETH  

---

## 📊 Results Summary (2015–2025)

### BTC/USDT  
- **Trades:** 1,536  
- **Win Rate:** 23.31%  
- **Sharpe Ratio:** **1.05** (improved with 250 MA filter)  
- **Max Drawdown:** 20.62%  
- **Net P/L:** $216,499  

### ETH/USDT  
- **Trades:** 1,511  
- **Win Rate:** 22.57%  
- **Sharpe Ratio:** **0.98** (slightly reduced when applying the 250 MA filter)  
- **Max Drawdown:** 13.97%  
- **Net P/L:** $182,244  

### Portfolio (BTC + ETH)  
- **Total Trades:** 3,047  
- **Win Rate:** 22.94%  
- **Sharpe Ratio:** 1.26  
- **Sortino Ratio:** 4.13  
- **Max Drawdown:** 16.05%  
- **Net P/L:** $398,744  

---

## 🔍 Key Observations
- Adding the **250-period Moving Average filter** significantly **improves BTC/USDT performance** (higher Sharpe and smoother equity curve).  
- The same filter slightly **reduces ETH/USDT performance**, likely due to different market structure and volatility regime.  
- Portfolio-level metrics still benefit, as BTC dominates the risk-adjusted return profile.  

---

## 🚧 Next Steps
- Incorporating **multi-timeframe confirmation** (Daily + Weekly filters).  
- Expanding to **NSE Options & FX pairs** for cross-market testing.  
- Optimizing allocation between BTC & ETH dynamically.  
- Further refining stop-loss/target placement to improve risk-adjusted returns.  

---

## 📂 Repository Structure
```
/project-root
│── backtest_strategy.py    # Main Python strategy code
│── results_output.txt      # Full console output of backtest runs
│── README.md               # Project documentation (this file)
```

---

## 📝 Notes for Recruiters
This project demonstrates:  
- Ability to **code and backtest strategies in Python**.  
- Handling of **realistic execution assumptions** (slippage, commissions).  
- Use of **risk metrics** (Sharpe, Sortino, Max Drawdown, Profit Factor).  
- An understanding of how market filters (250 MA) **affect performance across assets differently**.  
