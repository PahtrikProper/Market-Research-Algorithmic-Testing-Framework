import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from itertools import product
from tqdm import tqdm

# ========== USER CONFIG ==========
BYBIT_API_KEY = "here"
BYBIT_API_SECRET = "here"
symbol = "SOLUSDT"
category = "spot"        # For margin, keep 'spot'
agg_minutes = 14
backtest_days = 7
STARTING_BALANCE = 1000
bybit_fee = 0.001
imbalance_range = range(1, 35)
ema_range = range(5, 161, 5)
tp_range = [x / 100 for x in range(3, 41)]  # 3% → 40%

# ========== DATA FETCH & AGGREGATE ==========
def fetch_bybit_1m(symbol, category, limit=1000, days=60):
    end = int(datetime.utcnow().timestamp())
    start = end - days * 24 * 60 * 60
    df_list = []
    while start < end:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": "1",
            "start": start * 1000,
            "limit": limit
        }
        resp = requests.get(url, params=params).json()
        rows = resp["result"]["list"]
        if not rows:
            break
        df = pd.DataFrame(rows, columns=[
            "timestamp", "Open", "High", "Low", "Close", "Volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df = df.sort_values("timestamp")
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = df[col].astype(float)
        df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
        df.set_index("timestamp", inplace=True)
        df_list.append(df)
        start = int(df.index[-1].timestamp()) + 60
        time.sleep(0.3)
    return pd.concat(df_list).sort_index()

def resample_candles(df, agg_minutes=14):
    df = df.resample(f"{agg_minutes}T").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna()
    return df

def fetch_data(symbol, category):
    df_1m = fetch_bybit_1m(symbol, category, days=backtest_days)
    df_agg = resample_candles(df_1m, agg_minutes)
    return df_agg.tail(200)

def bybit_fee_fn(trade_value):
    return trade_value * bybit_fee

# ========== STRATEGY LOGIC (unchanged) ==========
def run_backtest(df, imbalance_lookback, ema_len, take_profit_pct):
    data = df.copy()
    data["ema"] = data["Close"].ewm(span=ema_len).mean()
    data["ema_up"] = data["ema"] > data["ema"].shift(1)
    data["ema_down"] = data["ema"] < data["ema"].shift(1)
    data["above_ema"] = data["Close"] > data["ema"]
    data["below_ema"] = data["Close"] < data["ema"]
    data["imb_low"] = data["Low"].rolling(imbalance_lookback).min()
    data["imb_high"] = data["High"].rolling(imbalance_lookback).max()
    data["tradable"] = True

    balance = STARTING_BALANCE
    equity_curve = []
    position = 0
    coins = 0
    entry_price = None
    tp_price = None
    wins = 0
    losses = 0
    win_sizes = []
    loss_sizes = []

    for i in range(imbalance_lookback + 2, len(data)):
        row = data.iloc[i]
        open_, high, low, close = row["Open"], row["High"], row["Low"], row["Close"]

        # ENTRY
        long_cond = (
            low <= row["imb_low"] and
            close > open_ and
            row["ema_up"] and
            row["above_ema"] and
            row["tradable"] and
            position == 0
        )

        if long_cond:
            entry_price = close
            coins = balance // entry_price
            if coins > 0:
                buy_value = entry_price * coins
                buy_fee = bybit_fee_fn(buy_value)
                balance -= (buy_value + buy_fee)
                tp_price = entry_price * (1 + take_profit_pct)
                position = 1

        # EXIT
        if position == 1:
            if high >= tp_price:
                exit_price = tp_price
                sell_value = exit_price * coins
                sell_fee = bybit_fee_fn(sell_value)
                balance += (sell_value - sell_fee)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                wins += 1
                win_sizes.append(pnl_pct)
                position = 0
                coins = 0
                entry_price = None
            else:
                short_cond = (
                    high >= row["imb_high"] and
                    close < open_ and
                    row["ema_down"] and
                    row["below_ema"] and
                    row["tradable"]
                )
                if short_cond:
                    exit_price = close
                    sell_value = exit_price * coins
                    sell_fee = bybit_fee_fn(sell_value)
                    balance += (sell_value - sell_fee)
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    if pnl_pct > 0:
                        wins += 1
                        win_sizes.append(pnl_pct)
                    else:
                        losses += 1
                        loss_sizes.append(pnl_pct)
                    position = 0
                    coins = 0
                    entry_price = None

        if position == 1:
            current_equity = balance + coins * close
        else:
            current_equity = balance
        equity_curve.append(current_equity)

    final_balance = equity_curve[-1]
    pnl_value = final_balance - STARTING_BALANCE
    pnl_pct = (pnl_value / STARTING_BALANCE) * 100
    avg_win = np.mean(win_sizes) if win_sizes else 0
    avg_loss = np.mean(loss_sizes) if loss_sizes else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
    rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else None
    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365*24*60/agg_minutes) if returns.std() != 0 else 0
    equity_series = pd.Series(equity_curve)
    roll_max = equity_series.cummax()
    drawdown = ((equity_series - roll_max) / roll_max).min() * 100

    return (
        pnl_pct, pnl_value, final_balance,
        avg_win, avg_loss, win_rate, rr_ratio,
        sharpe, drawdown, wins, losses
    )

# ========== OPTIMIZATION LOOP ==========
def optimize_symbol(symbol):
    df = fetch_data(symbol, category)
    best = None
    total = len(imbalance_range) * len(ema_range) * len(tp_range)
    for imb, ema, tp in tqdm(
        product(imbalance_range, ema_range, tp_range),
        total=total,
        desc=f"Optimizing {symbol}",
        ncols=80
    ):
        (pnl_pct, pnl_value, final_balance,
         avg_win, avg_loss, win_rate, rr_ratio,
         sharpe, drawdown, wins, losses) = run_backtest(df, imb, ema, tp)
        if best is None or pnl_pct > best["pnl_pct"]:
            best = {
                "symbol": symbol,
                "imbalance": imb,
                "ema": ema,
                "tp_pct": tp,
                "pnl_pct": pnl_pct,
                "pnl_value": pnl_value,
                "final_balance": final_balance,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "win_rate": win_rate,
                "rr_ratio": rr_ratio,
                "sharpe": sharpe,
                "drawdown": drawdown,
                "wins": wins,
                "losses": losses,
            }
    return best

# ========== MAIN OPTIMIZE + PRINT RESULTS ==========
print("\nRunning optimizer...\n")
result = optimize_symbol(symbol)
df_results = pd.DataFrame([result])
for col, f in [
    ("tp_pct", lambda x: f"{x*100:.2f}%"),
    ("pnl_pct", lambda x: f"{x:.2f}%"),
    ("avg_win", lambda x: f"{x:.2f}%"),
    ("avg_loss", lambda x: f"{x:.2f}%"),
    ("win_rate", lambda x: f"{x:.2f}%"),
    ("rr_ratio", lambda x: f"{x:.2f}" if x is not None else "N/A"),
    ("sharpe", lambda x: f"{x:.2f}"),
    ("drawdown", lambda x: f"{x:.2f}%"),
    ("pnl_value", lambda x: f"${x:,.2f}"),
    ("final_balance", lambda x: f"${x:,.2f}"),
]: df_results[col] = df_results[col].map(f)
print("\n==================== FINAL RESULTS ====================\n")
print(df_results.to_string(index=False))
print("\n=======================================================\n")

# ========== LIVE DEMO ISOLATED MARGIN TRADE LOOP ==========
from pybit.unified_trading import HTTP

session = HTTP(
    testnet=True,
    api_key='here',
    api_secret='here',
)

# 1. Ensure Isolated Margin mode is set for the symbol
print(f"Enabling Isolated Margin for {symbol} ...")
session.set_margin_mode(
    category=category,
    symbol=symbol,
    tradeMode=1,  # 1 = Isolated, 0 = Cross
)

imbalance = result["imbalance"]
ema_len = result["ema"]
tp_pct = result["tp_pct"] if isinstance(result["tp_pct"], float) else float(result["tp_pct"].replace("%", ""))/100

print(f"\n--- Starting live demo trading on {symbol} | Isolated Margin | 14m ---\n")
while True:
    # Fetch and aggregate latest data
    df_1m = fetch_bybit_1m(symbol, category, days=3)
    df = resample_candles(df_1m, agg_minutes)

    data = df.copy()
    data["ema"] = data["Close"].ewm(span=ema_len).mean()
    data["ema_up"] = data["ema"] > data["ema"].shift(1)
    data["above_ema"] = data["Close"] > data["ema"]
    data["imb_low"] = data["Low"].rolling(imbalance).min()

    row = data.iloc[-1]
    open_, close, low = row["Open"], row["Close"], row["Low"]

    long_cond = (
        low <= row["imb_low"]
        and close > open_
        and row["ema_up"]
        and row["above_ema"]
    )

    if long_cond:
        print(f"[{datetime.utcnow()}] BUY SIGNAL: {symbol} @ {close}")
        # Fetch demo balance (Isolated Margin)
        # Note: For real margin size, check your Isolated Margin wallet for this symbol!
        w = session.get_wallet_balance(accountType="UNIFIED")
        balance = float(w["result"]["list"][0]["totalEquity"])
        # Only trade 0.5% of balance
        trade_value = balance * 0.005
        qty = round(trade_value / close, 4)

        
        # Market buy (Isolated Margin)
        order = session.place_order(
            category=category,
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=qty,
            isLeverage=1,
            tradeMode=1,   # 1 = Isolated
            leverage=2,    # Change leverage as desired
        )
        print(order)
        tp_price = close * (1 + tp_pct)
        print(f"TP set at {tp_price:.4f}")
        while True:
            ticker = session.get_tickers(category=category, symbol=symbol)
            last = float(ticker["result"]["list"][0]["lastPrice"])
            if last >= tp_price:
                print(f"[{datetime.utcnow()}] TP HIT at {last:.4f}. Selling...")
                sell_order = session.place_order(
                    category=category,
                    symbol=symbol,
                    side="Sell",
                    orderType="Market",
                    qty=qty,
                    isLeverage=1,
                    tradeMode=1,
                    leverage=2,
                )
                print(sell_order)
                break
            time.sleep(10)
    time.sleep(agg_minutes * 60)
