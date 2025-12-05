import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product
from tqdm import tqdm

# ===================================================================
# SETTINGS
# ===================================================================
STARTING_BALANCE = 1_000
symbols = ["NDQ.AX"]
interval = "1d"

imbalance_range = range(1, 35)
ema_range = range(5, 161, 5)
tp_range = [x / 100 for x in range(3, 41)]  # 3% → 40%


# ===================================================================
# COMMSEC BROKERAGE FEE MODEL
# ===================================================================
def commsec_fee(trade_value):
    if trade_value <= 1000:
        return 5.00
    elif trade_value <= 10000:
        return 10.00
    elif trade_value <= 25000:
        return 19.95
    else:
        return trade_value * 0.0012


# ===================================================================
# FETCH DATA
# ===================================================================
def fetch_data(symbol):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=300)

    df = yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df.tail(200)


# ===================================================================
# REALISTIC BACKTEST FUNCTION
# ===================================================================
def run_backtest(df, imbalance_lookback, ema_len, take_profit_pct):

    data = df.copy()

    data["ema"] = data["Close"].ewm(span=ema_len).mean()
    data["ema_up"] = data["ema"] > data["ema"].shift(1)
    data["ema_down"] = data["ema"] < data["ema"].shift(1)
    data["above_ema"] = data["Close"] > data["ema"]
    data["below_ema"] = data["Close"] < data["ema"]

    data["imb_low"] = data["Low"].rolling(imbalance_lookback).min()
    data["imb_high"] = data["High"].rolling(imbalance_lookback).max()

    data["year"] = data.index.year
    data["tradable"] = data["year"] == 2025

    balance = STARTING_BALANCE
    equity_curve = []  # FIX: start empty, we will append proper equity each bar

    position = 0
    shares = 0
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
            shares = balance // entry_price
            if shares > 0:
                buy_value = entry_price * shares
                buy_fee = commsec_fee(buy_value)
                balance -= (buy_value + buy_fee)

                tp_price = entry_price * (1 + take_profit_pct)
                position = 1

        # EXIT
        if position == 1:

            # TAKE PROFIT
            if high >= tp_price:
                exit_price = tp_price
                sell_value = exit_price * shares
                sell_fee = commsec_fee(sell_value)
                balance += (sell_value - sell_fee)

                pnl_pct = (exit_price - entry_price) / entry_price * 100
                wins += 1
                win_sizes.append(pnl_pct)

                position = 0
                shares = 0
                entry_price = None

            else:
                # OPPOSITE EXIT
                short_cond = (
                    high >= row["imb_high"] and
                    close < open_ and
                    row["ema_down"] and
                    row["below_ema"] and
                    row["tradable"]
                )

                if short_cond:
                    exit_price = close
                    sell_value = exit_price * shares
                    sell_fee = commsec_fee(sell_value)
                    balance += (sell_value - sell_fee)

                    pnl_pct = (exit_price - entry_price) / entry_price * 100

                    if pnl_pct > 0:
                        wins += 1
                        win_sizes.append(pnl_pct)
                    else:
                        losses += 1
                        loss_sizes.append(pnl_pct)

                    position = 0
                    shares = 0
                    entry_price = None

        # ============================================================
        # FIX: Mark-to-market equity every bar (REAL equity tracking)
        # ============================================================
        if position == 1:
            current_equity = balance + shares * close
        else:
            current_equity = balance

        equity_curve.append(current_equity)

    # Final stats
    final_balance = equity_curve[-1]   # FIX: use true equity, not cash balance
    pnl_value = final_balance - STARTING_BALANCE
    pnl_pct = (pnl_value / STARTING_BALANCE) * 100

    avg_win = np.mean(win_sizes) if win_sizes else 0
    avg_loss = np.mean(loss_sizes) if loss_sizes else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    rr_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else None

    returns = pd.Series(equity_curve).pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    # Drawdown (now correct because equity_curve is correct)
    equity_series = pd.Series(equity_curve)
    roll_max = equity_series.cummax()
    drawdown = ((equity_series - roll_max) / roll_max).min() * 100

    return (
        pnl_pct, pnl_value, final_balance,
        avg_win, avg_loss, win_rate, rr_ratio,
        sharpe, drawdown, wins, losses
    )

# ===================================================================
# OPTIMIZE
# ===================================================================
def optimize_symbol(symbol):
    df = fetch_data(symbol)
    best = None

    for imb, ema, tp in product(imbalance_range, ema_range, tp_range):

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


# ===================================================================
# RUN ALL SYMBOLS
# ===================================================================
results = []

print("\nRunning realistic optimizer...\n")

for sym in symbols:
    print(f"Optimizing {sym} ...")
    results.append(optimize_symbol(sym))

df_results = pd.DataFrame(results).sort_values("pnl_pct", ascending=False)

# ===================================================================
# FORMATTING
# ===================================================================
df_results["tp_pct"]        = df_results["tp_pct"].map(lambda x: f"{x*100:.2f}%")
df_results["pnl_pct"]       = df_results["pnl_pct"].map(lambda x: f"{x:.2f}%")
df_results["avg_win"]       = df_results["avg_win"].map(lambda x: f"{x:.2f}%")
df_results["avg_loss"]      = df_results["avg_loss"].map(lambda x: f"{x:.2f}%")
df_results["win_rate"]      = df_results["win_rate"].map(lambda x: f"{x:.2f}%")
df_results["rr_ratio"]      = df_results["rr_ratio"].map(lambda x: f"{x:.2f}" if x is not None else "N/A")
df_results["sharpe"]        = df_results["sharpe"].map(lambda x: f"{x:.2f}")
df_results["drawdown"]      = df_results["drawdown"].map(lambda x: f"{x:.2f}%")
df_results["pnl_value"]     = df_results["pnl_value"].map(lambda x: f"${x:,.2f}")
df_results["final_balance"] = df_results["final_balance"].map(lambda x: f"${x:,.2f}")

print("\n==================== FINAL RESULTS ====================\n")
print(df_results.to_string(index=False))
print("\n=======================================================\n")
