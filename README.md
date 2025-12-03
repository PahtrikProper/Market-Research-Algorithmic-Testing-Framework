# Market-Research-Algorithmic-Testing-Framework

## **Purpose**

This project implements a **market-research and algorithm-testing framework** designed for **private, non-commercial use** by the author.
It focuses on analysing historical market behaviour, testing pattern-based hypotheses, and evaluating trading-rule performance under realistic cost assumptions.

This software is **not a trading platform**, does **not provide financial advice**, and is **not intended for use by others**.

---

# 🔒 **Important Disclaimer (Mandatory for Compliance)**

* This system is used **solely for the author’s personal market research**.
* It **does not execute trades for third parties**.
* It does **not provide investment recommendations**, signals, or financial advice.
* It does **not pool money**, hold client funds, or offer brokerage services.
* It is **not sold, licensed, or provided to customers**.
* All decisions made using this tool are **private decisions made by the author**.
* The strategy described is **not guaranteed**, **not universal**, and may not be appropriate for others.

This project is a **technical experiment** in pattern recognition and algorithm evaluation.

---

# 📊 **System Overview**

The framework performs:

* Historical data collection
* Indicator computation
* Rule-based simulation
* Fee-aware modelling
* Multi-parameter optimization
* Performance reporting

The goal is to help the author understand how certain market patterns behave over time under specific predefined conditions.

---

# 🔍 **Core Research Idea**

This research explores whether certain **price imbalance zones** combined with **trend-following conditions** produce consistent behaviour in historical data.

These concepts are derived from:

* Observed market structure
* Repetitive patterns in price movement
* Hypotheses about supply/demand imbalance
* Trend continuation and rejection patterns

The framework tests these hypotheses **mathematically**, without implying real-world tradability or suitability.

---

# 🧠 **Strategy Concept (Research Only)**

### **1. Imbalance Analysis**

Price occasionally moves in a way that indicates potential short-term supply or demand imbalance.
The framework identifies these zones using rolling highs/lows.

### **2. Trend Alignment**

A simple moving trend indicator (EMA) is used to determine whether price direction aligns with the hypothesized imbalance reaction.

### **3. Entry/Exit Conditions**

Rules are applied such that:

* Entries occur when price interacts with imbalance levels *and* trend conditions align.
* Exits occur via:

  * a predefined gain threshold (take-profit), or
  * rule-driven reversal signals.

### **4. No Recommendations**

These rules are **not recommendations**.
They are used only to test if the author’s pattern hypotheses show statistical properties worth further private research.

---

# 💰 **Realistic Fee Simulation**

To ensure analytical accuracy, the system includes fee-modelling based on publicly known brokerage fee structures (e.g., CommSec tiered pricing).
This helps ensure results reflect realistic frictional costs rather than idealized scenarios.

This fee model is only a **simulation** and may not reflect future, current, or individual brokerage conditions.

---

# 🧪 **Parameter Optimization**

A systematic brute-force optimizer evaluates combinations of:

* Imbalance lookback period
* EMA length
* Take-profit percentages

Across multiple ASX-listed symbols, the system reports:

* Total return percentage
* Dollar-modelled performance
* Number of positive/negative outcomes
* Average gain on positive outcomes

All results are **historical-model outputs only** and do not imply future performance.

---

# 🔧 **Software Architecture**

The system includes:

* OHLC data ingestion
* Indicator engine
* Backtest processor
* Brokerage-fee simulation layer
* Multi-symbol optimizer
* Human-readable summary output

No live trading, account connectivity, or order-routing is present.

---

# ⚠️ **Not a Live Trading Tool**

Although suitable for personal research, this system:

* Does **not** place trades
* Does **not** include broker APIs
* Does **not** connect to brokerage accounts
* Should **not** be used to manage real accounts for others

Future extensions, if any, will remain for **private use only**.

---

# 🛑 **Regulatory Position (Australia)**

Because this system:

* Trades **only the author’s own funds**
* Provides **no services to others**
* Does **not** recommend or advise
* Does **not** automate client orders
* Is **not sold commercially**

…it **does not fall under AFSL (Australian Financial Services Licence) requirements**, as confirmed by ASIC guidelines for personal investing and non-advice software tools.

---

# 📦 **Who This Project Is For**

This framework is appropriate for:

* The author’s own mathematical curiosity
* Software development experimentation
* Market structure research
* Algorithm evaluation
* Non-commercial learning purposes

It is **not intended for any external user** or commercial deployment.

---

# 📎 **Summary**

This project is:

* A private research tool
* A technical algorithm testing system
* Fee-aware and realistic
* Non-commercial
* Non-advisory
* Not a financial product
* Not intended for external users

It represents the author’s ongoing exploration of market behaviour through software engineering and mathematical analysis.

