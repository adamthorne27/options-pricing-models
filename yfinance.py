import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# 1. Get option chain data from Yahoo for SPY
ticker = yf.Ticker("SPY")
expirations = ticker.options[:10]  # Get a few expirations (limit to keep it fast)

surface_data = []

for exp in expirations:
    try:
        chain = ticker.option_chain(exp)
        calls = chain.calls

        # 2. Extract useful info
        strikes = calls['strike']
        ivs = calls['impliedVolatility']
        exp_date = datetime.strptime(exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.today()).days / 365.0

        for strike, iv in zip(strikes, ivs):
            if iv is not None and 0 < iv < 5:  # sanity filter
                surface_data.append({
                    "Strike": strike,
                    "Maturity": days_to_exp,
                    "ImpliedVol": iv
                })

    except Exception as e:
        print(f"Failed to get data for {exp}: {e}")

# 3. Create a DataFrame
df = pd.DataFrame(surface_data)

# Pivot for surface plot: Z=IV, X=Strike, Y=Maturity
pivot = df.pivot_table(index="Maturity", columns="Strike", values="ImpliedVol")

# Ensure grid consistency
pivot = pivot.sort_index().sort_index(axis=1)
X = pivot.columns.values
Y = pivot.index.values
Z = pivot.values

# 4. Plot surface
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
fig.update_layout(
    title="Implied Volatility Surface (SPY Options)",
    scene=dict(
        xaxis_title="Strike Price",
        yaxis_title="Time to Maturity (Years)",
        zaxis_title="Implied Volatility"
    ),
    height=700
)
fig.show()
