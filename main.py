import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Black-Scholes Model formula
# print("helloo")
def black_scholes(S, K, T, r, sigma, option_type='ce'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'ce':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'pe':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'ce' for call or 'pe' for put.")

    return price

# Load your CSV file
df = pd.read_csv('OPTIDX_NIFTY_CE_18-Aug-2024_TO_19-Aug-2024.csv')

# Convert the 'Date' and 'Expiry' columns to datetime
df['Date'] = pd.to_datetime(df['Date  '])
df['Expiry'] = pd.to_datetime(df['Expiry  '])

# Calculate time to maturity (T) in years
df['T'] = (df['Expiry'] - df['Date']).dt.days / 365.0

# Define the risk-free rate (example: 10% or 0.10)
r = 0.10

# Estimate volatility, for simplicity, we can assume a fixed volatility
# or calculate it based on historical data.
df['Volatility'] = 0.3  # Example: 30% or 0.3

# Calculate the Black-Scholes price for each option
df['Predicted_Price'] = df.apply(
    lambda row: black_scholes(row['LTP  '], row['Strike Price  '], row['T'], r, row['Volatility'], row['Option type  '].strip().lower()),
    axis=1
)

# Calculate accuracy metrics
actual_prices = df['LTP  ']
predicted_prices = df['Predicted_Price']

# Mean Absolute Error
mae = mean_absolute_error(actual_prices, predicted_prices)

# Mean Squared Error
mse = mean_squared_error(actual_prices, predicted_prices)

# Root Mean Squared Error
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Save the results to a new CSV file
df.to_csv('black_scholes_predictions_with_accuracy.csv', index=False)
