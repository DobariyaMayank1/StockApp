# --- train.py ---
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Step 1: Downloading historical data for Reliance from yfinance...")
# Fetch historical data directly from Yahoo Finance to train our model locally
ticker = yf.Ticker("RELIANCE.NS")
df = ticker.history(period="max") # Gets all available max history

# Reset index to make Date a column and sort chronologically
df = df.reset_index()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

print(f"Downloaded {len(df)} days of data.")

print("\nStep 2: Engineering features (Lags & Rolling Averages)...")
# Recreate your exact feature logic
df['Close_Lag1'] = df['Close'].shift(1)
df['Close_Lag2'] = df['Close'].shift(2)
df['Rolling_Mean_30'] = df['Close'].shift(1).rolling(window=30).mean()
df['Rolling_Mean_200'] = df['Close'].shift(1).rolling(window=200).mean()

# Drop rows with NaN values
df = df.dropna(subset=['Close_Lag1', 'Close_Lag2', 'Rolling_Mean_30', 'Rolling_Mean_200'])

# Define features and targets matching your Kaggle notebook
features = ['Close_Lag1', 'Close_Lag2', 'Rolling_Mean_30', 'Rolling_Mean_200']
targets = ['Open', 'High', 'Low', 'Close']

X = df[features]
y = df[targets]

print("\nStep 3: Training the Multi-Output Random Forest Model...")
# Initialize and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("Training complete!")

print("\nStep 4: Saving the model locally...")
# Save it directly into your StockApp folder
joblib.dump(model, 'multi_output_stock_model.pkl')
print("Successfully generated 'multi_output_stock_model.pkl' locally!")