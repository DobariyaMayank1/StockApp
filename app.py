from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the model we trained in Kaggle
MODEL_PATH = 'multi_output_stock_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: multi_output_stock_model.pkl not found in the current directory!")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_results = None
    error_message = None
    ticker_symbol = "RELIANCE.NS" # Default stock ticker

    if request.method == 'POST':
        ticker_symbol = request.form.get('ticker', 'RELIANCE.NS').upper()
        
        try:
            # 1. Fetch live historical data automatically using yfinance
            # We fetch 1 year of data to confidently calculate the 200-day rolling average
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            
            if df.empty:
                error_message = f"No data found for ticker symbol: {ticker_symbol}"
            else:
                # 2. Extract and calculate indicators from the live data
                close_series = df['Close']
                
                # Get the most recent finished closing prices
                close_lag1 = float(close_series.iloc[-1])
                close_lag2 = float(close_series.iloc[-2])
                
                # Calculate the exact rolling averages using pandas
                rolling_30 = float(close_series.rolling(window=30).mean().iloc[-1])
                rolling_200 = float(close_series.rolling(window=200).mean().iloc[-1])
                
                # 3. Structure the data exactly how our model expects it
                features = ['Close_Lag1', 'Close_Lag2', 'Rolling_Mean_30', 'Rolling_Mean_200']
                input_data = pd.DataFrame([{
                    'Close_Lag1': close_lag1,
                    'Close_Lag2': close_lag2,
                    'Rolling_Mean_30': rolling_30,
                    'Rolling_Mean_200': rolling_200
                }])[features]
                
                # 4. Run inference using our loaded model
                if model:
                    pred = model.predict(input_data)[0]
                    prediction_results = {
                        'open': round(pred[0], 2),
                        'high': round(pred[1], 2),
                        'low': round(pred[2], 2),
                        'close': round(pred[3], 2),
                        'as_of_date': df.index[-1].strftime('%Y-%m-%d')
                    }
                else:
                    error_message = "Model file is missing. Please place the .pkl file in the root folder."
                    
        except Exception as e:
            error_message = f"An error occurred while fetching data: {str(e)}"

    return render_template('index.html', prediction=prediction_results, error=error_message, ticker=ticker_symbol)

if __name__ == '__main__':
    app.run(debug=True)