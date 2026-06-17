from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import joblib
import os

app = Flask(__name__)

MODEL_PATH = 'multi_output_stock_model.pkl'
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_results = None
    comparison_results = None
    error_message = None
    ticker_symbol = "RELIANCE.NS"
    mode = "live"  # Can be 'live' or 'historical'
    selected_date = ""

    if request.method == 'POST':
        ticker_symbol = request.form.get('ticker', 'RELIANCE.NS').upper()
        mode = request.form.get('mode', 'live')
        selected_date = request.form.get('historical_date', '')

        try:
            stock = yf.Ticker(ticker_symbol)
            
            if mode == 'live':
                # --- LIVE MODE: Predict Tomorrow ---
                df = stock.history(period="1y")
                if df.empty:
                    error_message = f"No data found for ticker symbol: {ticker_symbol}"
                else:
                    df = df.reset_index()
                    close_series = df['Close']
                    
                    close_lag1 = float(close_series.iloc[-1])
                    close_lag2 = float(close_series.iloc[-2])
                    rolling_30 = float(close_series.rolling(window=30).mean().iloc[-1])
                    rolling_200 = float(close_series.rolling(window=200).mean().iloc[-1])
                    
                    features = ['Close_Lag1', 'Close_Lag2', 'Rolling_Mean_30', 'Rolling_Mean_200']
                    input_data = pd.DataFrame([{'Close_Lag1': close_lag1, 'Close_Lag2': close_lag2, 'Rolling_Mean_30': rolling_30, 'Rolling_Mean_200': rolling_200}])[features]
                    
                    if model:
                        pred = model.predict(input_data)[0]
                        prediction_results = {
                            'open': round(pred[0], 2), 'high': round(pred[1], 2),
                            'low': round(pred[2], 2), 'close': round(pred[3], 2),
                            'as_of_date': df['Date'].iloc[-1].strftime('%Y-%m-%d')
                        }
            
            elif mode == 'historical':
                # --- HISTORICAL MODE: Backtest a Past Date ---
                if not selected_date:
                    error_message = "Please select a valid historical date."
                else:
                    # Fetch extra history to make sure we have 200 days prior to the target date
                    df = stock.history(period="max").reset_index()
                    df['Date_Str'] = df['Date'].dt.strftime('%Y-%m-%d')
                    
                    # Find the row corresponding to the user's chosen target date
                    target_row = df[df['Date_Str'] == selected_date]
                    
                    if target_row.empty:
                        error_message = f"The date {selected_date} was not a trading day or no data exists."
                    else:
                        target_index = target_row.index[0]
                        
                        # We need at least 200 rows before this date to compute our indicators
                        if target_index < 200:
                            error_message = "Not enough historical data before this date to compute 200-day averages."
                        else:
                            # Extract actual performance metrics on that historical day
                            actual_data = {
                                'open': round(float(target_row['Open'].values[0]), 2),
                                'high': round(float(target_row['High'].values[0]), 2),
                                'low': round(float(target_row['Low'].values[0]), 2),
                                'close': round(float(target_row['Close'].values[0]), 2)
                            }
                            
                            # Isolate data entirely BEFORE the target date to prevent data leakage
                            historical_context = df.iloc[:target_index]
                            close_series = historical_context['Close']
                            
                            close_lag1 = float(close_series.iloc[-1])
                            close_lag2 = float(close_series.iloc[-2])
                            rolling_30 = float(close_series.iloc[-30:].mean())
                            rolling_200 = float(close_series.iloc[-200:].mean())
                            
                            features = ['Close_Lag1', 'Close_Lag2', 'Rolling_Mean_30', 'Rolling_Mean_200']
                            input_data = pd.DataFrame([{'Close_Lag1': close_lag1, 'Close_Lag2': close_lag2, 'Rolling_Mean_30': rolling_30, 'Rolling_Mean_200': rolling_200}])[features]
                            
                            if model:
                                pred = model.predict(input_data)[0]
                                pred_data = {
                                    'open': round(pred[0], 2), 'high': round(pred[1], 2),
                                    'low': round(pred[2], 2), 'close': round(pred[3], 2)
                                }
                                
                                # Package predictions alongside actuals for our front-end comparison table
                                comparison_results = {
                                    'date': selected_date,
                                    'metrics': [
                                        {'name': 'Open', 'pred': pred_data['open'], 'actual': actual_data['open'], 'err': round(abs(pred_data['open'] - actual_data['open']), 2)},
                                        {'name': 'High', 'pred': pred_data['high'], 'actual': actual_data['high'], 'err': round(abs(pred_data['high'] - actual_data['high']), 2)},
                                        {'name': 'Low', 'pred': pred_data['low'], 'actual': actual_data['low'], 'err': round(abs(pred_data['low'] - actual_data['low']), 2)},
                                        {'name': 'Close', 'pred': pred_data['close'], 'actual': actual_data['close'], 'err': round(abs(pred_data['close'] - actual_data['close']), 2)}
                                    ]
                                }

        except Exception as e:
            error_message = f"An error occurred: {str(e)}"

    return render_template('index.html', prediction=prediction_results, comparison=comparison_results, error=error_message, ticker=ticker_symbol, mode=mode, selected_date=selected_date)

if __name__ == '__main__':
    app.run(debug=True)