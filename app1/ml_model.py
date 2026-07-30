import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
import os

# Ensure the correct path relative to Django root
DB_PATH = 'detections.db'

def get_predicted_violations():
    """
    Trains a regression model on historical detection data
    to predict the expected number of PPE violations for the upcoming hour.
    """
    if not os.path.exists(DB_PATH):
        return {"count": 0, "peak_time": "Insufficient Data"}
        
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT timestamp, class_name FROM detections", conn)
        conn.close()
    except Exception as e:
        print("Database error in ml_model:", e)
        return {"count": 0, "peak_time": "Error"}

    if df.empty:
        return {"count": 0, "peak_time": "Insufficient Data"}

    # Convert timestamp to datetime objects
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])

    if df.empty:
        return {"count": 0, "peak_time": "Insufficient Data"}

    # Extract features for regression learning
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Aggregate: Count how many violations occur during each hour block
    hourly_counts = df.groupby(['date', 'hour', 'day_of_week']).size().reset_index(name='violations')

    # Fallback to simple average if barely any data exists yet
    if len(hourly_counts) < 5:
        # Before we had peak_hour logic here, let's just use the df logic since df is not empty
        peak_hour_str = "Insufficient Data"
        total_by_hour = df.groupby('hour').size().reset_index(name='total')
        if not total_by_hour.empty:
            peak_hour = int(total_by_hour.loc[total_by_hour['total'].idxmax()]['hour'])
            ampm1 = "AM" if peak_hour < 12 else "PM"
            ampm2 = "AM" if (peak_hour+1)%24 < 12 else "PM"
            h1 = peak_hour if peak_hour <= 12 else peak_hour - 12
            if h1 == 0: h1 = 12
            str_h2 = (peak_hour+1) % 24
            h2 = str_h2 if str_h2 <= 12 else str_h2 - 12
            if h2 == 0: h2 = 12
            peak_hour_str = f"{h1}:00 {ampm1} - {h2}:00 {ampm2}"
            
        return {"count": int(hourly_counts['violations'].mean()), "peak_time": peak_hour_str}

    # X features (Hour of day, Day of week) -> y target (Number of violations)
    X = hourly_counts[['hour', 'day_of_week']]
    y = hourly_counts['violations']

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)

    # Prepare features for the NEXT hour prediction
    now = datetime.now()
    next_hour = now + timedelta(hours=1)
    
    X_pred = pd.DataFrame([{
        'hour': next_hour.hour,
        'day_of_week': next_hour.weekday()
    }])
    # Predict the expected volume
    predicted_count = model.predict(X_pred)[0]

    # Calculate Peak Violation Time across history
    peak_hour_str = "Insufficient Data"
    if not df.empty:
        total_by_hour = df.groupby('hour').size().reset_index(name='total')
        if not total_by_hour.empty:
            peak_hour = int(total_by_hour.loc[total_by_hour['total'].idxmax()]['hour'])
            ampm1 = "AM" if peak_hour < 12 else "PM"
            ampm2 = "AM" if (peak_hour+1)%24 < 12 else "PM"
            h1 = peak_hour if peak_hour <= 12 else peak_hour - 12
            if h1 == 0: h1 = 12
            str_h2 = (peak_hour+1) % 24
            h2 = str_h2 if str_h2 <= 12 else str_h2 - 12
            if h2 == 0: h2 = 12
            peak_hour_str = f"{h1}:00 {ampm1} - {h2}:00 {ampm2}"

    return {
        "count": int(max(0, predicted_count)),
        "peak_time": peak_hour_str
    }
