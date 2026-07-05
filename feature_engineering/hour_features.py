import pandas as pd
import numpy as np

def process_hour_features(data, model):
    df = pd.DataFrame([data])

    # ----- BASIC -----
    df['year'] = 1
    df['day'] = 1
    df['day_of_week'] = df['weekday']
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # ----- WEATHER -----
    df['good_conditions'] = (df['weathersit'] <= 2).astype(int)
    df['bad_weather'] = (df['weathersit'] >= 3).astype(int)
    df['high_humidity'] = (df['hum'] > 0.75).astype(int)

    # ----- TIME FLAGS -----
    df['is_night'] = ((df['hr'] <= 5) | (df['hr'] >= 22)).astype(int)
    df['is_peak_hour'] = df['hr'].isin([8, 9, 17, 18]).astype(int)
    df['is_morning_peak'] = df['hr'].isin([8, 9]).astype(int)
    df['is_evening_peak'] = df['hr'].isin([17, 18]).astype(int)

    # ----- CYCLIC -----
    df['hr_sin'] = np.sin(2 * np.pi * df['hr'] / 24)
    df['hr_cos'] = np.cos(2 * np.pi * df['hr'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['mnth'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['mnth'] / 12)

    # ----- INTERACTIONS -----
    df['temp_hum'] = df['temp'] * df['hum']
    df['temp_squared'] = df['temp'] ** 2
    df['temp_humidity'] = df['temp'] * df['hum']
    df['temp_windspeed'] = df['temp'] * df['windspeed']
    df['workingday_hr'] = df['workingday'] * df['hr']
    df['holiday_hr'] = df['holiday'] * df['hr']

    # ----- LAGS (dummy) -----
    df['cnt_lag_1'] = 0
    df['cnt_lag_24'] = 0
    df['cnt_lag_168'] = 0
    df['cnt_roll_3'] = 0
    df['cnt_roll_24'] = 0
    df['cnt_roll_168'] = 0
    df['cnt_diff_1'] = 0

    # Drop unused
    df.drop(['weekday'], axis=1, inplace=True)

    # ----- FORCE EXACT FEATURE ORDER -----
    FEATURES = model.feature_names_in_
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURES]
