import pandas as pd

def process_day_features(data, model):
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
    df['weather_score'] = df['weathersit']

    # ----- TIME -----
    df['is_month_start'] = (df['mnth'] == 1).astype(int)
    df['is_month_end'] = (df['mnth'] == 12).astype(int)

    # ----- HOLIDAY -----
    df['holiday_weekend'] = df['holiday'] & df['is_weekend']
    df['non_working_day'] = ((df['holiday'] == 1) | (df['workingday'] == 0)).astype(int)

    # ----- TEMP -----
    df['temp_diff'] = 0   # no previous day → dummy

    # ----- LAGS (dummy) -----
    df['cnt_lag_1'] = 0
    df['cnt_7day_avg'] = 0

    # ----- ONE-HOT: season -----
    for s in [2, 3, 4]:
        df[f'season_{s}'] = int(df['season'].iloc[0] == s)

    # ----- ONE-HOT: month -----
    for m in range(2, 13):
        df[f'mnth_{m}'] = int(df['mnth'].iloc[0] == m)

    # ----- ONE-HOT: weather -----
    for w in [2, 3]:
        df[f'weathersit_{w}'] = int(df['weathersit'].iloc[0] == w)

    # Drop raw columns
    df.drop(['season','mnth','weekday','weathersit','temp'], axis=1, inplace=True)

    # ----- FORCE EXACT FEATURE ORDER -----
    FEATURES = model.feature_names_in_
    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[FEATURES]
