import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

data = "/home/user/Plocha/plot/data.csv"

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

if __name__ == "__main__":
    df = load_data(data)

    df['Čas'] = pd.to_datetime(df['Čas'].str.replace('\u202f', '', regex=True), format='%d.%m.%y', errors='coerce')
    df['Stoupání (m)'] = (
        df['Stoupání']
        .str.replace(r'\s+', '', regex=True)
        .str.replace('m', '', regex=False)
        .replace('-', np.nan)
        .astype(float)
    )
    df = df.dropna(subset=['Čas', 'Stoupání (m)'])


    df_daily = df.groupby(df['Čas'].dt.date)['Stoupání (m)'].sum().reset_index()
    df_daily['Čas'] = pd.to_datetime(df_daily['Čas'])

    df_daily = df_daily.sort_values('Čas')
    df_daily['den_v_roce'] = df_daily['Čas'].dt.dayofyear
    df_daily['den_v_týdnu'] = df_daily['Čas'].dt.weekday
    df_daily['měsíc'] = df_daily['Čas'].dt.month
    df_daily['rolling_7'] = df_daily['Stoupání (m)'].rolling(7).mean().fillna(0)
    df_daily['rolling_30'] = df_daily['Stoupání (m)'].rolling(30).mean().fillna(0)

    today = datetime(2025, 7, 15)
    train_df = df_daily[df_daily['Čas'] < today]
    features = ['den_v_roce', 'den_v_týdnu', 'měsíc', 'rolling_7', 'rolling_30']
    X = train_df[features]
    y = train_df['Stoupání (m)']


    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)


    end_of_year = datetime(2025, 12, 31)
    future_dates = pd.date_range(today + timedelta(days=1), end_of_year)
    future_df = pd.DataFrame({'Čas': future_dates})
    future_df['den_v_roce'] = future_df['Čas'].dt.dayofyear
    future_df['den_v_týdnu'] = future_df['Čas'].dt.weekday
    future_df['měsíc'] = future_df['Čas'].dt.month


    future_df['rolling_7'] = df_daily['rolling_7'].iloc[-1]
    future_df['rolling_30'] = df_daily['rolling_30'].iloc[-1]


    X_future = future_df[features]
    future_df['predikce (m)'] = model.predict(X_future)

    actual_2025 = df_daily[df_daily['Čas'].dt.year == 2025]
    soucet_2025 = actual_2025[actual_2025['Čas'] <= today]['Stoupání (m)'].sum()
    days_elapsed = (today - datetime(2025, 1, 1)).days + 1
    days_remaining = (end_of_year - today).days
    avg_daily = soucet_2025 / days_elapsed

    scenarios = {
        'Skutečnost (do 15.7.)': soucet_2025,
        'Lineární tempo': soucet_2025 + avg_daily * days_remaining,
        'ML predikce': soucet_2025 + future_df['predikce (m)'].sum(),
        '+20 % intenzita': soucet_2025 + (avg_daily * 1.2 * days_remaining),
        '–20 % intenzita': soucet_2025 + (avg_daily * 0.8 * days_remaining),
    }


    plt.figure(figsize=(10, 6))
    bars = plt.bar(scenarios.keys(), scenarios.values())

    plt.title("Projekce výškových metrů za rok 2025 (včetně ML predikce)")
    plt.ylabel("Výškové metry (m)")
    plt.xticks(rotation=30, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 500, f"{int(height):,}", ha='center')

    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
