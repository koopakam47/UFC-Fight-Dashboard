import os
import pandas as pd

def load_fighter_data():
    cleaned_path = 'data/cleaned_fighter_stats.csv'
    raw_path = 'data/data.csv'

    if os.path.exists(cleaned_path):
        print("✅ Loading cleaned dataset...")
        return pd.read_csv(cleaned_path)

    print("⚙️ Cleaned dataset not found. Loading and cleaning raw data...")
    df_raw = pd.read_csv(raw_path)

    # Fix column name if needed
    df_raw.columns = [col.replace('≤', '') for col in df_raw.columns]

    # Split into red and blue fighters
    red = df_raw.filter(like='R_').copy()
    red.columns = red.columns.str.replace('R_', '', regex=False)
    red['corner'] = 'red'

    blue = df_raw.filter(like='B_').copy()
    blue.columns = blue.columns.str.replace('B_', '', regex=False)
    blue['corner'] = 'blue'

    shared_cols = ['Winner', 'title_bout', 'weight_class', 'Referee', 'date', 'location']
    for col in shared_cols:
        red[col] = df_raw[col]
        blue[col] = df_raw[col]

    df_long = pd.concat([red, blue], ignore_index=True)

    df_long['won'] = (
        ((df_long['corner'] == 'red') & (df_long['Winner'] == 'red')) |
        ((df_long['corner'] == 'blue') & (df_long['Winner'] == 'blue'))
    ).astype(int)

    df_long.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned dataset saved to {cleaned_path}")
    return df_long
