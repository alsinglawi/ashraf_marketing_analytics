import pandas as pd
from pathlib import Path

def load_data(file_name="online.csv"):
    raw_path = Path("data/raw") / file_name
    processed_path = Path("data/processed")
    processed_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    df = df.drop_duplicates().dropna(how="all")

    df.to_csv(processed_path / "cleaned_online.csv", index=False)
    print(f"💾 Cleaned file saved to: {processed_path}/cleaned_online.csv")

    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())
