import pandas as pd
from scipy.stats import chi2_contingency
from pathlib import Path

def load_data(file_name="online.csv"):
    """Load processed campaign data."""
    processed_path = Path("data/processed") / file_name
    df = pd.read_csv(processed_path)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def calculate_metrics(df):
    """Calculate key marketing metrics."""
    # Example columns; adjust to your CSV
    df['conversion_rate'] = df['converted'] / df['leads']
    df['click_through_rate'] = df['clicks'] / df['impressions']
    df['response_rate'] = df['responses'] / df['emails_sent']
    return df

def attribution_analysis(df):
    """Aggregate conversions by channel."""
    channel_performance = df.groupby('channel')['converted'].sum().sort_values(ascending=False)
    print("\n📊 Conversions by channel:")
    print(channel_performance)
    return channel_performance

def ab_test(df):
    """Perform A/B test on conversion rate between groups A and B."""
    if 'group' not in df.columns:
        print("⚠️ No 'group' column found for A/B test")
        return None
    table = pd.crosstab(df['group'], df['converted'])
    chi2, p, dof, ex = chi2_contingency(table)
    print("\n🔬 A/B Test Result:")
    print(f"Chi2: {chi2:.2f}, p-value: {p:.4f}")
    if p < 0.05:
        print("✅ Significant difference between groups!")
    else:
        print("❌ No significant difference.")
    return chi2, p

def save_results(df, file_name="online.csv"):
    df.to_csv(Path("data/processed") / file_name, index=False)
    print(f"\n💾 Metrics saved to data/processed/{file_name}")

if __name__ == "__main__":
    df = load_data("cleaned_online.csv")
    df = calculate_metrics(df)
    attribution_analysis(df)
    ab_test(df)
    save_results(df)

