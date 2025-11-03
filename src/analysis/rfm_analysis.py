# src/analysis/rfm_analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Helper: detect likely columns
# -------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------------
# Load function (auto-detect file)
# -------------------------
def load_transactions(path_candidates=None):
    """
    Try to load a transactions CSV file.
    path_candidates: list of Path strings to try (optional).
    Returns dataframe and path used.
    """
    if path_candidates is None:
        path_candidates = [
            "data/processed/cleaned_online.csv",
            "data/raw/online.csv",
            "data/raw/transactions.csv",
            "data/raw/sales.csv",
            "data/raw/tidy_transactions.csv",
        ]

    for p in path_candidates:
        pth = Path(p)
        if pth.exists():
            df = pd.read_csv(pth, low_memory=False)
            print(f"Loaded file: {pth}  |  shape: {df.shape}")
            return df, pth
    raise FileNotFoundError(f"No candidate file found. Tried: {path_candidates}")

# -------------------------
# Cleaning
# -------------------------
def clean_transactions(df, date_col=None, cust_col=None, invoice_col=None, qty_col=None, price_col=None):
    """
    Clean common issues:
      - detect columns automatically
      - parse dates
      - remove cancelled orders (Invoice code starting with 'C', negative qty, or explicit 'Cancelled' flag)
      - drop rows where customer id is NA
    Returns cleaned df and dict of used column names.
    """
    # detect columns
    date_col = date_col or find_col(df, ["InvoiceDate", "invoice_date", "date", "Date", "transaction_date", "OrderDate"])
    cust_col = cust_col or find_col(df, ["CustomerID", "customer_id", "cust_id", "client_id", "buyer_id"])
    invoice_col = invoice_col or find_col(df, ["InvoiceNo", "invoice_no", "Invoice", "order_id", "OrderID"])
    qty_col = qty_col or find_col(df, ["Quantity", "quantity", "qty"])
    price_col = price_col or find_col(df, ["Price", "UnitPrice", "unit_price", "amount", "unitprice", "price"])

    used = {"date": date_col, "customer": cust_col, "invoice": invoice_col, "quantity": qty_col, "price": price_col}
    print("Auto-detected columns:", used)

    if date_col is None:
        raise ValueError("No date column detected. Please provide date_col.")
    # parse dates
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    # drop rows without date
    df = df[~df[date_col].isna()]

    # remove cancelled orders:
    if invoice_col:
        # common pattern: invoice starting with 'C' denotes cancellation
        cancelled_mask = df[invoice_col].astype(str).str.startswith("C", na=False)
    else:
        cancelled_mask = pd.Series([False]*len(df), index=df.index)
    # negative quantities are often returns
    if qty_col:
        cancelled_mask = cancelled_mask | (pd.to_numeric(df[qty_col], errors="coerce") < 0)
    # if there's a column named 'Cancelled' or 'is_cancelled'
    cancel_flag = find_col(df, ["Cancelled","is_cancelled","cancelled"])
    if cancel_flag:
        cancelled_mask = cancelled_mask | (df[cancel_flag].astype(str).str.lower().isin(["true","1","yes","y"]))
    before = len(df)
    df = df[~cancelled_mask]
    print(f"Removed cancelled/return rows: {before - len(df)}")

    # drop rows where customer id is NA
    if cust_col is None:
        raise ValueError("No customer id column detected. Please provide customer id column.")
    before = len(df)
    df = df[~df[cust_col].isna()]
    df = df[df[cust_col].astype(str).str.strip()!=""]
    print(f"Removed rows with missing customer id: {before - len(df)}")

    # compute revenue if not present
    revenue_col = find_col(df, ["Revenue","revenue","Amount","amount","Total","total"])
    if revenue_col is None and price_col and qty_col:
        df["revenue"] = pd.to_numeric(df[price_col], errors="coerce").fillna(0) * pd.to_numeric(df[qty_col], errors="coerce").fillna(0)
    elif revenue_col:
        df["revenue"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0)
    else:
        # if nothing else, create revenue from price only
        if price_col:
            df["revenue"] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
        else:
            df["revenue"] = 0.0

    return df.reset_index(drop=True), used

# -------------------------
# Restrict to one full year (most recent full year)
# -------------------------
def restrict_to_last_full_year(df, date_col):
    """
    Keep only transactions in the most recent full calendar year present in data.
    For example, if data includes 2021-01-... through 2024-07-... the most recent full year is 2023.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    years = sorted(df[date_col].dt.year.dropna().unique())
    if not years:
        raise ValueError("No valid years found in date column.")
    # find the most recent year that is *complete* (has data through 31 Dec)
    # We'll find years present and test if max date for that year is >= Dec 31 of that year
    candidate_years = []
    for y in years:
        max_date = df.loc[df[date_col].dt.year == y, date_col].max()
        if pd.notna(max_date) and max_date >= pd.Timestamp(year=y, month=12, day=31):
            candidate_years.append(y)
    if candidate_years:
        selected_year = max(candidate_years)
    else:
        # fallback: use the most recent year (partial)
        selected_year = max(years)
        print(f"Warning: no complete calendar year detected. Using most recent year {selected_year} (may be partial).")
    mask = df[date_col].dt.year == selected_year
    print(f"Restricting dataset to year {selected_year}: rows before={len(df)}, after={mask.sum()}")
    return df.loc[mask].copy(), selected_year

# -------------------------
# Compute RFM
# -------------------------
def compute_rfm(df, date_col, cust_col, invoice_col=None):
    """
    Compute RFM table per customer.
    Recency: days since last purchase (relative to snapshot_date)
    Frequency: number of unique invoices/orders
    Monetary: sum of revenue
    """
    # snapshot date: one day after the last transaction in dataset (common practice)
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)
    print("Snapshot date for recency:", snapshot_date.date())

    # Ensure revenue numeric
    df["revenue"] = pd.to_numeric(df.get("revenue", 0), errors="coerce").fillna(0)

    # group by customer
    if invoice_col and invoice_col in df.columns:
        grouped = df.groupby(cust_col).agg(
            recency_days = (date_col, lambda x: (snapshot_date - x.max()).days),
            frequency = (invoice_col, lambda x: x.nunique()),
            monetary = ("revenue", "sum")
        ).reset_index()
    else:
        grouped = df.groupby(cust_col).agg(
            recency_days = (date_col, lambda x: (snapshot_date - x.max()).days),
            frequency = (date_col, "count"),
            monetary = ("revenue", "sum")
        ).reset_index()

    # ensure numeric types
    grouped["frequency"] = grouped["frequency"].astype(int)
    grouped["monetary"] = grouped["monetary"].astype(float)
    return grouped, snapshot_date

# -------------------------
# Score into quartiles (1-4)
# -------------------------
def rfm_score_quartiles(rfm):
    """
    Score R (recency) such that lower recency (more recent) => higher score.
    F and M: larger => higher score.
    Scores are 1..4 (quartiles)
    """
    r_labels = [4,3,2,1]  # newer => 4
    rfm['R_score'] = pd.qcut(rfm['recency_days'], 4, labels=r_labels, duplicates='drop').astype(int)
    f_labels = [1,2,3,4]
    rfm['F_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 4, labels=f_labels, duplicates='drop').astype(int)
    m_labels = [1,2,3,4]
    rfm['M_score'] = pd.qcut(rfm['monetary'].rank(method='first'), 4, labels=m_labels, duplicates='drop').astype(int)

    rfm['RFM_score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)
    rfm['RFM_sum'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    return rfm

# -------------------------
# Map basic segments (simple rules)
# -------------------------
def label_segments(rfm):
    """
    Basic mapping from RFM to segment label using simple rules.
    You can expand rules later for more nuanced segments.
    """
    def seg(row):
        if row['RFM_sum'] >= 10:
            return "Champions"
        if row['R_score'] == 4 and row['F_score'] >= 3:
            return "Loyal"
        if row['R_score'] >= 3 and row['M_score'] >= 3:
            return "Potential"
        if row['R_score'] <= 2 and row['F_score'] <= 2:
            return "At Risk"
        if row['R_score'] == 1 and row['F_score'] <= 2:
            return "Lost"
        return "Other"
    rfm['segment'] = rfm.apply(seg, axis=1)
    return rfm

# -------------------------
# Summary plots and export
# -------------------------
def export_and_plot(rfm, out_path=Path("data/processed/rfm_summary.csv")):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rfm.to_csv(out_path, index=False)
    print(f"Saved RFM table to: {out_path}")

    # simple visualizations
    plt.figure(figsize=(8,4))
    sns.countplot(y='segment', data=rfm, order=rfm['segment'].value_counts().index)
    plt.title("Customer segments (count)")
    plt.tight_layout()
    plt.savefig("reports/figures/rfm_segments_count.png")
    plt.close()
    print("Saved segment counts figure to reports/figures/rfm_segments_count.png")

    plt.figure(figsize=(6,4))
    sns.histplot(rfm['RFM_sum'], bins=9)
    plt.title("Distribution of RFM sum")
    plt.tight_layout()
    plt.savefig("reports/figures/rfm_sum_hist.png")
    plt.close()
    print("Saved RFM sum histogram to reports/figures/rfm_sum_hist.png")

# -------------------------
# Orchestration function
# -------------------------
def run_rfm(path_candidates=None):
    df, used_path = load_transactions(path_candidates)
    df, used_cols = clean_transactions(df)
    # restrict to last full year if possible
    df_year, selected_year = restrict_to_last_full_year(df, used_cols['date'])
    rfm, snapshot = compute_rfm(df_year, used_cols['date'], used_cols['customer'], invoice_col=used_cols['invoice'])
    rfm = rfm_score_quartiles(rfm)
    rfm = label_segments(rfm)
    export_and_plot(rfm)
    print("RFM pipeline complete. Snapshot date:", snapshot.date())
    return rfm

if __name__ == "__main__":
    run_rfm()
