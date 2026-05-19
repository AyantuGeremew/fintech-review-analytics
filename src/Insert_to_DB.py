import pandas as pd
from sqlalchemy import create_engine

# ================= DATABASE =================
engine = create_engine(
    "mssql+pyodbc://@localhost\\SQLEXPRESS/bank_reviews"
    "?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

# ================= INPUT FILES =================
files = {
    "Commercial Bank of Ethiopia": "data/processed/cbe_clean.csv",
    "Bank of Abyssinia": "data/processed/boa_clean.csv",
    "Dashen Bank": "data/processed/dashen_clean.csv"
}

# ================= SAFE COLUMN STANDARDIZER =================
def standardize(df):
    # rename safely (won't crash if column missing)
    rename_map = {}

    if "review" in df.columns:
        rename_map["review"] = "review"

    if "at" in df.columns:
        rename_map["at"] = "date"

    df = df.rename(columns=rename_map)

    # ensure required columns exist
    if "review_text" not in df.columns:
        df["review_text"] = None

    if "rating" not in df.columns:
        df["rating"] = None

    if "review_date" not in df.columns:
        df["review_date"] = None

    return df

# ================= LOAD ALL DATA =================
dfs = []

for bank, path in files.items():
    df = pd.read_csv(path)

    df = standardize(df)

    df["bank_name"] = bank
    df["source"] = "Google Play"

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

print("After concat:", df.shape)

# ================= CLEAN TEXT =================
df["bank_name"] = df["bank_name"].astype(str).str.strip().str.lower()

# ================= LOAD BANK DIMENSION =================
banks = pd.read_sql("SELECT bank_id, bank_name FROM Banks", engine)

banks["bank_name"] = banks["bank_name"].str.strip().str.lower()

# ================= MERGE SAFELY =================
df = df.merge(banks, on="bank_name", how="inner")

print("After merge:", df.shape)

# ================= FINAL VALIDATION =================
required_cols = ["bank_id", "review_text", "rating", "review_date", "source"]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    raise Exception(f"Missing columns: {missing}")

df_final = df[required_cols].copy()

df_final = df_final.dropna(subset=["review_text", "rating"])

df_final["rating"] = pd.to_numeric(df_final["rating"], errors="coerce")

df_final = df_final.dropna(subset=["rating"])

df_final["rating"] = df_final["rating"].astype(int)

print("Final rows to insert:", len(df_final))

# ================= INSERT =================
df_final.to_sql("Reviews", engine, if_exists="append", index=False)

print("SUCCESS: Data inserted into Reviews table")