import pandas as pd


# =====================================================
# 1. LOAD DATASET
# =====================================================
def load_dataset(file_path):
    """
    Load raw dataset from CSV file.
    """
    df = pd.read_csv(file_path)

    print(f"Loaded dataset with {len(df)} rows.")

    return df


# =====================================================
# 2. REMOVE DUPLICATE REVIEWS
# =====================================================
def remove_duplicates(df, id_column="id"):
    """
    Remove duplicate reviews using review ID.
    """
    before_count = len(df)

    df_cleaned = df.drop_duplicates(subset=[id_column], keep="first")

    after_count = len(df_cleaned)
    removed = before_count - after_count

    print("\n[DUPLICATE REMOVAL]")
    print(f"Duplicates removed: {removed}")

    return df_cleaned


# =====================================================
# 3. HANDLE MISSING VALUES
# =====================================================
def handle_missing_values(df):
    """
    Drop rows missing review text or rating.
    """
    before_count = len(df)

    df_cleaned = df.dropna(subset=["review_text", "rating"])

    after_count = len(df_cleaned)
    removed = before_count - after_count

    print("\n[MISSING VALUE HANDLING]")
    print(f"Rows removed due to missing values: {removed}")

    return df_cleaned


# =====================================================
# 4. NORMALIZE DATE FORMAT
# =====================================================
def normalize_dates(df, date_column="review_date"):
    """
    Convert dates to YYYY-MM-DD format.
    """
    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    print("\n[DATE NORMALIZATION]")
    print("Dates converted to YYYY-MM-DD format.")

    return df


# =====================================================
# 5. STANDARDIZE COLUMN NAMES
# =====================================================
def standardize_columns(df):
    """
    Rename columns to required output schema.
    """

    df = df.rename(columns={
        "review_text": "review",
        "review_date": "date",
        "bank_app_name": "bank"
    })

    required_columns = [
        "review",
        "rating",
        "date",
        "bank",
        "source"
    ]

    df = df[required_columns]

    print("\n[COLUMN STANDARDIZATION]")
    print("Columns renamed and reordered.")

    return df


# =====================================================
# 6. SAVE CLEANED DATASET
# =====================================================
def save_dataset(df, output_file):
    """
    Save cleaned dataset to CSV.
    """
    df.to_csv(output_file, index=False)

    print("\n[DATASET SAVED]")
    print(f"Saved cleaned dataset → {output_file}")
