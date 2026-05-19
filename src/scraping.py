import pandas as pd
import numpy as np
import os
from google_play_scraper import reviews, Sort


# ======================================================
# CONFIGURATION
# ======================================================
BANK_APPS = {
    "CBE": "com.combanketh.mobilebanking",
    "BOA": "com.boa.boaMobileBanking",
    "DB": "com.dashen.dashensuperapp"
}

SOURCE = "Google Play"
TARGET_REVIEWS = 400


# ======================================================
#  FETCH REVIEWS WITH LIMIT CONTROL
# ======================================================
def fetch_reviews_with_limit(app_id, app_name, target=400):
    """
    Collect reviews for a single app.
    Ensures target coverage where possible (≥400).
    """

    collected = []
    continuation_token = None

    while len(collected) < target:
        result, continuation_token = reviews(
            app_id,
            lang="en",
            country="us",
            sort=Sort.NEWEST,
            count=200,
            continuation_token=continuation_token
        )

        if not result:
            break

        for r in result:
            collected.append({
                # (c) required fields
                "review_text": r.get("content"),
                "rating": r.get("score"),
                "review_date": r.get("at"),
                "bank_app_name": app_name,
                "source": SOURCE
            })

        # Stop if no more pages
        if continuation_token is None:
            break

    return collected[:target], len(collected) < target


# ======================================================
# COLLECT ALL BANKS
# ======================================================
def collect_all_banks(bank_apps, target=400):
    """
    Collect reviews for all banks.
    Returns dataset + limitation report.
    """

    dataset = []
    limitations = {}

    for bank_name, app_id in bank_apps.items():
        print(f"Collecting reviews for {bank_name}...")

        data, limited = fetch_reviews_with_limit(app_id, bank_name, target)

        dataset.extend(data)

        if len(data) < target:
            limitations[bank_name] = len(data)

    return dataset, limitations


# ======================================================
# CONVERT TO DATAFRAME
# ======================================================
def to_dataframe(data):
    return pd.DataFrame(data)


# ======================================================
# SAVE OUTPUT
# ======================================================
def save_data(df, filename="bank_reviews_google_play.csv"):
    # Ensure the data folder exists
    os.makedirs("data", exist_ok=True)

    # Create full file path
    filepath = os.path.join("data", filename)

    # Save CSV inside data folder
    df.to_csv(filepath, index=False)

    print(f"\nDataset saved → {filepath}")


# ======================================================
# LIMITATION REPORT (IMPORTANT FOR PART b)
# ======================================================
def report_limitations(limitations, target):
    print("\n================ LIMITATION REPORT ================")

    if not limitations:
        print("✔ Successfully collected 400+ reviews for all banks.")
    else:
        print("⚠ Some banks returned fewer than 400 reviews due to Google Play API limits:\n")
        for bank, count in limitations.items():
            print(f"- {bank}: {count}/{target} reviews collected")

        print("\nNote:")
        print("Google Play Store may restrict available reviews or pagination depth.")
        print("All available reviews were collected up to system limits.")

