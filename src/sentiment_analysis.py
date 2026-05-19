import pandas as pd
import torch

from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =========================================================
# LOAD CLEANED DATASET
# =========================================================
def load_dataset(file_path):
    """
    Load cleaned review dataset.
    """
    df = pd.read_csv(file_path)

    print(f"Loaded dataset with {len(df)} reviews.")

    return df


# =========================================================
# LOAD TRANSFORMER MODEL
# =========================================================
def load_transformer_model():
    """
    Load DistilBERT sentiment analysis pipeline.
    
    Model:
    distilbert-base-uncased-finetuned-sst-2-english
    """
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return classifier


# =========================================================
# CLASSIFY SENTIMENT USING DISTILBERT
# =========================================================
def classify_sentiment_transformer(df, classifier):
    """
    Classify sentiment using DistilBERT.
    Produces:
    - sentiment label
    - confidence score
    """

    sentiments = []
    confidence_scores = []

    for review in df["review"]:
        try:
            result = classifier(str(review))[0]

            label = result["label"]
            score = result["score"]

            # Add neutral threshold manually
            if score < 0.60:
                final_label = "neutral"
            elif label == "POSITIVE":
                final_label = "positive"
            else:
                final_label = "negative"

            sentiments.append(final_label)
            confidence_scores.append(round(score, 4))

        except Exception:
            sentiments.append("neutral")
            confidence_scores.append(0.0)

    df["sentiment"] = sentiments
    df["confidence_score"] = confidence_scores

    print("\n[TRANSFORMER SENTIMENT]")
    print("Sentiment classification completed.")

    return df


# =========================================================
# 4. OPTIONAL: VADER SENTIMENT ANALYSIS
# =========================================================
def vader_sentiment_analysis(df):
    """
    Alternative sentiment analysis using VADER.
    Useful for comparison with transformer model.
    """

    analyzer = SentimentIntensityAnalyzer()

    vader_labels = []
    vader_scores = []

    for review in df["review"]:
        scores = analyzer.polarity_scores(str(review))

        compound = scores["compound"]

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        vader_labels.append(label)
        vader_scores.append(compound)

    df["vader_sentiment"] = vader_labels
    df["vader_score"] = vader_scores

    print("\n[VADER SENTIMENT]")
    print("VADER sentiment analysis completed.")

    return df


# =========================================================
# 5. AGGREGATE SENTIMENT BY BANK
# =========================================================
def aggregate_by_bank(df):
    """
    Aggregate sentiment confidence scores by bank.
    """

    aggregation = (
        df.groupby("bank")["confidence_score"]
        .mean()
        .reset_index()
        .rename(columns={
            "confidence_score": "mean_sentiment_score"
        })
    )

    print("\n[SENTIMENT BY BANK]")
    print(aggregation)

    return aggregation


# =========================================================
# 6. AGGREGATE SENTIMENT BY STAR RATING
# =========================================================
def aggregate_by_rating(df):
    """
    Aggregate sentiment scores by rating.
    """

    aggregation = (
        df.groupby("rating")["confidence_score"]
        .mean()
        .reset_index()
        .rename(columns={
            "confidence_score": "mean_sentiment_score"
        })
    )

    print("\n[SENTIMENT BY STAR RATING]")
    print(aggregation)

    return aggregation


# =========================================================
# 7. SAVE RESULTS
# =========================================================
def save_results(df, bank_summary, rating_summary):
    """
    Save analysis outputs.
    """

    df.to_csv("data/sentiment_analysis_reviews.csv", index=False)
    bank_summary.to_csv("data/bank_sentiment_summary.csv", index=False)
    rating_summary.to_csv("data/rating_sentiment_summary.csv", index=False)

    print("\n[FILES SAVED]")
    print("Saved:")
    print("- sentiment_analysis_reviews.csv")
    print("- bank_sentiment_summary.csv")
    print("- rating_sentiment_summary.csv")


# =========================================================
# 8. TOOL SELECTION RATIONALE
# =========================================================
def print_rationale():
    """
    Brief documentation for model selection.
    """

    rationale = """
    TOOL SELECTION RATIONALE
    ------------------------
    DistilBERT was selected because it is a transformer-based
    NLP model fine-tuned on SST-2 sentiment classification tasks.
    It generally provides higher contextual understanding and
    better accuracy than rule-based methods.

    VADER was additionally included for comparison because it is:
    - lightweight,
    - fast,
    - effective for short social-media-style text.

    Transformer models are preferred for final analysis due to
    superior semantic understanding and confidence scoring.
    """

    print(rationale)
