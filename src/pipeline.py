import pandas as pd
import spacy
from transformers import pipeline


# =========================================================
# 1. LOAD SPACY MODEL
# =========================================================
def load_spacy_model():
    """
    Load spaCy English NLP model.
    """
    nlp = spacy.load("en_core_web_sm")
    return nlp


# =========================================================
# 2. LOAD SENTIMENT MODEL
# =========================================================
def load_sentiment_model():
    """
    Load DistilBERT sentiment analysis pipeline.
    """
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    return classifier


# =========================================================
# 3. LOAD DATASET
# =========================================================
def load_dataset(file_path):
    """
    Load review dataset.
    """
    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} reviews.")

    return df


# =========================================================
# 4. TEXT PREPROCESSING
# =========================================================
def preprocess_text(text, nlp, lemmatize=True):
    """
    Perform:
    - tokenization
    - stop-word removal
    - punctuation removal
    - optional lemmatization
    """

    doc = nlp(str(text).lower())

    processed_tokens = []

    for token in doc:

        # Skip stopwords, punctuation, spaces
        if token.is_stop or token.is_punct or token.is_space:
            continue

        if lemmatize:
            processed_tokens.append(token.lemma_)
        else:
            processed_tokens.append(token.text)

    return " ".join(processed_tokens)


# =========================================================
# 5. APPLY PREPROCESSING TO DATAFRAME
# =========================================================
def preprocess_reviews(df, nlp):
    """
    Clean all reviews.
    """

    df["processed_review"] = df["review"].apply(
        lambda x: preprocess_text(x, nlp)
    )

    print("\n[TEXT PREPROCESSING COMPLETED]")

    return df


# =========================================================
# 6. SENTIMENT ANALYSIS
# =========================================================
def classify_sentiment(df, classifier):
    """
    Classify review sentiment using DistilBERT.
    """

    labels = []
    scores = []

    for review in df["processed_review"]:

        try:
            result = classifier(review[:512])[0]

            label = result["label"]
            score = round(result["score"], 4)

            # Convert labels
            if score < 0.60:
                final_label = "neutral"
            elif label == "POSITIVE":
                final_label = "positive"
            else:
                final_label = "negative"

            labels.append(final_label)
            scores.append(score)

        except Exception:
            labels.append("neutral")
            scores.append(0.0)

    df["sentiment_label"] = labels
    df["sentiment_score"] = scores

    print("\n[SENTIMENT ANALYSIS COMPLETED]")

    return df


# =========================================================
# 7. IDENTIFY THEMES
# =========================================================
def identify_theme(review):
    """
    Rule-based theme classification.
    """

    review = review.lower()

    theme_keywords = {
        "Account Access Issues": [
            "login", "password", "account", "signin"
        ],

        "Transaction Performance": [
            "transfer", "payment", "transaction",
            "slow", "delay", "failed"
        ],

        "UI & Design": [
            "ui", "design", "interface",
            "layout", "navigation"
        ],

        "Customer Support": [
            "support", "service", "help",
            "customer", "response"
        ],

        "Feature Requests": [
            "feature", "update", "option",
            "request", "add"
        ]
    }

    for theme, keywords in theme_keywords.items():

        for keyword in keywords:

            if keyword in review:
                return theme

    return "Other"


# =========================================================
# 8. APPLY THEME IDENTIFICATION
# =========================================================
def apply_theme_identification(df):
    """
    Assign themes to reviews.
    """

    df["identified_theme"] = df["processed_review"].apply(
        identify_theme
    )

    print("\n[THEME IDENTIFICATION COMPLETED]")

    return df


# =========================================================
# 9. FINAL OUTPUT SELECTION
# =========================================================
def prepare_final_output(df):
    """
    Keep required output columns only.
    """

    output_df = df[[
        "id",
        "review",
        "sentiment_label",
        "sentiment_score",
        "identified_theme"
    ]].copy()

    output_df.columns = [
        "review_id",
        "review_text",
        "sentiment_label",
        "sentiment_score",
        "identified_theme"
    ]

    return output_df


# =========================================================
# 10. SAVE RESULTS
# =========================================================
def save_results(df, output_file):
    """
    Save final dataset.
    """

    df.to_csv(output_file, index=False)

    print(f"\nSaved results → {output_file}")
