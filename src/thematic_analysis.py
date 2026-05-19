import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF


# =========================================================
# 1. LOAD DATASET
# =========================================================
def load_dataset(file_path):
    """
    Load review dataset.
    """

    df = pd.read_csv(file_path)

    print(f"Loaded {len(df)} reviews.")

    return df


# =========================================================
# 2. CLEAN TEXT
# =========================================================
def clean_text(text):
    """
    Basic text preprocessing.
    """

    text = str(text).lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# 3. PREPROCESS REVIEWS
# =========================================================
def preprocess_reviews(df):
    """
    Clean review text column.
    """

    df["clean_review"] = df["review"].apply(clean_text)

    print("\nText preprocessing completed.")

    return df


# =========================================================
# 4. EXTRACT TF-IDF KEYWORDS & N-GRAMS
# =========================================================
def extract_tfidf_keywords(df, top_n=20):
    """
    Extract important keywords and n-grams using TF-IDF.
    """

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=1000
    )

    tfidf_matrix = vectorizer.fit_transform(
        df["clean_review"]
    )

    feature_names = vectorizer.get_feature_names_out()

    scores = tfidf_matrix.sum(axis=0).A1

    keyword_scores = list(
        zip(feature_names, scores)
    )

    sorted_keywords = sorted(
        keyword_scores,
        key=lambda x: x[1],
        reverse=True
    )

    top_keywords = sorted_keywords[:top_n]

    print("\n[TOP TF-IDF KEYWORDS & N-GRAMS]")

    for keyword, score in top_keywords:
        print(f"{keyword} -> {score:.4f}")

    return vectorizer, tfidf_matrix, top_keywords


# =========================================================
# 5. DISCOVER THEMES USING NMF
# =========================================================
def discover_topics(
        tfidf_matrix,
        vectorizer,
        n_topics=5,
        n_words=10
):
    """
    Discover latent themes using NMF topic modeling.
    """

    nmf_model = NMF(
        n_components=n_topics,
        random_state=42
    )

    nmf_model.fit(tfidf_matrix)

    feature_names = vectorizer.get_feature_names_out()

    discovered_topics = {}

    print("\n[DISCOVERED THEMES USING NMF]")

    for topic_idx, topic in enumerate(
            nmf_model.components_
    ):

        top_features = [
            feature_names[i]
            for i in topic.argsort()[:-n_words - 1:-1]
        ]

        discovered_topics[
            f"Theme_{topic_idx + 1}"
        ] = top_features

        print(f"\nTheme {topic_idx + 1}:")
        print(", ".join(top_features))

    return discovered_topics


# =========================================================
# 6. GROUP KEYWORDS INTO BUSINESS THEMES
# =========================================================
def group_keywords_into_themes(topics):
    """
    Group discovered keywords into business themes.
    """

    grouped_themes = {
        "Account Access Issues": [],
        "Transaction Performance": [],
        "UI & Design": [],
        "Customer Support": [],
        "Feature Requests": []
    }

    theme_mapping = {

        # Account Access
        "login": "Account Access Issues",
        "password": "Account Access Issues",
        "account": "Account Access Issues",
        "signin": "Account Access Issues",

        # Transactions
        "transfer": "Transaction Performance",
        "payment": "Transaction Performance",
        "transaction": "Transaction Performance",
        "slow": "Transaction Performance",
        "failed": "Transaction Performance",

        # UI/UX
        "ui": "UI & Design",
        "design": "UI & Design",
        "interface": "UI & Design",
        "navigation": "UI & Design",
        "easy": "UI & Design",

        # Customer Support
        "support": "Customer Support",
        "service": "Customer Support",
        "help": "Customer Support",
        "customer": "Customer Support",

        # Features
        "feature": "Feature Requests",
        "update": "Feature Requests",
        "option": "Feature Requests",
        "request": "Feature Requests"
    }

    for topic_name, keywords in topics.items():

        for keyword in keywords:

            for key, theme in theme_mapping.items():

                if key in keyword:
                    grouped_themes[theme].append(keyword)

    print("\n[GROUPED BUSINESS THEMES]")

    for theme, keywords in grouped_themes.items():

        unique_keywords = list(set(keywords))

        print(f"\n{theme}:")
        print(unique_keywords)

    return grouped_themes


# =========================================================
# 7. GENERATE THEMES PER BANK
# =========================================================
def generate_bank_themes(df, grouped_themes):
    """
    Generate theme counts per bank.
    """

    bank_theme_summary = []

    for bank in df["bank"].unique():

        bank_reviews = " ".join(
            df[df["bank"] == bank]["clean_review"]
        )

        for theme, keywords in grouped_themes.items():

            theme_count = sum(
                bank_reviews.count(keyword)
                for keyword in keywords
            )

            bank_theme_summary.append({
                "bank": bank,
                "theme": theme,
                "keyword_frequency": theme_count
            })

    summary_df = pd.DataFrame(
        bank_theme_summary
    )

    print("\n[THEME SUMMARY BY BANK]")
    print(summary_df)

    return summary_df


# =========================================================
# 8. SAVE OUTPUTS
# =========================================================
def save_outputs(summary_df):
    """
    Save theme analysis outputs.
    """

    summary_df.to_csv(
        "data/bank_theme_analysis.csv",
        index=False
    )

    print("\nSaved:")
    print("- bank_theme_analysis.csv")


# =========================================================
# 9. MARKDOWN DOCUMENTATION
# =========================================================
def print_markdown_grouping_logic():
    """
    Markdown documentation for grouping logic.
    """

    markdown = """
# Theme Grouping Logic

Thematic analysis was performed using TF-IDF keyword extraction
and NMF topic modeling.

## Process Overview

1. Reviews were cleaned and preprocessed.
2. TF-IDF extracted important keywords and n-grams.
3. NMF identified latent discussion topics.
4. Keywords were manually mapped into business-relevant themes.

---

## Business Themes

### 1. Account Access Issues
Related to:
- login problems
- authentication
- password failures
- account access

Example keywords:
- login error
- password reset
- cannot login

---

### 2. Transaction Performance
Related to:
- failed transfers
- slow transactions
- payment processing delays

Example keywords:
- slow transfer
- payment failed
- transaction pending

---

### 3. UI & Design
Related to:
- user interface
- app usability
- navigation
- design quality

Example keywords:
- good ui
- easy navigation
- bad design

---

### 4. Customer Support
Related to:
- customer care
- issue handling
- support responsiveness

Example keywords:
- poor support
- customer service
- help center

---

### 5. Feature Requests
Related to:
- requested features
- user suggestions
- desired improvements

Example keywords:
- add fingerprint login
- dark mode
- more features
"""

    print(markdown)
