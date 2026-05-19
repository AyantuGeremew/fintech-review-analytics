import matplotlib.pyplot as plt


# ======================================================
# VISUALIZE SENTIMENT BY BANK
# ======================================================
def visualize_sentiment_by_bank(aggregation):
    """
    Create a bar chart for mean sentiment score by bank.
    """

    # Create figure
    plt.figure(figsize=(8, 6))

    # Bar chart
    plt.bar(
        aggregation["bank"],
        aggregation["mean_sentiment_score"]
    )

    # Labels and title
    plt.xlabel("Bank")
    plt.ylabel("Mean Sentiment Score")
    plt.title("Average Sentiment Score by Bank")

    # Rotate labels if needed
    plt.xticks(rotation=15)

    # Add value labels on bars
    for index, value in enumerate(
        aggregation["mean_sentiment_score"]
    ):
        plt.text(
            index,
            value,
            round(value, 2),
            ha='center',
            va='bottom'
        )

    # Improve layout
    plt.tight_layout()

    # Show plot
    plt.show()

# ======================================================
# VISUALIZE SENTIMENT BY STAR RATING
# ======================================================
def visualize_sentiment_by_rating(aggregation):
    """
    Create a line chart for mean sentiment score by star rating.
    """

    # Create figure
    plt.figure(figsize=(10, 6))

    # Line plot
    plt.plot(
        aggregation["rating"],
        aggregation["mean_sentiment_score"],
        marker='o'
    )

    # Labels and title
    plt.xlabel("Star Rating")
    plt.ylabel("Mean Sentiment Score")
    plt.title("Average Sentiment Score by Star Rating")

    # Set x-axis ticks
    plt.xticks([1, 2, 3, 4, 5])

    # Add value labels
    for x, y in zip(
        aggregation["rating"],
        aggregation["mean_sentiment_score"]
    ):
        plt.text(
            x,
            y,
            round(y, 2),
            ha='center',
            va='bottom'
        )

    # Improve layout
    plt.tight_layout()

    # Show chart
    plt.show()