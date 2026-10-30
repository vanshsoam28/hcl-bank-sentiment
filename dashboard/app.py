import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Review Sentiment Analyzer",
    page_icon="🏦",
    layout="wide"
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "cleaned_reviews.csv"
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


df = load_data()
model, vectorizer = load_model()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🏦 Bank Customer Review Sentiment Analyzer")

st.markdown(
    """
    Analyze customer reviews using **Natural Language Processing (NLP)**
    and a **Machine Learning sentiment classification model**.
    """
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Sentiment Prediction",
        "Review Explorer",
        "Model Information"
    ]
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("📊 Dashboard Overview")

    total_reviews = len(df)

    positive_count = int(
        (df["sentiment"] == "Positive").sum()
    )

    neutral_count = int(
        (df["sentiment"] == "Neutral").sum()
    )

    negative_count = int(
        (df["sentiment"] == "Negative").sum()
    )

    # KPI CARDS

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Reviews",
            total_reviews
        )

    with col2:
        st.metric(
            "😊 Positive",
            positive_count
        )

    with col3:
        st.metric(
            "😐 Neutral",
            neutral_count
        )

    with col4:
        st.metric(
            "😞 Negative",
            negative_count
        )

    st.divider()

    # SENTIMENT CHART

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            df["sentiment"]
            .value_counts()
            .reset_index()
        )

        sentiment_counts.columns = [
            "sentiment",
            "count"
        ]

        fig = px.pie(
            sentiment_counts,
            names="sentiment",
            values="count",
            hole=0.4,
            title="Customer Sentiment"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # RATING CHART

    with col2:

        st.subheader("Rating Distribution")

        rating_counts = (
            df["rating"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_counts.columns = [
            "rating",
            "count"
        ]

        fig = px.bar(
            rating_counts,
            x="rating",
            y="count",
            title="Customer Ratings",
            labels={
                "rating": "Rating",
                "count": "Number of Reviews"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # SENTIMENT PERCENTAGE

    st.subheader("Sentiment Percentages")

    sentiment_percentage = (
        df["sentiment"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    percentage_df = sentiment_percentage.reset_index()

    percentage_df.columns = [
        "Sentiment",
        "Percentage"
    ]

    fig = px.bar(
        percentage_df,
        x="Sentiment",
        y="Percentage",
        text="Percentage",
        title="Percentage of Reviews by Sentiment"
    )

    fig.update_traces(
        texttemplate="%{text}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------------------------
# SENTIMENT PREDICTION
# --------------------------------------------------

elif page == "Sentiment Prediction":

    st.header("🔮 Predict Customer Sentiment")

    st.write(
        "Enter a customer review below and the machine learning "
        "model will predict its sentiment."
    )

    review = st.text_area(
        "Enter customer review:",
        height=180,
        placeholder=(
            "Example: The mobile banking app is very easy to use "
            "and transactions are fast."
        )
    )

    if st.button(
        "🔍 Analyze Sentiment",
        type="primary"
    ):

        if not review.strip():

            st.warning(
                "Please enter a customer review."
            )

        else:

            features = vectorizer.transform(
                [review]
            )

            prediction = model.predict(
                features
            )[0]

            probabilities = model.predict_proba(
                features
            )[0]

            classes = model.classes_

            probability_df = pd.DataFrame({
                "Sentiment": classes,
                "Probability": probabilities
            })

            confidence = max(probabilities) * 100

            st.divider()

            # RESULT

            if prediction == "Positive":

                st.success(
                    f"😊 Predicted Sentiment: {prediction}"
                )

            elif prediction == "Negative":

                st.error(
                    f"😞 Predicted Sentiment: {prediction}"
                )

            else:

                st.warning(
                    f"😐 Predicted Sentiment: {prediction}"
                )

            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            # PROBABILITY CHART

            st.subheader(
                "Prediction Probabilities"
            )

            fig = px.bar(
                probability_df,
                x="Sentiment",
                y="Probability",
                text=probability_df[
                    "Probability"
                ].map(
                    lambda x: f"{x:.2%}"
                ),
                title="Sentiment Probability"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_yaxes(
                tickformat=".0%"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# --------------------------------------------------
# REVIEW EXPLORER
# --------------------------------------------------

elif page == "Review Explorer":

    st.header("🔎 Review Explorer")

    st.write(
        "Explore customer reviews from the dataset."
    )

    sentiment_filter = st.selectbox(
        "Filter by Sentiment",
        [
            "All",
            "Positive",
            "Neutral",
            "Negative"
        ]
    )

    rating_filter = st.selectbox(
        "Filter by Rating",
        ["All"] +
        sorted(
            df["rating"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    filtered_df = df.copy()

    if sentiment_filter != "All":

        filtered_df = filtered_df[
            filtered_df["sentiment"]
            == sentiment_filter
        ]

    if rating_filter != "All":

        filtered_df = filtered_df[
            filtered_df["rating"]
            == rating_filter
        ]

    st.write(
        f"Showing **{len(filtered_df)}** reviews"
    )

    display_columns = [
        column
        for column in [
            "review",
            "rating",
            "sentiment",
            "bank",
            "date"
        ]
        if column in filtered_df.columns
    ]

    st.dataframe(
        filtered_df[display_columns],
        use_container_width=True,
        hide_index=True
    )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

elif page == "Model Information":

    st.header("🤖 Model Information")

    st.subheader(
        "Machine Learning Pipeline"
    )

    st.markdown(
        """
        **1. Data Collection**  
        Bank customer reviews are loaded from the dataset.

        **2. Data Cleaning**  
        Reviews are converted to lowercase and unnecessary
        characters and URLs are removed.

        **3. Sentiment Labelling**  
        Reviews are assigned Positive, Neutral or Negative
        sentiment labels.

        **4. TF-IDF Vectorization**  
        Text is converted into numerical features using
        Term Frequency-Inverse Document Frequency.

        **5. Logistic Regression**  
        A Logistic Regression classifier predicts the sentiment.

        **6. FastAPI / Streamlit**  
        The trained model is exposed through an API and
        interactive dashboard.
        """
    )

    st.divider()

    st.subheader("Dataset Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:

        st.metric(
            "Total Reviews",
            len(df)
        )

        st.metric(
            "Dataset Columns",
            len(df.columns)
        )

    with info_col2:

        st.metric(
            "Positive Reviews",
            int(
                (df["sentiment"] == "Positive").sum()
            )
        )

        st.metric(
            "Negative Reviews",
            int(
                (df["sentiment"] == "Negative").sum()
            )
        )

    st.divider()

    st.subheader(
        "Model Evaluation"
    )

    st.markdown(
        """
        The Logistic Regression model achieved approximately:

        **Accuracy: 85%**

        The dataset contains significantly more Positive reviews
        than Neutral and Negative reviews, so accuracy alone should
        not be used to judge model performance. Per-class precision,
        recall and F1-score should also be considered.
        """
    )

    st.info(
        "Model files: sentiment_model.pkl + tfidf_vectorizer.pkl"
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Bank Customer Review Sentiment Analyzer | "
    "NLP + TF-IDF + Logistic Regression + FastAPI + Streamlit"
)
