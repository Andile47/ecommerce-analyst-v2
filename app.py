from __future__ import annotations

import hashlib
import html
import math
import re
from pathlib import Path
from typing import Any

import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer


st.set_page_config(page_title="SentimentIQ", page_icon="📊", layout="wide")


@st.cache_resource
def load_nltk():
    nltk.download("vader_lexicon", quiet=True)


load_nltk()
sia = SentimentIntensityAnalyzer()
ANALYSIS_SENTIMENT_COLUMN = "__sentimentiq_sentiment"
ANALYSIS_SCORE_COLUMN = "__sentimentiq_score"
SOURCE_COLUMNS: list[str] = []
THEME_KEYWORDS = {
    "Delivery and fulfilment": ("delivery", "deliver", "delivered", "shipping", "shipped", "arrival", "arrived", "late", "tracking", "courier"),
    "Product quality and reliability": ("quality", "broken", "defect", "defective", "damaged", "faulty", "unreliable", "stopped working", "not working"),
    "Customer service and support": ("customer service", "support", "response", "responded", "refund", "helpful", "staff", "agent", "contacted"),
    "Price and value": ("price", "value", "expensive", "cheap", "cost", "worth", "money", "affordable"),
    "Design and ease of use": ("design", "easy to use", "setup", "set up", "comfortable", "instructions", "intuitive", "fit"),
}
COMPILED_THEME_RULES = {
    theme: re.compile(r"\b(?:" + "|".join(re.escape(keyword) for keyword in keywords) + r")\b", re.IGNORECASE)
    for theme, keywords in THEME_KEYWORDS.items()
}


def resolve_default_csv() -> Path | None:
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "sample_reviews.csv",
        base_dir / "data" / "sample_reviews.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def analyze_sentiment(text: str) -> str:
    score = sia.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def source_text(value: Any) -> str:
    return "" if pd.isna(value) else str(value)


def explain_sentiment(sentiment: str) -> str:
    explanations = {
        "Positive": "The wording has an overall positive tone. VADER classifies compound scores of +0.05 or higher as Positive.",
        "Neutral": "The wording has a balanced or low-intensity tone. VADER classifies compound scores above -0.05 and below +0.05 as Neutral.",
        "Negative": "The wording has an overall negative tone. VADER classifies compound scores of -0.05 or lower as Negative.",
    }
    return explanations[sentiment]


def save_review_analysis(text: str, *, populate_input: bool = False) -> None:
    cleaned_text = text.strip()
    if populate_input:
        st.session_state["review_input"] = text

    if not cleaned_text:
        st.session_state["review_analysis_error"] = "Enter a review before analysing it."
        st.session_state["review_analysis_result"] = None
        return

    try:
        score = float(sia.polarity_scores(cleaned_text)["compound"])
        sentiment = analyze_sentiment(cleaned_text)
        result = {
            "review": cleaned_text,
            "sentiment": sentiment,
            "score": score,
            "explanation": explain_sentiment(sentiment),
            "analyzed_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        st.session_state["review_analysis_result"] = result
        st.session_state["review_analysis_error"] = None
        st.session_state.setdefault("review_analysis_history", []).insert(0, result)
    except Exception:
        st.session_state["review_analysis_result"] = None
        st.session_state["review_analysis_error"] = "The review could not be analysed. Please check the text and try again."


def analyze_review_input() -> None:
    save_review_analysis(st.session_state.get("review_input", ""))


def clear_review_input() -> None:
    st.session_state["review_input"] = ""
    st.session_state["review_analysis_result"] = None
    st.session_state["review_analysis_error"] = None


def clear_review_result() -> None:
    st.session_state["review_analysis_result"] = None
    st.session_state["review_analysis_error"] = None


def get_sentiment_eps(value: str) -> str:
    return {
        "Positive": "✅",
        "Neutral": "◐",
        "Negative": "❌",
    }.get(value, "•")


def get_dashboard_css() -> str:
    return """
    <style>
        :root {
            --bg-deep: #020d1f;
            --bg-mid: #071a35;
            --panel: rgba(11, 28, 56, 0.88);
            --panel-strong: rgba(18, 38, 76, 0.96);
            --border: rgba(124, 148, 226, 0.26);
            --text: #e8f0ff;
            --muted: #9eb4d5;
            --blue: #3ea7ff;
            --cyan: #7bdcf7;
            --purple: #8d79ff;
            --green: #41e0a1;
            --red: #ff5f7a;
            --amber: #ffd166;
            --shadow: rgba(12, 19, 44, 0.7);
        }

        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: radial-gradient(circle at top left, rgba(83, 96, 255, 0.18), transparent 24%),
                        radial-gradient(circle at bottom right, rgba(0, 219, 255, 0.18), transparent 25%),
                        linear-gradient(135deg, #020d1f 0%, #071a35 35%, #06172f 100%);
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: rgba(6, 16, 31, 0.94);
            border-right: 1px solid var(--border);
            box-shadow: inset -1px 0 0 rgba(255,255,255,0.04);
        }

        .brand-wrap {
            display: flex;
            align-items: center;
            gap: 10px;
            margin: 8px 0 24px 0;
            padding: 0 0 12px 0;
        }

        .brand-mark {
            width: 28px;
            height: 28px;
            border-radius: 9px;
            background: linear-gradient(135deg, var(--cyan), var(--blue), var(--purple));
            box-shadow: 0 0 20px rgba(101, 176, 255, 0.8);
            position: relative;
        }

        .brand-mark::before,
        .brand-mark::after {
            content: "";
            position: absolute;
            width: 10px;
            height: 2px;
            background: rgba(255, 255, 255, 0.8);
            border-radius: 999px;
            top: 7px;
            left: 9px;
            box-shadow: 0 8px 0 rgba(255,255,255,0.8), 0 16px 0 rgba(255,255,255,0.8);
        }

        .brand-mark::after {
            left: 16px;
            width: 2px;
            height: 18px;
            top: 5px;
            box-shadow: none;
            background: rgba(255,255,255,0.94);
        }

        .brand-text {
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
            color: #f7fbff;
        }

        [data-testid="stSidebar"] button[kind="primary"] {
            background: linear-gradient(90deg, rgba(79, 135, 255, 0.34), rgba(79, 135, 255, 0.1));
            border: 1px solid rgba(122, 188, 255, 0.38);
            color: white;
        }

        .page-title {
            font-size: clamp(2rem, 2.4vw, 3rem);
            font-weight: 800;
            letter-spacing: -0.05em;
            color: #f4faff;
            margin: 0;
            line-height: 1.1;
        }

        .page-subtitle {
            color: var(--muted);
            margin-top: 6px;
            font-size: 0.97rem;
        }

        .metric-card {
            background: linear-gradient(180deg, rgba(16, 39, 71, 0.92), rgba(10, 28, 54, 0.9));
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 18px 30px rgba(8, 14, 27, 0.36);
            padding: 18px 18px 16px;
            min-height: 130px;
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--cyan), var(--blue), var(--purple));
            opacity: 0.85;
        }

        .metric-card.green::before { background: linear-gradient(90deg, var(--green), var(--cyan)); }
        .metric-card.red::before { background: linear-gradient(90deg, #ff6d8d, #ff9d9d); }
        .metric-card.purple::before { background: linear-gradient(90deg, #a38cff, var(--violet, #9b89ff)); }

        .metric-card .label-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: var(--muted);
            font-size: 0.83rem;
            margin-bottom: 12px;
        }

        .metric-card .icon-badge {
            width: 30px;
            height: 30px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(103, 177, 255, 0.12);
            border: 1px solid rgba(120, 183, 255, 0.2);
            color: #d0ecff;
            font-size: 0.96rem;
        }

        .metric-card.green .icon-badge { background: rgba(70, 211, 161, 0.12); }
        .metric-card.red .icon-badge { background: rgba(255, 101, 123, 0.12); }
        .metric-card.purple .icon-badge { background: rgba(143, 120, 255, 0.12); }

        .metric-value {
            font-size: clamp(1.6rem, 2vw, 2.1rem);
            font-weight: 800;
            letter-spacing: -0.04em;
            color: #f3f9ff;
            margin: 6px 0 0;
        }

        .metric-delta {
            margin-top: 8px;
            font-size: 0.8rem;
            color: var(--muted);
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .metric-delta .up { color: var(--green); }
        .metric-delta .down { color: var(--red); }

        .panel-heading {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }

        .panel-title {
            font-size: 1.06rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: #edf8ff;
            margin: 0;
        }

        .panel-note {
            color: var(--muted);
            font-size: 0.8rem;
        }

        .donut-legend {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px 12px;
            margin-top: 8px;
        }

        .legend-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            color: var(--muted);
            font-size: 0.9rem;
        }

        .legend-left {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            white-space: nowrap;
            border-radius: 999px;
            padding: 4px 8px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.01em;
            border: 1px solid transparent;
        }

        .chip.positive {
            background: rgba(65, 224, 161, 0.15);
            border-color: rgba(65, 224, 161, 0.35);
            color: var(--green);
        }

        .chip.neutral {
            background: rgba(255, 209, 102, 0.12);
            border-color: rgba(255, 209, 102, 0.28);
            color: #f6d57f;
        }

        .chip.negative {
            background: rgba(255, 95, 122, 0.14);
            border-color: rgba(255, 95, 122, 0.34);
            color: #ff8ba1;
        }

        .insight-card {
            padding: 12px;
            border: 1px solid rgba(118, 156, 240, 0.24);
            border-radius: 10px;
            background: linear-gradient(145deg, rgba(18, 42, 78, 0.72), rgba(9, 26, 51, 0.76));
            color: var(--text);
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .insight-card strong {
            color: #f1f7ff;
            font-size: 0.96rem;
        }

        .insight-card div {
            color: var(--cyan);
            font-size: 0.82rem;
        }

        .analysis-result {
            margin-top: 8px;
            padding: 12px 14px;
            border-radius: 12px;
            background: rgba(24, 48, 83, 0.8);
            border: 1px solid rgba(124, 170, 255, 0.18);
            color: #eaf5ff;
        }

        .analysis-result.positive {
            border-color: rgba(65, 224, 161, 0.4);
            box-shadow: inset 3px 0 0 var(--green);
        }

        .analysis-result.neutral {
            border-color: rgba(255, 209, 102, 0.38);
            box-shadow: inset 3px 0 0 var(--amber);
        }

        .analysis-result.negative {
            border-color: rgba(255, 95, 122, 0.4);
            box-shadow: inset 3px 0 0 var(--red);
        }

        @media (max-width: 1100px) {
            .page-title { font-size: 2rem; }
        }

        @media (max-width: 720px) {
            .panel-heading { align-items: flex-start; flex-direction: column; gap: 4px; }
        }
    </style>
    """


st.markdown(get_dashboard_css(), unsafe_allow_html=True)


if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Overview"


def sidebar_nav() -> str:
    nav_items = [
        "Overview",
        "Review Explorer",
        "Analyse a Review",
        "Business Insights",
        "Settings",
    ]
    st.sidebar.markdown('<div class="brand-wrap"><div class="brand-mark"></div><div class="brand-text">SentimentIQ</div></div>', unsafe_allow_html=True)

    active = st.session_state.active_nav
    for item in nav_items:
        label = item
        icon = {
            "Overview": "🏠",
            "Review Explorer": "🔎",
            "Analyse a Review": "📝",
            "Business Insights": "📈",
            "Settings": "⚙️",
        }[item]
        button_key = f"nav_{item}"
        if st.sidebar.button(
            f"{icon} {label}",
            key=button_key,
            type="primary" if item == active else "secondary",
            use_container_width=True,
        ):
            st.session_state.active_nav = item
            active = item

    return active


def render_metric_card(title: str, value: str, delta_text: str, delta_class: str, icon: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card {accent}">
            <div class="label-row">
                <span>{title}</span>
                <span class="icon-badge">{icon}</span>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta"><span class="{delta_class}">{delta_text}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_date_column(df: pd.DataFrame) -> str | None:
    date_candidates = {
        "date",
        "reviewdate",
        "timestamp",
        "createdat",
        "dateadded",
        "orderdate",
        "reviewtimestamp",
        "publishedat",
        "submittedat",
    }
    for column in df.columns:
        normalized_name = "".join(character for character in str(column).lower() if character.isalnum())
        if normalized_name in date_candidates:
            converted = pd.to_datetime(df[column], errors="coerce")
            if converted.notna().any():
                return column
    return None


def build_sentiment_summary(df: pd.DataFrame) -> pd.DataFrame:
    sentiment_order = ["Positive", "Neutral", "Negative"]
    summary = pd.DataFrame({
        "sentiment": sentiment_order,
        "count": [int((df[ANALYSIS_SENTIMENT_COLUMN] == value).sum()) for value in sentiment_order],
    })
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100 if summary["count"].sum() else 0
    summary["percentage"] = summary["percentage"].fillna(0)
    return summary


def build_donut_chart(summary_df: pd.DataFrame, mode: str = "count") -> go.Figure:
    if summary_df.empty or summary_df["count"].sum() == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.add_annotation(text="No review sentiment data available", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#d9ebff"))
        return fig

    values = summary_df["count" if mode == "count" else "percentage"].tolist()
    labels = summary_df["sentiment"].tolist()
    colors = ["#41e0a1", "#b8c2d9", "#ff5f7a"]
    value_template = "%{label}<br>%{value}" if mode == "count" else "%{label}<br>%{percent}"

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.68,
                marker=dict(colors=colors),
                texttemplate=value_template,
                hovertemplate="<b>%{label}</b><br>Count: %{customdata[0]}<br>Percentage: %{percent:.1%}<extra></extra>",
                customdata=[[int(count), float(percent)] for count, percent in zip(summary_df["count"], summary_df["percentage"])],
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    return fig


def build_trend_chart(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.add_annotation(text="No data available for the selected filters", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=14, color="#d9ebff"))
        return fig

    date_col = find_date_column(df)
    if date_col is None:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.add_annotation(text="No valid review dates available for trend analysis", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13, color="#d9ebff"))
        return fig

    chart_df = df.copy()
    chart_df[date_col] = pd.to_datetime(chart_df[date_col], errors="coerce")
    chart_df = chart_df.dropna(subset=[date_col]).copy()
    if chart_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig.add_annotation(text="No valid review dates available for trend analysis", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13, color="#d9ebff"))
        return fig

    chart_df = chart_df.sort_values(date_col)
    chart_df["date_bucket"] = chart_df[date_col].dt.strftime("%Y-%m-%d")
    grouped = chart_df.groupby("date_bucket", as_index=False).agg(
        Positive=(ANALYSIS_SENTIMENT_COLUMN, lambda s: (s == "Positive").sum()),
        Neutral=(ANALYSIS_SENTIMENT_COLUMN, lambda s: (s == "Neutral").sum()),
        Negative=(ANALYSIS_SENTIMENT_COLUMN, lambda s: (s == "Negative").sum()),
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grouped["date_bucket"],
        y=grouped["Positive"],
        mode="lines+markers",
        name="Positive",
        line=dict(color="#41e0a1", width=3),
        marker=dict(size=7),
        hovertemplate="<b>Positive</b><br>Date: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["date_bucket"],
        y=grouped["Neutral"],
        mode="lines+markers",
        name="Neutral",
        line=dict(color="#b8c2d9", width=3),
        marker=dict(size=7),
        hovertemplate="<b>Neutral</b><br>Date: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=grouped["date_bucket"],
        y=grouped["Negative"],
        mode="lines+markers",
        name="Negative",
        line=dict(color="#ff5f7a", width=3),
        marker=dict(size=7),
        hovertemplate="<b>Negative</b><br>Date: %{x}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18, r=18, t=20, b=28),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, x=0.0),
        xaxis=dict(showgrid=True, gridcolor="rgba(152, 178, 214, 0.12)", tickangle=-25),
        yaxis=dict(showgrid=True, gridcolor="rgba(152, 178, 214, 0.12)", rangemode="tozero"),
        hovermode="x unified",
    )
    return fig


def reset_review_explorer_page() -> None:
    st.session_state["review_explorer_page"] = 1


def reset_dashboard_filters() -> None:
    st.session_state["sentiment_filter"] = ["Positive", "Neutral", "Negative"]
    for key in ("date_start", "date_end", "date_range"):
        st.session_state.pop(key, None)


def reset_review_explorer_filters() -> None:
    st.session_state["review_explorer_search"] = ""
    st.session_state["review_explorer_sentiments"] = ["Positive", "Neutral", "Negative"]
    st.session_state["review_explorer_sort"] = "Score: highest first"
    st.session_state["review_explorer_page_size"] = 10
    st.session_state["review_explorer_page"] = 1
    clear_insight_filter()


def add_analysis_columns_for_export(filtered_df: pd.DataFrame) -> pd.DataFrame:
    export_df = filtered_df[SOURCE_COLUMNS].copy()
    sentiment_name = "Detected sentiment"
    while sentiment_name in export_df.columns:
        sentiment_name = f"{sentiment_name} (analysis)"
    score_name = "Compound polarity"
    while score_name in export_df.columns:
        score_name = f"{score_name} (analysis)"
    export_df[sentiment_name] = filtered_df[ANALYSIS_SENTIMENT_COLUMN]
    export_df[score_name] = filtered_df[ANALYSIS_SCORE_COLUMN]
    return export_df


def render_table_section(df: pd.DataFrame, *, explorer: bool = False) -> None:
    if explorer:
        sentiment_options = ["Positive", "Neutral", "Negative"]
        sort_options = ["Score: highest first", "Score: lowest first", "Sentiment: A to Z", "Sentiment: Z to A"]
        date_column = find_date_column(df)
        if date_column:
            sort_options.extend(["Date: newest first", "Date: oldest first"])

        if "review_explorer_sentiments" not in st.session_state:
            st.session_state["review_explorer_sentiments"] = sentiment_options.copy()
        if st.session_state.get("review_explorer_sort") not in sort_options:
            st.session_state["review_explorer_sort"] = "Score: highest first"

        search_col, sentiment_col, sort_col = st.columns([1.6, 1.4, 1.5])
        with search_col:
            search_term = st.text_input(
                "Search review text",
                placeholder="Type to search reviews...",
                key="review_explorer_search",
                on_change=reset_review_explorer_page,
            )
        with sentiment_col:
            selected_sentiments = st.multiselect(
                "Sentiment",
                options=sentiment_options,
                key="review_explorer_sentiments",
                on_change=reset_review_explorer_page,
            )
        with sort_col:
            sort_order = st.selectbox(
                "Sort reviews",
                options=sort_options,
                key="review_explorer_sort",
                on_change=reset_review_explorer_page,
            )
        _, reset_col = st.columns([4, 1])
        with reset_col:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            st.button("Reset filters", key="review_explorer_reset", on_click=reset_review_explorer_filters, use_container_width=True)

        working = df[df[ANALYSIS_SENTIMENT_COLUMN].isin(selected_sentiments)].copy()
        insight_indices = st.session_state.get("insight_review_indices")
        if insight_indices is not None:
            working = working[working.index.isin(insight_indices)]
        if search_term.strip():
            review_values = working["review"].map(source_text)
            working = working[review_values.str.contains(search_term.strip(), case=False, regex=False, na=False)]

        if sort_order == "Score: highest first":
            working = working.sort_values(ANALYSIS_SCORE_COLUMN, ascending=False, na_position="last")
        elif sort_order == "Score: lowest first":
            working = working.sort_values(ANALYSIS_SCORE_COLUMN, ascending=True, na_position="last")
        elif sort_order == "Sentiment: A to Z":
            working = working.sort_values(ANALYSIS_SENTIMENT_COLUMN, ascending=True, na_position="last")
        elif sort_order == "Sentiment: Z to A":
            working = working.sort_values(ANALYSIS_SENTIMENT_COLUMN, ascending=False, na_position="last")
        elif sort_order.startswith("Date:") and date_column:
            working = working.sort_values(
                date_column,
                ascending=sort_order == "Date: oldest first",
                key=lambda values: pd.to_datetime(values, errors="coerce"),
                na_position="last",
            )

        result_count = len(working)
        st.markdown(f"**{result_count:,} matching reviews**")

        export_df = add_analysis_columns_for_export(working)
        st.download_button(
            "Download filtered CSV",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="filtered_reviews.csv",
            mime="text/csv",
            key="review_explorer_download",
        )

        page_size = st.selectbox(
            "Rows per page",
            options=[5, 10, 25, 50, 100],
            key="review_explorer_page_size",
            on_change=reset_review_explorer_page,
        )
        total_pages = max(1, math.ceil(result_count / page_size))
        current_page = min(max(1, st.session_state.get("review_explorer_page", 1)), total_pages)
        st.session_state["review_explorer_page"] = current_page
        start = (current_page - 1) * page_size
        page_df = working.iloc[start:start + page_size]

        if page_df.empty:
            st.info("No reviews match the current search and sentiment filters.")
        else:
            for row_number, (row_index, row) in enumerate(page_df.iterrows(), start=start + 1):
                sentiment = row[ANALYSIS_SENTIMENT_COLUMN]
                sentiment_class = sentiment.lower()
                display_date = ""
                if date_column and pd.notna(row[date_column]):
                    parsed_date = pd.to_datetime(row[date_column], errors="coerce")
                    display_date = parsed_date.strftime("%Y-%m-%d") if pd.notna(parsed_date) else ""
                row_columns = st.columns([1.6, 5.4, 1.1, 1.4] if date_column else [1.6, 6.2, 1.1])
                badge_col, review_col, score_col = row_columns[:3]
                date_col = row_columns[3] if date_column else None
                with badge_col:
                    st.markdown(
                        f'<span class="chip {sentiment_class}">{get_sentiment_eps(sentiment)} {sentiment}</span>',
                        unsafe_allow_html=True,
                    )
                with review_col:
                    review_value = source_text(row["review"])
                    st.write(review_value if review_value else "(empty review)")
                with score_col:
                    st.caption("Score")
                    st.write(f"{row[ANALYSIS_SCORE_COLUMN]:+.2f}")
                if date_col is not None:
                    with date_col:
                        st.caption("Date")
                        st.write(display_date or "No date")

                with st.expander(f"Review details · {row_number}"):
                    st.markdown("**Full review**")
                    st.write(review_value if review_value else "(empty review)")
                    source_values = [source_text(row[column]) or "(empty)" for column in SOURCE_COLUMNS]
                    details_df = pd.DataFrame({"Source column": SOURCE_COLUMNS, "Original value": source_values})
                    st.dataframe(details_df, hide_index=True, use_container_width=True)

        previous_col, page_status_col, next_col = st.columns([1, 2, 1])
        with previous_col:
            st.button(
                "Previous",
                key="review_explorer_previous",
                disabled=current_page <= 1,
                on_click=lambda: st.session_state.update(review_explorer_page=current_page - 1),
                use_container_width=True,
            )
        with page_status_col:
            st.caption(f"Page {current_page} of {total_pages} · showing {len(page_df)} of {result_count}")
        with next_col:
            st.button(
                "Next",
                key="review_explorer_next",
                disabled=current_page >= total_pages,
                on_click=lambda: st.session_state.update(review_explorer_page=current_page + 1),
                use_container_width=True,
            )
        return

    search_term = st.text_input("Search reviews", placeholder="Search reviews...", key="table_search")
    working = df.copy()
    if search_term.strip():
        review_values = working["review"].map(source_text)
        working = working[review_values.str.contains(search_term.strip(), case=False, regex=False, na=False)]
    working = working.sort_values(ANALYSIS_SCORE_COLUMN, ascending=False, na_position="last")
    page_size = 5
    total_pages = max(1, math.ceil(len(working) / page_size))
    current_page = min(max(1, st.session_state.get("table_page", 1)), total_pages)
    st.session_state.table_page = current_page
    subset = working.iloc[(current_page - 1) * page_size:current_page * page_size]
    display_data = pd.DataFrame({
        "Review": subset["review"].map(lambda value: source_text(value) or "(empty review)"),
        "Sentiment": subset[ANALYSIS_SENTIMENT_COLUMN],
        "Compound polarity": subset[ANALYSIS_SCORE_COLUMN],
    })
    date_column = find_date_column(subset)
    if date_column:
        display_data["Date"] = subset[date_column].map(lambda value: source_text(value) or "—")
    st.dataframe(
        display_data,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Review": st.column_config.TextColumn(width="large"),
            "Compound polarity": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    page_col1, page_col2 = st.columns([1, 1])
    with page_col1:
        if st.button("Prev", key="prev_page", disabled=current_page <= 1):
            st.session_state.table_page = max(1, current_page - 1)
    with page_col2:
        if st.button("Next", key="next_page", disabled=current_page >= total_pages):
            st.session_state.table_page = min(total_pages, current_page + 1)
    st.caption(f"Showing {len(subset)} of {len(working)} reviews")


def find_group_columns(df: pd.DataFrame) -> list[str]:
    group_column_names = {
        "product": "Product",
        "productname": "Product",
        "item": "Product",
        "itemname": "Product",
        "sku": "Product",
        "category": "Category",
        "productcategory": "Category",
        "producttype": "Category",
        "department": "Category",
    }
    result = []
    for column in SOURCE_COLUMNS:
        normalized_name = "".join(character for character in str(column).lower() if character.isalnum())
        if normalized_name in group_column_names and column in df.columns:
            result.append(column)
    return result


def analyze_dataset_insights(df: pd.DataFrame) -> dict:
    sentiment_order = ["Positive", "Neutral", "Negative"]
    total = len(df)
    sentiment_counts = {
        sentiment: int((df[ANALYSIS_SENTIMENT_COLUMN] == sentiment).sum())
        for sentiment in sentiment_order
    }
    sentiment_details = {}
    for sentiment, count in sentiment_counts.items():
        matching = df[df[ANALYSIS_SENTIMENT_COLUMN] == sentiment]
        sentiment_details[sentiment] = {
            "count": count,
            "percentage": count / total * 100 if total else 0.0,
            "indices": matching.index.tolist(),
        }

    review_text = df["review"].map(source_text).astype(str)
    empty_review_count = int(review_text.str.strip().eq("").sum())
    themes = []
    for theme, pattern in COMPILED_THEME_RULES.items():
        matched_mask = review_text.map(lambda text: bool(pattern.search(str(text))))
        matched = df.loc[matched_mask]
        if matched.empty:
            continue
        positive = matched[matched[ANALYSIS_SENTIMENT_COLUMN] == "Positive"]
        neutral = matched[matched[ANALYSIS_SENTIMENT_COLUMN] == "Neutral"]
        negative = matched[matched[ANALYSIS_SENTIMENT_COLUMN] == "Negative"]
        themes.append({
            "name": theme,
            "keywords": THEME_KEYWORDS[theme],
            "count": len(matched),
            "positive_count": len(positive),
            "neutral_count": len(neutral),
            "negative_count": len(negative),
            "positive_indices": positive.index.tolist(),
            "negative_indices": negative.index.tolist(),
            "indices": matched.index.tolist(),
        })

    group_comparisons = []
    for column in find_group_columns(df):
        group_values = df[column].map(source_text).str.strip()
        group_summaries = []
        for group_name in group_values[group_values != ""].unique():
            group_mask = group_values == group_name
            group = df.loc[group_mask]
            count = len(group)
            positive_count = int((group[ANALYSIS_SENTIMENT_COLUMN] == "Positive").sum())
            neutral_count = int((group[ANALYSIS_SENTIMENT_COLUMN] == "Neutral").sum())
            negative_count = int((group[ANALYSIS_SENTIMENT_COLUMN] == "Negative").sum())
            group_summaries.append({
                "group": group_name,
                "count": count,
                "positive_count": positive_count,
                "neutral_count": neutral_count,
                "negative_count": negative_count,
                "positive_percentage": positive_count / count * 100 if count else 0.0,
                "negative_percentage": negative_count / count * 100 if count else 0.0,
                "indices": group.index.tolist(),
            })
        group_summaries.sort(key=lambda item: item["count"], reverse=True)
        if len(group_summaries) > 1:
            group_comparisons.append({
                "column": column,
                "label": "Category" if "category" in column.lower() or "department" in column.lower() else "Product",
                "groups": group_summaries,
            })

    positive_themes = sorted(themes, key=lambda item: item["positive_count"], reverse=True)
    negative_themes = sorted(themes, key=lambda item: item["negative_count"], reverse=True)
    positive_themes = [theme for theme in positive_themes if theme["positive_count"] > 0]
    negative_themes = [theme for theme in negative_themes if theme["negative_count"] > 0]

    recommendations = []
    if total:
        negative_detail = sentiment_details["Negative"]
        if negative_detail["count"]:
            recommendations.append({
                "title": "Triage negative feedback first",
                "statistic": f"{negative_detail['count']} of {total} reviews ({negative_detail['percentage']:.1f}%) are Negative",
                "explanation": "Review these original comments and route confirmed issues to the relevant product or service owner.",
                "indices": negative_detail["indices"],
            })
        else:
            recommendations.append({
                "title": "Keep monitoring incoming feedback",
                "statistic": f"0 of {total} reviews are Negative",
                "explanation": "No negative-classified reviews appear in this dataset. Continue monitoring; this sample alone does not establish that issues are absent.",
                "indices": df.index.tolist(),
            })

        if negative_themes:
            theme = negative_themes[0]
            recommendations.append({
                "title": f"Investigate {theme['name'].lower()} complaints",
                "statistic": f"{theme['negative_count']} negative reviews mention this theme",
                "explanation": "Compare these comments with the affected customer journey and verify the underlying issue before prioritizing a fix.",
                "indices": theme["negative_indices"],
            })
        elif sentiment_details["Negative"]["count"]:
            recommendations.append({
                "title": "Read ungrouped negative feedback",
                "statistic": f"{sentiment_details['Negative']['count']} negative reviews; no configured theme rule matched",
                "explanation": "Inspect the original negative comments to identify issues that the transparent keyword rules do not cover.",
                "indices": sentiment_details["Negative"]["indices"],
            })
        elif positive_themes:
            theme = positive_themes[0]
            recommendations.append({
                "title": f"Preserve the experience around {theme['name'].lower()}",
                "statistic": f"{theme['positive_count']} positive reviews mention this theme",
                "explanation": "Keep the practices associated with these positive comments visible while monitoring future feedback.",
                "indices": theme["positive_indices"],
            })
        else:
            neutral_detail = sentiment_details["Neutral"]
            recommendations.append({
                "title": "Review neutral feedback for unmet expectations",
                "statistic": f"{neutral_detail['count']} of {total} reviews ({neutral_detail['percentage']:.1f}%) are Neutral",
                "explanation": "Read these comments for specific requests or friction not reflected by a strongly positive or negative tone.",
                "indices": neutral_detail["indices"],
            })

        if positive_themes:
            theme = positive_themes[0]
            recommendations.append({
                "title": f"Reinforce the positive {theme['name'].lower()} experience",
                "statistic": f"{theme['positive_count']} positive reviews mention this theme",
                "explanation": "Use these comments to understand which existing experience customers value; avoid assuming the same result across all products.",
                "indices": theme["positive_indices"],
            })
        elif sentiment_details["Neutral"]["count"]:
            neutral_detail = sentiment_details["Neutral"]
            recommendations.append({
                "title": "Inspect neutral comments for improvement opportunities",
                "statistic": f"{neutral_detail['count']} neutral reviews ({neutral_detail['percentage']:.1f}%)",
                "explanation": "Read the original comments for concrete unmet needs before changing the customer experience.",
                "indices": neutral_detail["indices"],
            })
        else:
            recommendations.append({
                "title": "Continue reviewing verbatim feedback",
                "statistic": f"{total} reviews analyzed",
                "explanation": "Keep sentiment summaries paired with the original comments so emerging issues are not hidden by aggregate percentages.",
                "indices": df.index.tolist(),
            })

        for comparison in group_comparisons:
            eligible_groups = [group for group in comparison["groups"] if group["count"] >= 2]
            complaint_groups = [group for group in eligible_groups if group["negative_count"] > 0]
            if complaint_groups:
                group = max(complaint_groups, key=lambda item: item["negative_percentage"])
                recommendations.append({
                    "title": f"Review {comparison['label'].lower()} feedback for {group['group']}",
                    "statistic": f"{group['negative_count']} of {group['count']} reviews ({group['negative_percentage']:.1f}%) are Negative",
                    "explanation": "This is the highest observed negative share among groups with at least two reviews; it is descriptive, not a significance test.",
                    "indices": group["indices"],
                })
                break

    return {
        "total": total,
        "empty_review_count": empty_review_count,
        "sentiments": sentiment_details,
        "themes": themes,
        "positive_themes": positive_themes,
        "negative_themes": negative_themes,
        "group_comparisons": group_comparisons,
        "recommendations": recommendations[:5],
    }


def filter_explorer_to_insight(indices: list, title: str, dataset_signature: str) -> None:
    st.session_state["insight_review_indices"] = indices
    st.session_state["insight_filter_title"] = title
    st.session_state["insight_dataset_signature"] = dataset_signature
    st.session_state["review_explorer_search"] = ""
    st.session_state["review_explorer_sentiments"] = ["Positive", "Neutral", "Negative"]
    st.session_state["review_explorer_sort"] = "Score: highest first"
    st.session_state["review_explorer_page_size"] = 10
    st.session_state["review_explorer_page"] = 1
    st.session_state["active_nav"] = "Review Explorer"


def clear_insight_filter() -> None:
    st.session_state.pop("insight_review_indices", None)
    st.session_state.pop("insight_filter_title", None)
    st.session_state["review_explorer_page"] = 1


def render_insight_card(title: str, statistic: str, explanation: str, indices: list, key: str, dataset_signature: str) -> None:
    st.markdown(
        f'<div class="insight-card"><strong>{html.escape(title)}</strong><div>{html.escape(statistic)}</div></div>',
        unsafe_allow_html=True,
    )
    st.write(explanation)
    st.button(
        "View related reviews",
        key=f"view_insight_{key}",
        on_click=filter_explorer_to_insight,
        args=(indices, title, dataset_signature),
    )


def build_insights_report(insights: dict) -> str:
    lines = [
        "# E-Commerce Sentiment Insights",
        "",
        f"Reviews analyzed: {insights['total']}",
        f"Reviews with empty text: {insights['empty_review_count']} (these receive the existing Neutral sentiment label)",
        "",
        "## Sentiment distribution",
    ]
    for sentiment, details in insights["sentiments"].items():
        lines.append(f"- {sentiment}: {details['count']} ({details['percentage']:.1f}%)")

    lines.extend(["", "## Review themes"])
    if insights["positive_themes"]:
        lines.append("### Positive themes")
        for theme in insights["positive_themes"]:
            lines.append(f"- {theme['name']}: {theme['positive_count']} positive reviews mention it")
    if insights["negative_themes"]:
        lines.append("### Negative themes and complaints")
        for theme in insights["negative_themes"]:
            lines.append(f"- {theme['name']}: {theme['negative_count']} negative reviews mention it")
    if not insights["positive_themes"] and not insights["negative_themes"]:
        lines.append("No configured theme terms were found in the review text.")

    for comparison in insights["group_comparisons"]:
        lines.extend(["", f"## {comparison['label']} comparison: {comparison['column']}"])
        for group in comparison["groups"]:
            lines.append(
                f"- {group['group']}: {group['count']} reviews; "
                f"{group['positive_percentage']:.1f}% positive; {group['negative_percentage']:.1f}% negative"
            )

    lines.extend(["", "## Recommendations"])
    for recommendation in insights["recommendations"]:
        lines.append(f"- **{recommendation['title']}** ({recommendation['statistic']}): {recommendation['explanation']}")

    lines.extend([
        "",
        "## Method",
        "Sentiment labels use the dashboard's existing VADER compound-score thresholds. Themes use case-insensitive keyword matching against the original review text; a review can match more than one theme. Product/category comparisons use only matching source columns present in the dataset. Counts and percentages are descriptive; no statistical significance is inferred.",
        "",
    ])
    return "\n".join(lines)


def render_key_insights(df: pd.DataFrame, *, compact: bool = False) -> None:
    insights = analyze_dataset_insights(df)
    dataset_signature = st.session_state.get("current_dataset_signature", "")
    st.caption(
        "Findings use the current dataset. Theme matching is case-insensitive keyword matching; a review may appear in multiple themes. "
        f"{insights['empty_review_count']} empty review(s) are included in sentiment totals and receive the existing Neutral label."
    )
    report = build_insights_report(insights)
    st.download_button(
        "Download summary report",
        data=report.encode("utf-8"),
        file_name="ecommerce_sentiment_insights.md",
        mime="text/markdown",
        key="insights_summary_download",
    )

    if insights["total"] == 0:
        st.info("No reviews are available for insight analysis.")
        return

    if compact:
        counts = insights["sentiments"]
        st.caption(
            f"Positive {counts['Positive']['percentage']:.1f}% · "
            f"Neutral {counts['Neutral']['percentage']:.1f}% · "
            f"Negative {counts['Negative']['percentage']:.1f}%"
        )
        compact_themes = []
        if insights["positive_themes"]:
            compact_themes.append(("Positive", insights["positive_themes"][0], "positive_count", "positive_indices"))
        if insights["negative_themes"]:
            compact_themes.append(("Complaint", insights["negative_themes"][0], "negative_count", "negative_indices"))
        if compact_themes:
            for sentiment, theme, count_key, indices_key in compact_themes:
                render_insight_card(
                    theme["name"],
                    f"{theme[count_key]} {sentiment.lower()} reviews mention this theme",
                    "Matched with transparent keyword rules against the original review text.",
                    theme[indices_key],
                    f"compact_{sentiment.lower()}",
                    dataset_signature,
                )
        elif insights["recommendations"]:
            recommendation = insights["recommendations"][0]
            render_insight_card(
                recommendation["title"],
                recommendation["statistic"],
                recommendation["explanation"],
                recommendation["indices"],
                "compact_recommendation",
                dataset_signature,
            )
        return

    st.markdown("#### Sentiment distribution")
    sentiment_cols = st.columns(3)
    for column, (sentiment, details) in zip(sentiment_cols, insights["sentiments"].items()):
        with column:
            render_insight_card(
                f"{sentiment} sentiment",
                f"{details['count']} reviews · {details['percentage']:.1f}%",
                f"{details['count']} of {insights['total']} reviews were labeled {sentiment} by the existing sentiment analysis.",
                details["indices"],
                f"sentiment_{sentiment.lower()}",
                dataset_signature,
            )

    positive_col, negative_col = st.columns(2)
    with positive_col:
        st.markdown("#### Common positive themes")
        if insights["positive_themes"]:
            for index, theme in enumerate(insights["positive_themes"][:3]):
                denominator = insights["sentiments"]["Positive"]["count"]
                percentage = theme["positive_count"] / denominator * 100 if denominator else 0
                render_insight_card(
                    theme["name"],
                    f"{theme['positive_count']} of {denominator} positive reviews · {percentage:.1f}%",
                    f"Matched review terms: {', '.join(theme['keywords'])}. Theme counts overlap when a review mentions multiple topics.",
                    theme["positive_indices"],
                    f"positive_theme_{index}",
                    dataset_signature,
                )
        else:
            st.caption("No positive review matched the configured theme terms.")
    with negative_col:
        st.markdown("#### Common complaints")
        if insights["negative_themes"]:
            for index, theme in enumerate(insights["negative_themes"][:3]):
                denominator = insights["sentiments"]["Negative"]["count"]
                percentage = theme["negative_count"] / denominator * 100 if denominator else 0
                render_insight_card(
                    theme["name"],
                    f"{theme['negative_count']} of {denominator} negative reviews · {percentage:.1f}%",
                    f"Matched review terms: {', '.join(theme['keywords'])}. These are descriptive matches, not proof of a cause.",
                    theme["negative_indices"],
                    f"negative_theme_{index}",
                    dataset_signature,
                )
        else:
            st.caption("No negative review matched the configured theme terms.")

    if insights["group_comparisons"]:
        st.markdown("#### Product and category comparisons")
        for comparison_index, comparison in enumerate(insights["group_comparisons"]):
            st.markdown(f"**{comparison['label']} · source column: {html.escape(str(comparison['column']))}**")
            comparison_df = pd.DataFrame(comparison["groups"][:8])
            comparison_df = comparison_df.rename(columns={
                "group": comparison["label"],
                "count": "Reviews",
                "positive_count": "Positive",
                "positive_percentage": "Positive %",
                "negative_count": "Negative",
                "negative_percentage": "Negative %",
            })[[comparison["label"], "Reviews", "Positive", "Positive %", "Negative", "Negative %"]]
            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Positive %": st.column_config.NumberColumn(format="%.1f%%"),
                    "Negative %": st.column_config.NumberColumn(format="%.1f%%"),
                },
            )
            for group_index, group in enumerate(comparison["groups"][:5]):
                render_insight_card(
                    f"{comparison['label']}: {group['group']}",
                    f"{group['count']} reviews · {group['positive_percentage']:.1f}% positive · {group['negative_percentage']:.1f}% negative",
                    "Descriptive comparison using the source field shown above; small groups should be interpreted cautiously.",
                    group["indices"],
                    f"group_{comparison_index}_{group_index}",
                    dataset_signature,
                )

    st.markdown("#### Customer experience opportunities")
    if insights["negative_themes"]:
        for index, theme in enumerate(insights["negative_themes"][:3]):
            st.markdown(f"- Review {theme['negative_count']} negative comments mentioning **{theme['name'].lower()}** and validate the issue with the relevant team.")
    else:
        st.write("No recurring complaint theme was identified by the configured keyword rules. Review original negative comments directly where available.")

    st.markdown("#### Evidence-based recommendations")
    for index, recommendation in enumerate(insights["recommendations"]):
        render_insight_card(
            recommendation["title"],
            recommendation["statistic"],
            recommendation["explanation"],
            recommendation["indices"],
            f"recommendation_{index}",
            dataset_signature,
        )


uploaded_file = st.file_uploader("Upload CSV review dataset", type=["csv"], key="csv_upload")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV is empty. Add a header row and review data, then try again.")
        st.stop()
    except pd.errors.ParserError:
        st.error("The CSV could not be read. Check that every row has the same number of columns and that quoted text is closed.")
        st.stop()
    except UnicodeDecodeError:
        st.error("The CSV text encoding could not be read. Export the file as UTF-8 CSV and upload it again.")
        st.stop()
    except Exception:
        st.error("The uploaded file could not be read as a CSV. Check the file and try uploading it again.")
        st.stop()
    dataset_name = uploaded_file.name
else:
    default_csv = resolve_default_csv()
    if default_csv is not None:
        df = pd.read_csv(default_csv)
        dataset_name = default_csv.name
    else:
        st.warning("Please upload a CSV file or add a sample_reviews.csv file in the project root.")
        st.stop()

if "review" not in df.columns:
    st.error("This dataset is missing the required 'review' column. Rename the review-text column to 'review' and upload the CSV again.")
    st.stop()

SOURCE_COLUMNS = list(df.columns)
while ANALYSIS_SENTIMENT_COLUMN in SOURCE_COLUMNS:
    ANALYSIS_SENTIMENT_COLUMN = f"_{ANALYSIS_SENTIMENT_COLUMN}"
while ANALYSIS_SCORE_COLUMN in SOURCE_COLUMNS or ANALYSIS_SCORE_COLUMN == ANALYSIS_SENTIMENT_COLUMN:
    ANALYSIS_SCORE_COLUMN = f"_{ANALYSIS_SCORE_COLUMN}"

clean_df = df.copy()
clean_df[ANALYSIS_SENTIMENT_COLUMN] = clean_df["review"].map(lambda value: analyze_sentiment(source_text(value)))
clean_df[ANALYSIS_SCORE_COLUMN] = clean_df["review"].map(lambda value: sia.polarity_scores(source_text(value))["compound"])
dataset_hash = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy(dtype="uint64").tobytes()).hexdigest()
dataset_signature = f"{dataset_name}:{len(df)}:{dataset_hash}"
if st.session_state.get("current_dataset_signature") != dataset_signature:
    st.session_state.pop("insight_review_indices", None)
    st.session_state.pop("insight_filter_title", None)
    st.session_state.pop("insight_dataset_signature", None)
    for state_key in (
        "date_start",
        "date_end",
        "date_range",
        "review_explorer_search",
        "review_explorer_sentiments",
        "review_explorer_sort",
        "review_explorer_page_size",
        "review_explorer_page",
        "table_search",
        "table_page",
    ):
        st.session_state.pop(state_key, None)
    st.session_state["sentiment_filter"] = ["Positive", "Neutral", "Negative"]
st.session_state["current_dataset_signature"] = dataset_signature

selected_nav = sidebar_nav()

if "sentiment_filter" not in st.session_state:
    st.session_state["sentiment_filter"] = ["Positive", "Neutral", "Negative"]

col_title, col_actions = st.columns([2.5, 1.1])
with col_title:
    st.markdown('<div class="page-title">E-commerce sentiment overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Understand what customers are saying and where to improve.</div>', unsafe_allow_html=True)
with col_actions:
    st.caption(f"Dataset: {Path(dataset_name).name}")

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)


def apply_dashboard_filters(source_df: pd.DataFrame):
    sentiment_order = ["Positive", "Neutral", "Negative"]
    filtered = source_df.copy()
    date_col = find_date_column(source_df)
    if date_col is not None:
        valid_dates = pd.to_datetime(source_df[date_col], errors="coerce").dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            if "date_start" not in st.session_state:
                st.session_state["date_start"] = min_date
            if "date_end" not in st.session_state:
                st.session_state["date_end"] = max_date
            date_start = st.session_state.get("date_start", min_date)
            date_end = st.session_state.get("date_end", max_date)
            if date_start < min_date:
                date_start = min_date
            if date_end > max_date:
                date_end = max_date
            if min_date == max_date:
                st.caption("Only one valid review date is available; the date filter is fixed.")
                start_value, end_value = min_date, max_date
            else:
                start_value, end_value = st.slider(
                    "Review date range",
                    min_value=min_date,
                    max_value=max_date,
                    value=(date_start, date_end),
                    key="date_range",
                )
            st.session_state["date_start"] = start_value
            st.session_state["date_end"] = end_value
            filtered = filtered[
                (pd.to_datetime(filtered[date_col], errors="coerce").dt.date >= start_value)
                & (pd.to_datetime(filtered[date_col], errors="coerce").dt.date <= end_value)
            ]

    selected_sentiments = st.multiselect(
        "Filter by sentiment",
        options=sentiment_order,
        key="sentiment_filter",
    )
    _, reset_col = st.columns([4, 1])
    with reset_col:
        st.button("Reset filters", key="dashboard_filter_reset", on_click=reset_dashboard_filters)

    filtered = filtered[filtered[ANALYSIS_SENTIMENT_COLUMN].isin(selected_sentiments)].copy()
    return filtered


if selected_nav == "Overview":
    filtered_df = apply_dashboard_filters(clean_df)
    summary_df = build_sentiment_summary(filtered_df)
    total_reviews = len(filtered_df)
    positive_reviews = int((filtered_df[ANALYSIS_SENTIMENT_COLUMN] == "Positive").sum())
    negative_reviews = int((filtered_df[ANALYSIS_SENTIMENT_COLUMN] == "Negative").sum())
    neutral_reviews = int((filtered_df[ANALYSIS_SENTIMENT_COLUMN] == "Neutral").sum())
    avg_score = float(filtered_df[ANALYSIS_SCORE_COLUMN].mean()) if not filtered_df.empty else 0.0
    positive_pct = (positive_reviews / total_reviews * 100) if total_reviews else 0
    negative_pct = (negative_reviews / total_reviews * 100) if total_reviews else 0

    metric_cols = st.columns(2)
    with metric_cols[0]:
        render_metric_card("Total Reviews", f"{total_reviews:,}", "Live filtered data", "up", "◔", "blue")
    with metric_cols[1]:
        render_metric_card("Positive sentiment", f"{positive_pct:.1f}%", f"{positive_reviews} reviews", "up", "😊", "green")
    with metric_cols[0]:
        render_metric_card("Negative reviews", f"{negative_pct:.1f}%", f"{negative_reviews} reviews", "down", "☹️", "red")
    with metric_cols[1]:
        render_metric_card("Average sentiment score", f"{avg_score:+.2f}", f"Neutral: {neutral_reviews}", "up", "⭐", "purple")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns([1.1, 1.2])
    with chart_col1:
        st.markdown('<div class="panel-heading"><h3 class="panel-title">Sentiment distribution</h3><div class="panel-note">Breakdown of review sentiment in your dataset</div></div>', unsafe_allow_html=True)
        donut_mode = st.radio("Display", options=["Count", "Percentage"], horizontal=True, key="donut_mode")
        donut_fig = build_donut_chart(summary_df, mode="count" if donut_mode == "Count" else "percentage")
        st.plotly_chart(donut_fig, use_container_width=True, config={"displayModeBar": False})
        legend_values = {
            sentiment: int(count)
            for sentiment, count in zip(summary_df["sentiment"].tolist(), summary_df["count"].tolist())
        }
        st.markdown(
            """
            <div class="donut-legend">
                <div class="legend-row"><div class="legend-left"><span class="dot" style="background:#41e0a1"></span>Positive</div><strong>{}</strong></div>
                <div class="legend-row"><div class="legend-left"><span class="dot" style="background:#b8c2d9"></span>Neutral</div><strong>{}</strong></div>
                <div class="legend-row"><div class="legend-left"><span class="dot" style="background:#ff5f7a"></span>Negative</div><strong>{}</strong></div>
            </div>
            """.format(
                legend_values.get("Positive", 0),
                legend_values.get("Neutral", 0),
                legend_values.get("Negative", 0),
            ),
            unsafe_allow_html=True,
        )
    with chart_col2:
        st.markdown('<div class="panel-heading"><h3 class="panel-title">Sentiment trends</h3><div class="panel-note">How sentiment changes over time</div></div>', unsafe_allow_html=True)
        trend_fig = build_trend_chart(filtered_df)
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    table_col, insight_col = st.columns([1.5, 1])

    with table_col:
        st.markdown('<div class="panel-heading"><h3 class="panel-title">Customer reviews</h3><div class="panel-note">Browse and search customer feedback</div></div>', unsafe_allow_html=True)
        render_table_section(filtered_df)

    with insight_col:
        st.markdown('<div class="panel-heading"><h3 class="panel-title">Key insights</h3><div class="panel-note">Data-driven findings from your reviews</div></div>', unsafe_allow_html=True)
        render_key_insights(filtered_df, compact=True)

elif selected_nav == "Review Explorer":
    st.markdown('<div class="panel-heading"><h3 class="panel-title">Review Explorer</h3><div class="panel-note">Search, filter, sort, inspect and export customer feedback</div></div>', unsafe_allow_html=True)
    insight_indices = st.session_state.get("insight_review_indices")
    if insight_indices is not None:
        if st.session_state.get("insight_dataset_signature") != dataset_signature:
            clear_insight_filter()
        else:
            st.info(f"Showing reviews for insight: {st.session_state.get('insight_filter_title', 'selected insight')}")
            st.button("Clear insight filter", key="clear_insight_filter_button", on_click=clear_insight_filter)
    render_table_section(clean_df, explorer=True)

elif selected_nav == "Analyse a Review":
    st.session_state.setdefault("review_input", "")
    st.session_state.setdefault("review_analysis_result", None)
    st.session_state.setdefault("review_analysis_error", None)
    st.session_state.setdefault("review_analysis_history", [])

    example_reviews = [
        ("Positive", "The product arrived quickly and works perfectly. Excellent quality and I would buy it again."),
        ("Neutral", "The package arrived on Tuesday. The item matches the description and includes the standard accessories."),
        ("Negative", "The item stopped working after two days. The quality is poor and customer support has not responded."),
    ]

    st.markdown('<div class="panel-heading"><h3 class="panel-title">Analyse a review</h3><div class="panel-note">VADER lexicon-based sentiment analysis, processed locally</div></div>', unsafe_allow_html=True)

    editor_col, examples_col = st.columns([1.35, 1])
    with editor_col:
        review_text = st.text_area(
            "Customer review",
            height=180,
            max_chars=2000,
            placeholder="Paste or write a customer review...",
            key="review_input",
            on_change=clear_review_result,
        )
        st.caption(f"{len(review_text)} / 2,000 characters")

        analyze_col, clear_col = st.columns(2)
        with analyze_col:
            st.button("Analyse Review", key="analyze_single", type="primary", use_container_width=True, on_click=analyze_review_input)
        with clear_col:
            st.button("Clear", key="clear_single", use_container_width=True, on_click=clear_review_input)

        if st.session_state["review_analysis_error"]:
            st.error(st.session_state["review_analysis_error"])

        result = st.session_state["review_analysis_result"]
        if result:
            sentiment_class = result["sentiment"].lower()
            st.markdown(
                f"""
                <div class="analysis-result {sentiment_class}">
                    <strong>Result</strong>
                    <span class="chip {sentiment_class}" style="margin-left: 10px">{get_sentiment_eps(result["sentiment"])} {result["sentiment"]}</span>
                    <p style="margin: 12px 0 0 0">{result["explanation"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("VADER compound polarity", f"{result['score']:+.3f}")
            st.caption("Compound polarity ranges from -1 (most negative) to +1 (most positive). It is not a probability or calibrated confidence score.")

    with examples_col:
        st.markdown('<div class="panel-heading"><h3 class="panel-title">Example reviews</h3><div class="panel-note">Select one to analyse it immediately</div></div>', unsafe_allow_html=True)
        for index, (example_sentiment, sample) in enumerate(example_reviews):
            st.caption(f"{get_sentiment_eps(example_sentiment)} {example_sentiment}")
            st.write(sample)
            st.button(
                "Analyse this example",
                key=f"example_review_{index}",
                use_container_width=True,
                on_click=save_review_analysis,
                args=(sample,),
                kwargs={"populate_input": True},
            )

    history = st.session_state["review_analysis_history"]
    with st.expander(f"Analysis history ({len(history)})", expanded=bool(history)):
        if history:
            history_df = pd.DataFrame(history).rename(
                columns={
                    "analyzed_at": "Analysed at",
                    "review": "Review",
                    "sentiment": "Sentiment",
                    "score": "Compound polarity",
                }
            )[["Analysed at", "Review", "Sentiment", "Compound polarity"]]
            st.dataframe(
                history_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Review": st.column_config.TextColumn(width="large"),
                    "Compound polarity": st.column_config.NumberColumn(format="%+.3f"),
                },
            )
        else:
            st.caption("Analysed reviews will appear here for this session.")
elif selected_nav == "Business Insights":
    st.markdown('<div class="panel-heading"><h3 class="panel-title">Business insights</h3></div>', unsafe_allow_html=True)
    render_key_insights(clean_df)

else:
    st.markdown('<div class="panel-heading"><h3 class="panel-title">Analysis settings</h3></div>', unsafe_allow_html=True)
    st.markdown("**Sentiment method**")
    st.write("VADER compound polarity, calculated locally from review text.")
    st.markdown("**Classification thresholds**")
    st.write("Positive: compound score at least +0.05. Neutral: above -0.05 and below +0.05. Negative: compound score at most -0.05.")
    st.caption("Review text is not sent to a paid API. Thresholds are fixed by the current analyzer.")
