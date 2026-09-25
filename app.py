# =========================
# Gold Price Forecasting Dashboard
# Clean Stable Version
# =========================

import json
import joblib
import time
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from tavily import TavilyClient
from groq import Groq


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Gold Price Forecasting Dashboard",
    layout="wide"
)


# =========================
# CSS Styling
# =========================

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: #0b0f19;
    }

    [data-testid="stAppViewContainer"] .main .block-container {
        max-width: 1250px;
        padding-top: 2.2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        padding-bottom: 2.5rem;
        margin: auto;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-title {
        text-align: center;
        font-size: 2.45rem;
        font-weight: 850;
        color: #f8fafc;
        letter-spacing: 0.4px;
        margin-bottom: 0.4rem;
    }

    .main-subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2.4rem;
    }

    .section-heading {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.75rem;
        border-left: 4px solid #38bdf8;
    }

    /* Native Streamlit metric card styling */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #162033 100%);
        border: 1px solid #2f3b52;
        border-radius: 18px;
        padding: 1.05rem 1rem;
        min-height: 135px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 8px 22px rgba(0,0,0,0.18);
        text-align: center;
    }

    [data-testid="stMetricLabel"] {
        justify-content: center;
        text-align: center;
        width: 100%;
    }

    [data-testid="stMetricLabel"] p {
        color: #a8b3c7;
        font-weight: 750;
        font-size: 0.86rem;
        text-align: center;
        width: 100%;
    }

    [data-testid="stMetricValue"] {
        text-align: center;
        width: 100%;
        justify-content: center;
        display: flex;
    }

    [data-testid="stMetricValue"] div {
        color: #f8fafc;
        font-weight: 850;
        font-size: 1.75rem;
        text-align: center;
        white-space: normal;
        word-break: break-word;
    }

    [data-testid="stMetricDelta"] {
        justify-content: center;
        text-align: center;
        width: 100%;
    }

    [data-testid="stMetricDelta"] div {
        justify-content: center;
        text-align: center;
        font-weight: 700;
    }

    div.stButton > button {
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        font-weight: 800;
        border: 1px solid #334155;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    .info-box {
        background: #0f172a;
        border: 1px solid #263244;
        border-left: 5px solid #38bdf8;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        color: #e5e7eb;
        line-height: 1.55;
    }

    .signal-box {
        background: linear-gradient(135deg, #172554 0%, #111827 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        text-align: center;
    }

    .signal-label {
        color: #93c5fd;
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .signal-value {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 850;
    }

    @media (max-width: 900px) {
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .main-title {
            font-size: 1.8rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Header
# =========================

st.markdown(
    """
    <div class="main-title">Gold Price Forecasting & News Risk Dashboard</div>
    <div class="main-subtitle">
        LSTM-based 7-day gold price forecast with live macro-news sentiment context
    </div>
    """,
    unsafe_allow_html=True
)


def section_heading(text):
    st.markdown(
        f'<div class="section-heading">{text}</div>',
        unsafe_allow_html=True
    )


def info_box(title, text):
    st.markdown(
        f"""
        <div class="info-box">
            <b>{title}</b><br>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def signal_box(label, value):
    st.markdown(
        f"""
        <div class="signal-box">
            <div class="signal-label">{label}</div>
            <div class="signal-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# Sidebar
# =========================

st.sidebar.header("API Configuration")

tavily_api_key = st.sidebar.text_input(
    "Tavily API Key",
    type="password"
)

groq_api_key = st.sidebar.text_input(
    "Groq API Key",
    type="password"
)

st.sidebar.info("API keys are used only during this session.")


# =========================
# Load Model Assets
# =========================

@st.cache_resource
def load_model_assets():
    model = load_model("models/gold_market_only_lstm.keras")
    scaler = joblib.load("models/gold_feature_scaler.pkl")
    feature_columns = joblib.load("models/gold_feature_columns.pkl")
    time_steps = joblib.load("models/gold_time_steps.pkl")
    return model, scaler, feature_columns, time_steps


try:
    model, scaler, feature_columns, time_steps = load_model_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()


# =========================
# Data Functions
# =========================

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_gold_data():
    """
    Fetches gold futures data from Yahoo Finance.
    Includes multiple fallback attempts because yfinance can sometimes
    return an empty dataframe on cloud deployments.
    """

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    last_error = None

    download_attempts = [
        {"period": "5y", "interval": "1d"},
        {"period": "2y", "interval": "1d"},
        {"start": "2021-01-01", "interval": "1d"},
    ]

    for params in download_attempts:
        for _ in range(2):
            try:
                gold = yf.download(
                    "GC=F",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                    **params
                )

                if gold is not None and not gold.empty:
                    if isinstance(gold.columns, pd.MultiIndex):
                        gold.columns = gold.columns.get_level_values(0)

                    if all(col in gold.columns for col in required_cols):
                        gold = gold[required_cols].dropna()

                        if len(gold) >= 100:
                            return gold

                time.sleep(1)

            except Exception as e:
                last_error = e
                time.sleep(1)

    # Final fallback using Ticker history
    try:
        ticker = yf.Ticker("GC=F")
        gold = ticker.history(
            period="5y",
            interval="1d",
            auto_adjust=False
        )

        if gold is not None and not gold.empty:
            if isinstance(gold.columns, pd.MultiIndex):
                gold.columns = gold.columns.get_level_values(0)

            if all(col in gold.columns for col in required_cols):
                gold = gold[required_cols].dropna()

                if len(gold) >= 100:
                    return gold

    except Exception as e:
        last_error = e

    st.error(
        "Gold price data could not be fetched from Yahoo Finance right now. "
        "Please refresh the app after a few minutes."
    )

    if last_error is not None:
        st.caption(f"Last data-fetch error: {last_error}")

    st.stop()
    
@st.cache_data(ttl=300, show_spinner=False)
def fetch_gold_intraday_quote():
    """
    Fetch latest available intraday gold futures quote.
    Used only for dashboard market snapshot, not for LSTM prediction.
    """

    attempts = [
        {"period": "5d", "interval": "1h"},
        {"period": "1d", "interval": "15m"},
        {"period": "5d", "interval": "30m"},
    ]

    for params in attempts:
        try:
            intraday = yf.download(
                "GC=F",
                auto_adjust=False,
                progress=False,
                threads=False,
                **params
            )

            if intraday is not None and not intraday.empty:
                if isinstance(intraday.columns, pd.MultiIndex):
                    intraday.columns = intraday.columns.get_level_values(0)

                if "Close" in intraday.columns:
                    intraday = intraday.dropna(subset=["Close"])

                    if not intraday.empty:
                        return {
                            "price": float(intraday["Close"].iloc[-1]),
                            "timestamp": intraday.index[-1]
                        }

        except Exception:
            time.sleep(1)

    return None


def prepare_gold_features(gold_df):
    df = gold_df.copy()

    df["Return"] = df["Close"].pct_change()

    df["MA_7"] = df["Close"].rolling(window=7).mean()
    df["MA_30"] = df["Close"].rolling(window=30).mean()

    df["Volatility_7"] = df["Return"].rolling(window=7).std()
    df["Volatility_30"] = df["Return"].rolling(window=30).std()

    df["Momentum_7"] = df["Close"] - df["Close"].shift(7)
    df["Momentum_30"] = df["Close"] - df["Close"].shift(30)

    df["Volume_MA_7"] = df["Volume"].rolling(window=7).mean()
    df["Volume_MA_30"] = df["Volume"].rolling(window=30).mean()
    df["Volume_Change"] = df["Volume"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    return df


# =========================
# Prediction Functions
# =========================

def classify_gold_risk(predicted_return):
    if predicted_return >= 0.02:
        return "Bullish / High Upside"
    elif predicted_return >= 0.005:
        return "Moderately Bullish"
    elif predicted_return <= -0.02:
        return "Bearish / Downside Risk"
    elif predicted_return <= -0.005:
        return "Moderately Bearish"
    else:
        return "Neutral"


def get_gold_recommendation(risk_signal):
    if risk_signal == "Bullish / High Upside":
        return "Model indicates strong short-term upside; monitor macro news and avoid aggressive short positions."
    elif risk_signal == "Moderately Bullish":
        return "Model indicates mild upside; continue monitoring dollar index, inflation, and rate-related news."
    elif risk_signal == "Bearish / Downside Risk":
        return "Model indicates downside risk; monitor dollar strength, rate expectations, and profit-booking pressure."
    elif risk_signal == "Moderately Bearish":
        return "Model indicates mild downside; avoid overexposure and wait for confirmation."
    else:
        return "Model indicates limited movement; continue normal monitoring."


def predict_latest_gold_price(
    model,
    scaler,
    feature_data,
    feature_columns,
    time_steps
):
    latest_data = feature_data[feature_columns].tail(time_steps)
    latest_scaled = scaler.transform(latest_data)

    latest_seq = latest_scaled.reshape(
        1,
        time_steps,
        len(feature_columns)
    )

    predicted_return = model.predict(latest_seq, verbose=0)[0][0]

    current_price = feature_data["Close"].iloc[-1]
    predicted_future_price = current_price * (1 + predicted_return)

    expected_change = predicted_future_price - current_price
    expected_change_percent = predicted_return * 100

    risk_signal = classify_gold_risk(predicted_return)
    recommendation = get_gold_recommendation(risk_signal)

    return {
        "Current Gold Price": round(float(current_price), 2),
        "Predicted 7-Day Future Gold Price": round(float(predicted_future_price), 2),
        "Expected Change": round(float(expected_change), 2),
        "Expected Change (%)": round(float(expected_change_percent), 4),
        "Risk Signal": risk_signal,
        "Recommendation": recommendation
    }


# =========================
# News Sentiment Functions
# =========================

def fetch_latest_gold_news(tavily_api_key, max_results=10):
    client = TavilyClient(api_key=tavily_api_key)

    query = (
        "latest gold price news inflation interest rates dollar "
        "Federal Reserve safe haven demand geopolitical risk"
    )

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
        include_raw_content=False
    )

    results = response.get("results", [])

    rows = []
    for item in results:
        rows.append({
            "title": item.get("title"),
            "url": item.get("url"),
            "content": item.get("content"),
            "score": item.get("score")
        })

    return pd.DataFrame(rows)


def format_news_for_llm(news_df):
    lines = []

    for i, row in news_df.iterrows():
        title = str(row.get("title", "")).strip()
        content = str(row.get("content", "")).strip()

        lines.append(
            f"{i + 1}. Title: {title}\nSummary: {content}"
        )

    return "\n\n".join(lines)


def analyze_gold_news_with_groq(news_text, groq_api_key):
    client = Groq(api_key=groq_api_key)

    prompt = f"""
You are a financial news analyst for gold price forecasting.

Analyze the following gold-related news headlines and summaries.
Estimate their likely short-term impact on gold prices over the next 7 trading days.

Return exactly ONE valid JSON object.
Do not return multiple alternatives.
Do not include markdown.
Do not include explanation outside JSON.
Do not include trailing text.

Use this exact JSON schema:

{{
  "overall_news_signal": "bullish/bearish/neutral/mixed",
  "gold_sentiment_score": 0.0,
  "inflation_risk_score": 0.0,
  "interest_rate_risk_score": 0.0,
  "dollar_pressure_score": 0.0,
  "safe_haven_demand_score": 0.0,
  "geopolitical_risk_score": 0.0,
  "market_uncertainty_score": 0.0,
  "short_term_gold_impact": "positive/negative/mixed/neutral",
  "confidence_score": 0.0,
  "key_drivers": ["driver 1", "driver 2", "driver 3"],
  "summary": "2-3 sentence summary"
}}

Scoring rules:
- gold_sentiment_score must be between -1 and 1.
- dollar_pressure_score must be between -1 and 1.
- all other numeric scores must be between 0 and 1.
- Positive gold_sentiment_score means bullish for gold.
- Negative gold_sentiment_score means bearish for gold.
- Negative dollar_pressure_score means dollar strength is pressuring gold.

News:
{news_text}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a strict JSON generator. Return exactly one JSON object and nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()


def parse_first_json_object(text):
    text = text.strip()
    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found in Groq output.")

    brace_count = 0

    for i in range(start, len(text)):
        if text[i] == "{":
            brace_count += 1
        elif text[i] == "}":
            brace_count -= 1

            if brace_count == 0:
                json_text = text[start:i + 1]
                return json.loads(json_text)

    raise ValueError("Could not find a complete JSON object.")


def combine_model_and_news_signal(prediction_summary, news_context):
    model_signal = prediction_summary["Risk Signal"]
    news_signal = str(news_context["overall_news_signal"]).lower()
    news_score = float(news_context["gold_sentiment_score"])

    if model_signal in ["Bullish / High Upside", "Moderately Bullish"] and news_score > 0.25:
        final_signal = "Bullish"
    elif model_signal in ["Bearish / Downside Risk", "Moderately Bearish"] and news_score < -0.25:
        final_signal = "Bearish"
    elif model_signal == "Neutral" and news_score < -0.25:
        final_signal = "Cautious Neutral / News Bearish"
    elif model_signal == "Neutral" and news_score > 0.25:
        final_signal = "Cautious Neutral / News Bullish"
    elif news_score < -0.50:
        final_signal = "Bearish News Risk"
    elif news_score > 0.50:
        final_signal = "Bullish News Support"
    else:
        final_signal = "Neutral / Mixed"

    if final_signal == "Bullish":
        recommendation = "Model and news both support upside. Monitor confirmation before taking aggressive positions."
    elif final_signal == "Bearish":
        recommendation = "Model and news both indicate downside pressure. Monitor dollar strength, rates, and support levels."
    elif final_signal == "Cautious Neutral / News Bearish":
        recommendation = (
            "Model forecast is neutral, but news sentiment is bearish. "
            "Avoid aggressive bullish exposure and monitor inflation, Fed, and dollar signals."
        )
    elif final_signal == "Cautious Neutral / News Bullish":
        recommendation = (
            "Model forecast is neutral, but news sentiment supports upside. "
            "Monitor safe-haven demand and dollar weakness."
        )
    elif final_signal == "Bullish News Support":
        recommendation = (
            "News sentiment strongly supports upside, but model confirmation is limited. "
            "Monitor price action before making strong conclusions."
        )
    elif final_signal == "Bearish News Risk":
        recommendation = (
            "News sentiment shows downside risk, but model confirmation is limited. "
            "Monitor dollar strength, yields, and Fed-related updates."
        )
    else:
        recommendation = "Signals are mixed. Continue monitoring price action and macro news."

    return {
        "Model Risk Signal": model_signal,
        "News Signal": news_signal,
        "News Sentiment Score": news_score,
        "Final Combined Signal": final_signal,
        "Key News Drivers": news_context.get("key_drivers", []),
        "News Summary": news_context.get("summary", ""),
        "Final Recommendation": recommendation
    }


# =========================
# Fetch Data
# =========================

gold_raw = fetch_gold_data()

if gold_raw is None or gold_raw.empty:
    st.error("No gold price data was returned. Please refresh the app.")
    st.stop()

gold_intraday_quote = fetch_gold_intraday_quote()

gold_features = prepare_gold_features(gold_raw)

if gold_features is None or gold_features.empty or len(gold_features) < time_steps:
    st.error(
        "Not enough gold price data available to create the LSTM input sequence. "
        "Please refresh the app after a few minutes."
    )
    st.stop()


# =========================
# Market Snapshot
# =========================

section_heading("Market Snapshot")

latest_daily_close = float(gold_raw["Close"].iloc[-1])
previous_daily_close = float(gold_raw["Close"].iloc[-2])
latest_daily_date = gold_raw.index[-1].date()

if gold_intraday_quote is not None:
    display_price = gold_intraday_quote["price"]
    raw_timestamp = gold_intraday_quote["timestamp"]

    try:
        if raw_timestamp.tzinfo is not None:
            display_timestamp = raw_timestamp.tz_convert("Asia/Kolkata")
        else:
            display_timestamp = raw_timestamp
    except Exception:
        display_timestamp = raw_timestamp

    market_time_text = display_timestamp.strftime("%Y-%m-%d %H:%M")
else:
    display_price = latest_daily_close
    market_time_text = str(latest_daily_date)

price_change = display_price - previous_daily_close
price_change_pct = (price_change / previous_daily_close) * 100

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.metric(
        "Current Gold Price",
        f"${display_price:,.2f}",
        f"{price_change_pct:.2f}% vs previous close"
    )

with col2:
    st.metric(
        "Market Data Time",
        market_time_text
    )

with col3:
    st.metric(
        "Model Daily Candle",
        str(latest_daily_date)
    )

# =========================
# Gold Price Trend
# =========================

section_heading("Gold Price Trend")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=gold_raw.index,
        y=gold_raw["Close"],
        mode="lines",
        name="Gold Close Price",
        line=dict(width=2)
    )
)

fig.update_layout(
    title="Gold Futures Closing Price - Last 5 Years",
    xaxis_title="Date",
    yaxis_title="Price",
    height=450,
    template="plotly_dark",
    margin=dict(l=30, r=30, t=50, b=30),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)


# =========================
# Forecast
# =========================

section_heading("Forecast")

run_prediction = st.button("Predict Next 7 Days")

if run_prediction:
    prediction_summary = predict_latest_gold_price(
        model=model,
        scaler=scaler,
        feature_data=gold_features,
        feature_columns=feature_columns,
        time_steps=time_steps
    )

    st.session_state["prediction_summary"] = prediction_summary

    if "combined_summary" in st.session_state:
        del st.session_state["combined_summary"]
    if "latest_news" in st.session_state:
        del st.session_state["latest_news"]


if "prediction_summary" in st.session_state:
    prediction_summary = st.session_state["prediction_summary"]

    st.markdown("#### Latest 7-Day Forecast")

    col1, col2, col3, col4 = st.columns(4, gap="large")

    with col1:
        st.metric(
            "Model input close",
            f"${prediction_summary['Current Gold Price']:,.2f}"
        )

    with col2:
        st.metric(
            "Predicted 7-Day Price",
            f"${prediction_summary['Predicted 7-Day Future Gold Price']:,.2f}"
        )

    with col3:
        st.metric(
            "Expected Change",
            f"${prediction_summary['Expected Change']:,.2f}",
            f"{prediction_summary['Expected Change (%)']}%"
        )

    with col4:
        st.metric(
            "Model Signal",
            prediction_summary["Risk Signal"]
        )

    info_box(
        "Model Recommendation",
        prediction_summary["Recommendation"]
    )


# =========================
# Live News Sentiment
# =========================

section_heading("Live News Sentiment Context")

if "prediction_summary" not in st.session_state:
    st.warning("Run the 7-day prediction first to enable combined news analysis.")

elif not tavily_api_key or not groq_api_key:
    st.warning("Enter Tavily and Groq API keys in the sidebar to enable live news sentiment analysis.")

else:
    run_news_analysis = st.button("Analyze Latest Gold News")

    if run_news_analysis:
        with st.spinner("Fetching latest gold news and analyzing sentiment..."):
            try:
                latest_news = fetch_latest_gold_news(
                    tavily_api_key=tavily_api_key,
                    max_results=10
                )

                if latest_news.empty:
                    st.warning("No latest news articles found.")
                    st.stop()

                news_text = format_news_for_llm(latest_news)

                groq_output = analyze_gold_news_with_groq(
                    news_text=news_text,
                    groq_api_key=groq_api_key
                )

                news_context = parse_first_json_object(groq_output)

                combined_summary = combine_model_and_news_signal(
                    prediction_summary=st.session_state["prediction_summary"],
                    news_context=news_context
                )

                st.session_state["latest_news"] = latest_news
                st.session_state["combined_summary"] = combined_summary

            except Exception as e:
                st.error(f"News analysis failed: {e}")


if "combined_summary" in st.session_state:
    combined_summary = st.session_state["combined_summary"]

    st.success("News sentiment analysis completed.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.metric(
            "News Signal",
            combined_summary["News Signal"].title()
        )

    with col2:
        st.metric(
            "News Sentiment Score",
            round(combined_summary["News Sentiment Score"], 3)
        )

    signal_box(
        "Final Combined Signal",
        combined_summary["Final Combined Signal"]
    )

    st.write("**Key News Drivers:**")
    st.write(", ".join(combined_summary["Key News Drivers"]))

    st.write("**News Summary:**")
    st.write(combined_summary["News Summary"])

    info_box(
        "Final Recommendation",
        combined_summary["Final Recommendation"]
    )

    if "latest_news" in st.session_state:
        with st.expander("View fetched news articles"):
            latest_news = st.session_state["latest_news"]

            st.dataframe(
                latest_news[["title", "url", "score"]],
                use_container_width=True,
                column_config={
                    "title": st.column_config.TextColumn(
                        "Headline",
                        width="large"
                    ),
                    "url": st.column_config.LinkColumn(
                        "Article Link",
                        display_text="Open article"
                    ),
                    "score": st.column_config.NumberColumn(
                        "Relevance Score",
                        format="%.4f"
                    )
                },
                hide_index=True
            )


# =========================
# Model Comparison
# =========================

with st.expander("Model Performance Comparison"):
    try:
        comparison_df = pd.read_csv("outputs/gold_model_comparison.csv")

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:
        st.warning(f"Could not load model comparison file: {e}")


# =========================
# About
# =========================

with st.expander("About this project"):
    st.write(
        """
        This dashboard uses a return-based LSTM model trained on historical gold futures data
        to forecast the 7-day future gold price.

        It also uses live news search and LLM-based sentiment analysis to provide macroeconomic
        context around the model forecast. The final recommendation combines the model output
        with live news sentiment.
        """
    )


# =========================
# Disclaimer
# =========================

st.caption(
    "Disclaimer: This dashboard is for educational and analytical purposes only. "
    "It is not financial advice."
)