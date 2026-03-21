import os
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from src.features import log_returns, rolling_volatility, rsi
from src.models import decode_states, train_hmm
from src.utils import load_config

st.set_page_config(page_title="market regime dashboard", page_icon="chart_with_upwards_trend", layout="wide")

CONFIG_PATH = Path("config/config.yaml")
DEFAULT_MODEL_PATH = Path("models/hmm_model.pkl")
DEFAULT_COLORS = ["#c26b2d", "#5a7da0", "#3c8c66", "#9b5d73", "#728c2f", "#b38332"]


def apply_dark_theme():
    st.markdown(
        """
        <style>
        :root {
            --paper: #0b1220;
            --panel: #121a2b;
            --panel-soft: #172133;
            --ink: #e6edf7;
            --muted: #9dadc5;
            --line: #29364d;
            --accent: #f2a65a;
            --accent-soft: #3b2a1d;
            --navy: #8fb8ff;
            --sidebar: #0f1726;
            --success: #7ed0a7;
            --warning-bg: #3a2d14;
            --warning-line: #8f6a2c;
            --warning-ink: #f8df9a;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(242, 166, 90, 0.12), transparent 24%),
                radial-gradient(circle at top right, rgba(84, 135, 255, 0.10), transparent 22%),
                linear-gradient(180deg, #0b1220 0%, #0f1726 100%);
            color: var(--ink);
        }
        html, body, [class*="css"] {
            color: var(--ink);
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.4rem;
            max-width: 1400px;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1726 0%, #111b2d 100%);
            border-right: 1px solid var(--line);
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] * {
            color: var(--ink);
        }
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {
            color: inherit;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] *,
        [data-testid="stText"],
        [data-testid="stText"] *,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {
            color: var(--ink) !important;
        }
        [data-testid="stCaptionContainer"] {
            opacity: 0.85;
        }
        [data-testid="stHeading"] * {
            color: var(--ink) !important;
        }
        .stTextInput input,
        .stNumberInput input,
        .stDateInput input,
        .stSelectbox div[data-baseweb="select"] > div,
        .stMultiSelect div[data-baseweb="select"] > div {
            background: #121a2b !important;
            color: var(--ink) !important;
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
        }
        .stDateInput button, .stSelectbox svg {
            color: var(--muted) !important;
            fill: var(--muted) !important;
        }
        .stSlider [data-baseweb="slider"] {
            padding-top: 0.4rem;
        }
        .stSlider [role="slider"] {
            background: var(--accent) !important;
            border-color: var(--accent) !important;
        }
        .stSlider [data-testid="stTickBar"] {
            background: #35425a !important;
        }
        .stButton > button {
            background: linear-gradient(135deg, #f2a65a 0%, #d58436 100%);
            color: #101827;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            font-weight: 700;
            box-shadow: 0 10px 22px rgba(242, 166, 90, 0.18);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #f6b777 0%, #dc8f47 100%);
            color: #101827;
        }
        div[data-testid="stMetric"] {
            background: rgba(18, 26, 43, 0.96);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.95rem 1rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.22);
        }
        div[data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-weight: 600;
        }
        div[data-testid="stMetricValue"] {
            color: var(--ink) !important;
        }
        div[data-testid="stMetricDelta"] {
            color: var(--accent) !important;
        }
        div[data-testid="stMetric"] * {
            text-shadow: none !important;
        }
        .story-card {
            background: rgba(18, 26, 43, 0.96);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
            height: 100%;
        }
        .story-card h4 {
            margin: 0 0 0.35rem 0;
            color: var(--accent);
            font-size: 0.92rem;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }
        .story-card p {
            margin: 0;
            color: #d6e0ee;
            line-height: 1.5;
        }
        .regime-strip {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            flex-wrap: wrap;
            background: rgba(18, 26, 43, 0.96);
            border: 1px solid var(--line);
            border-left: 6px solid var(--accent);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
            margin-top: 0.35rem;
            margin-bottom: 0.4rem;
        }
        .regime-strip strong {
            color: var(--ink);
            font-size: 1.02rem;
        }
        .regime-meta {
            color: var(--muted);
            font-weight: 600;
        }
        .section-note {
            color: var(--muted);
            margin-bottom: 0.85rem;
            font-size: 0.98rem;
        }
        .element-container {
            color: var(--ink);
        }
        div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--warning-line);
            background: linear-gradient(135deg, #2d2210 0%, #3b2d14 100%);
            color: var(--warning-ink);
        }
        div[data-testid="stAlert"] * {
            color: var(--warning-ink) !important;
        }
        div[data-baseweb="tab-list"] {
            gap: 1rem;
            border-bottom: 1px solid var(--line);
        }
        button[data-baseweb="tab"] {
            background: transparent !important;
            color: var(--muted) !important;
            border: none !important;
            font-weight: 700 !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }
        button[data-baseweb="tab"] p,
        button[data-baseweb="tab"] span {
            color: inherit !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--accent) !important;
        }
        [data-testid="stTabPanel"] {
            padding-top: 1rem;
        }
        div[data-testid="stPlotlyChart"] {
            background: rgba(18, 26, 43, 0.96);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.35rem 0.35rem 0.1rem 0.35rem;
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stDataFrame"] {
            background: rgba(18, 26, 43, 0.96);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.3rem;
        }
        [data-testid="stExpander"] {
            background: rgba(18, 26, 43, 0.92);
            border: 1px solid var(--line);
            border-radius: 16px;
        }
        [data-testid="stExpander"] * {
            color: var(--ink) !important;
        }
        [data-testid="stCheckbox"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stDateInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSlider"] label {
            color: var(--muted) !important;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def app_badge(label: str, color: str, text_color: str = "#ffffff") -> str:
    return (
        f"<span style='background:{color};color:{text_color};padding:0.5rem 1rem;"
        "border-radius:999px;font-weight:700;display:inline-block;"
        "border:1px solid rgba(255,255,255,0.18);box-shadow:0 6px 14px rgba(23,50,77,0.12);'>"
        f"{label}</span>"
    )


def render_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="story-card">
            <h4>{title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_regime_strip(label: str, color: str, probability: float):
    st.markdown(
        f"""
        <div class="regime-strip">
            <strong>current regime</strong>
            {app_badge(label, color, "#ffffff")}
            <span class="regime-meta">posterior probability: {probability:.1%}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(probabilities, 1e-12, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1)
    max_entropy = np.log(probs.shape[1]) if probs.shape[1] > 1 else 1.0
    if np.isclose(max_entropy, 0.0):
        return np.zeros(len(probs))
    return entropy / max_entropy


def top_two_gap(probabilities: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probabilities, axis=1)
    if sorted_probs.shape[1] == 1:
        return np.ones(len(sorted_probs))
    return sorted_probs[:, -1] - sorted_probs[:, -2]


@st.cache_data(show_spinner=False)
def load_project_config(config_path: str) -> dict:
    if os.path.exists(config_path):
        return load_config(config_path)
    return {}


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_price_data(ticker: str, start_date: date, end_date: date) -> pd.Series:
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if data.empty or "Close" not in data:
        raise ValueError(f"No price data returned for {ticker} between {start_date} and {end_date}.")

    prices = data["Close"]
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]

    prices = prices.sort_index().ffill().dropna().copy()
    if prices.empty:
        raise ValueError(f"Price series for {ticker} is empty after cleaning.")

    prices.name = "Close"
    return prices


@st.cache_resource(show_spinner=False)
def load_saved_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def coerce_model_object(loaded_obj):
    if loaded_obj is None:
        return None, {}

    if hasattr(loaded_obj, "predict") and hasattr(loaded_obj, "predict_proba"):
        return loaded_obj, {}

    if isinstance(loaded_obj, dict):
        model = loaded_obj.get("model")
        metadata = {key: value for key, value in loaded_obj.items() if key != "model"}
        if model is not None:
            return model, metadata

    raise ValueError("Unsupported model format in the saved model file.")


def get_feature_settings(config: dict):
    feature_config = config.get("features", {})
    feature_settings = {
        "include_returns": feature_config.get("returns", True),
        "volatility_window": feature_config.get("rolling_volatility_window"),
        "rsi_window": feature_config.get("rsi_window"),
    }
    feature_columns = []
    if feature_settings["include_returns"]:
        feature_columns.append("return")
    if feature_settings["volatility_window"]:
        feature_columns.append("volatility")
    if feature_settings["rsi_window"]:
        feature_columns.append("rsi")
    return feature_settings, feature_columns


def build_feature_frame(prices: pd.Series, feature_settings: dict) -> pd.DataFrame:
    feature_frames = {}
    cleaned_prices = prices.sort_index().ffill().dropna()

    returns = log_returns(cleaned_prices).rename("return")
    if feature_settings["include_returns"]:
        feature_frames["return"] = returns

    volatility_window = feature_settings["volatility_window"]
    if volatility_window:
        feature_frames["volatility"] = rolling_volatility(returns, window=volatility_window).rename("volatility")

    rsi_window = feature_settings["rsi_window"]
    if rsi_window:
        feature_frames["rsi"] = rsi(cleaned_prices, window=rsi_window).rename("rsi").reindex(returns.index)

    if not feature_frames:
        raise ValueError("At least one feature must be enabled.")

    feature_df = pd.concat(feature_frames.values(), axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if feature_df.empty:
        raise ValueError("Not enough clean history to compute the selected features.")

    return feature_df


def resolve_model_feature_columns(metadata: dict, fallback_columns: list[str]) -> list[str]:
    metadata_columns = metadata.get("feature_columns")
    if metadata_columns:
        return list(metadata_columns)
    return fallback_columns


def validate_model_compatibility(model, feature_df: pd.DataFrame, feature_columns: list[str]):
    missing_columns = [column for column in feature_columns if column not in feature_df.columns]
    if missing_columns:
        raise ValueError(f"Feature data is missing columns required by the model: {missing_columns}")

    if hasattr(model, "means_") and model.means_.shape[1] != len(feature_columns):
        raise ValueError(
            f"Saved model expects {model.means_.shape[1]} features, but the app prepared {len(feature_columns)}."
        )


def train_live_model(feature_df: pd.DataFrame, feature_columns: list[str], n_states: int, config: dict):
    model_config = config.get("model", {})
    return train_hmm(
        feature_df[feature_columns].values,
        n_states=n_states,
        covariance_type=model_config.get("covariance_type", "full"),
        n_iter=model_config.get("n_iter", 1000),
        random_state=model_config.get("random_state", 42),
    )


def choose_model(feature_df: pd.DataFrame, feature_columns: list[str], selected_states: int, retrain: bool, model_path: str, config: dict):
    loaded_model_obj = load_saved_model(model_path)
    loaded_model, metadata = coerce_model_object(loaded_model_obj)

    if retrain:
        with st.spinner("retraining hmm on the latest data..."):
            model = train_live_model(feature_df, feature_columns, selected_states, config)
        return model, {"source": "retrained in app", "feature_columns": feature_columns}

    if loaded_model is None:
        st.warning(f"saved model not found at `{model_path}`. training a fresh model for this session.")
        model = train_live_model(feature_df, feature_columns, selected_states, config)
        return model, {"source": "auto-trained fallback", "feature_columns": feature_columns}

    metadata = metadata or {}
    model_feature_columns = resolve_model_feature_columns(metadata, feature_columns)

    try:
        validate_model_compatibility(loaded_model, feature_df, model_feature_columns)
    except ValueError as exc:
        st.warning(f"saved model is not compatible with the current feature setup: {exc} retraining instead.")
        model = train_live_model(feature_df, feature_columns, selected_states, config)
        return model, {"source": "retrained due to compatibility mismatch", "feature_columns": feature_columns}

    if getattr(loaded_model, "n_components", selected_states) != selected_states:
        st.info(
            f"saved model has {loaded_model.n_components} states, so the app is training a fresh {selected_states}-state model for this session."
        )
        model = train_live_model(feature_df, feature_columns, selected_states, config)
        return model, {"source": f"auto-trained {selected_states}-state session model", "feature_columns": feature_columns}

    metadata["source"] = metadata.get("source", f"loaded from {model_path}")
    metadata["feature_columns"] = model_feature_columns
    return loaded_model, metadata


def decode_with_model(model, feature_df: pd.DataFrame, feature_columns: list[str]):
    x = feature_df[feature_columns].values
    states = decode_states(model, x)
    probabilities = model.predict_proba(x)
    return states, probabilities


def average_regime_duration(states: pd.Series, regime_id: int) -> float:
    durations = []
    run_length = 0
    for value in states:
        if value == regime_id:
            run_length += 1
        elif run_length > 0:
            durations.append(run_length)
            run_length = 0
    if run_length > 0:
        durations.append(run_length)
    return float(np.mean(durations)) if durations else 0.0


def describe_level(rank: int, total: int, low_label: str, mid_label: str, high_label: str) -> str:
    if total <= 1:
        return mid_label
    fraction = rank / (total - 1)
    if fraction <= 0.33:
        return low_label
    if fraction >= 0.67:
        return high_label
    return mid_label


def market_return_name(return_rank: int, total_states: int) -> str:
    if total_states <= 2:
        names = ["Bearish", "Bullish"]
    elif total_states == 3:
        names = ["Bearish", "Neutral", "Bullish"]
    else:
        names = ["Bearish", "Neutral", "Constructive", "Bullish"]
    return names[min(return_rank, len(names) - 1)]


def market_volatility_name(volatility_rank: int, total_states: int) -> str:
    if total_states <= 2:
        return "Volatile" if volatility_rank == total_states - 1 else "Calm"

    return describe_level(volatility_rank, total_states, "Calm", "Moderate", "Volatile")


def market_oriented_regime_name(return_rank: int, volatility_rank: int, total_states: int) -> str:
    return_level = market_return_name(return_rank, total_states)
    volatility_level = market_volatility_name(volatility_rank, total_states)
    return f"{return_level} {volatility_level}"


def infer_regime_descriptions(feature_df: pd.DataFrame, states: np.ndarray, color_sequence: list[str]):
    summary = feature_df.copy()
    summary["state"] = states
    state_stats = summary.groupby("state").agg(
        mean_return=("return", "mean"),
        mean_volatility=("volatility", "mean") if "volatility" in summary.columns else ("return", "std"),
    )

    return_order = state_stats["mean_return"].sort_values().index.tolist()
    volatility_order = state_stats["mean_volatility"].sort_values().index.tolist()
    return_rank = {state: rank for rank, state in enumerate(return_order)}
    volatility_rank = {state: rank for rank, state in enumerate(volatility_order)}

    label_map = {}
    color_map = {}
    for color_rank, state in enumerate(return_order):
        label_map[state] = market_oriented_regime_name(
            return_rank[state],
            volatility_rank[state],
            len(return_order),
        )
        color_map[state] = color_sequence[color_rank % len(color_sequence)]

    duplicate_counts = pd.Series(label_map.values()).value_counts()
    for state, label in list(label_map.items()):
        if duplicate_counts[label] > 1:
            label_map[state] = f"{label} (State {state})"

    return label_map, color_map, state_stats.reset_index()


def regime_statistics(feature_df: pd.DataFrame, states: np.ndarray, label_map: dict) -> pd.DataFrame:
    stats_df = feature_df.copy()
    stats_df["state"] = states
    total_days = len(stats_df)

    rows = []
    for state in sorted(stats_df["state"].unique()):
        regime_slice = stats_df[stats_df["state"] == state]
        row = {
            "Regime": label_map.get(state, f"State {state}"),
            "State": state,
            "Mean Return (%)": regime_slice["return"].mean() if "return" in regime_slice else np.nan,
            "Share of Days (%)": 100 * len(regime_slice) / total_days,
            "Avg Duration (days)": average_regime_duration(stats_df["state"], state),
        }
        if "volatility" in regime_slice:
            row["Mean Volatility"] = regime_slice["volatility"].mean()
        if "rsi" in regime_slice:
            row["Avg RSI"] = regime_slice["rsi"].mean()
        rows.append(row)

    ordered_columns = [
        column
        for column in [
            "Regime",
            "State",
            "Mean Return (%)",
            "Mean Volatility",
            "Avg RSI",
            "Share of Days (%)",
            "Avg Duration (days)",
        ]
        if column in rows[0]
    ]
    return pd.DataFrame(rows)[ordered_columns].sort_values("State").reset_index(drop=True)


def add_regime_backgrounds(fig: go.Figure, frame: pd.DataFrame, color_map: dict):
    shapes = []
    dates = frame.index.to_list()
    states = frame["state"].to_list()
    if not dates:
        return fig

    start_idx = 0
    for idx in range(1, len(states) + 1):
        changed = idx == len(states) or states[idx] != states[start_idx]
        if changed:
            state = states[start_idx]
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "paper",
                    "x0": dates[start_idx],
                    "x1": dates[idx - 1],
                    "y0": 0,
                    "y1": 1,
                    "fillcolor": color_map[state],
                    "opacity": 0.12,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )
            start_idx = idx

    fig.update_layout(shapes=shapes)
    return fig


def price_chart(regime_frame: pd.DataFrame, color_map: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=regime_frame.index,
            y=regime_frame["Close"],
            mode="lines",
            name="price",
            line={"color": "#dbe7ff", "width": 2.7},
            hovertemplate="%{x|%d %b %Y}<br>price: %{y:.2f}<extra></extra>",
        )
    )
    fig = add_regime_backgrounds(fig, regime_frame, color_map)
    fig.update_layout(
        title="price with historical regimes",
        xaxis_title="date",
        yaxis_title="adjusted close",
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor="rgba(18, 26, 43, 1)",
        plot_bgcolor="rgba(18, 26, 43, 1)",
        font={"color": "#dbe7ff"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        showgrid=False,
        color="#c1cee3",
        rangeslider={"bgcolor": "#11192a", "bordercolor": "#2a3750", "thickness": 0.08},
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(193, 206, 227, 0.10)", color="#c1cee3")
    return fig


def probability_chart(display_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=display_df.index,
            y=display_df["Posterior Probability"],
            mode="lines",
            name="posterior probability",
            line={"color": "#f2a65a", "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(242, 166, 90, 0.18)",
            hovertemplate="%{x|%d %b %Y}<br>posterior probability: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_df.index,
            y=display_df["Top Two Gap"],
            mode="lines",
            name="top-two gap",
            line={"color": "#8fb8ff", "width": 1.8, "dash": "dot"},
            hovertemplate="%{x|%d %b %Y}<br>top-two gap: %{y:.1%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=display_df.index,
            y=display_df["Uncertainty"],
            mode="lines",
            name="uncertainty",
            line={"color": "#7ed0a7", "width": 1.6},
            hovertemplate="%{x|%d %b %Y}<br>uncertainty: %{y:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="posterior probability and uncertainty",
        xaxis_title="date",
        yaxis_title="value",
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor="rgba(18, 26, 43, 1)",
        plot_bgcolor="rgba(18, 26, 43, 1)",
        font={"color": "#dbe7ff"},
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor="rgba(193, 206, 227, 0.10)", color="#c1cee3")
    fig.update_xaxes(showgrid=False, color="#c1cee3")
    return fig


def regime_scatter(feature_df: pd.DataFrame, color_map: dict, label_map: dict) -> go.Figure:
    if "return" not in feature_df.columns or "volatility" not in feature_df.columns:
        raise ValueError("Scatter plot needs both return and volatility features.")

    color_discrete_map = {label_map[state]: color_map[state] for state in sorted(color_map)}
    fig = px.scatter(
        feature_df,
        x="return",
        y="volatility",
        color="Regime",
        color_discrete_map=color_discrete_map,
        hover_data={column: ":.2f" for column in ["rsi", "return", "volatility"] if column in feature_df.columns},
        title="return vs volatility by regime",
        template="plotly_dark",
    )
    fig.update_traces(marker={"size": 9, "opacity": 0.82, "line": {"width": 0.4, "color": "#0f1726"}})
    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="rgba(18, 26, 43, 1)",
        plot_bgcolor="rgba(18, 26, 43, 1)",
        font={"color": "#dbe7ff"},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(193, 206, 227, 0.10)", zeroline=True, zerolinecolor="#46556f", color="#c1cee3")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(193, 206, 227, 0.10)", zeroline=True, zerolinecolor="#46556f", color="#c1cee3")
    return fig


def main():
    apply_dark_theme()
    config = load_project_config(str(CONFIG_PATH))
    feature_settings, _ = get_feature_settings(config)
    default_ticker = config.get("ticker", "RELIANCE.NS")
    model_config = config.get("model", {})
    plot_config = config.get("plot", {})
    color_sequence = plot_config.get("colors") or DEFAULT_COLORS
    default_start = pd.to_datetime(config.get("start_date", "2020-01-01")).date()
    default_end = pd.to_datetime(config.get("end_date", date.today())).date()
    default_state_count = model_config.get("n_states", 3)

    st.title("indian stock market regime dashboard")
    st.caption(
        "a simple visual summary of how returns, volatility and momentum can be used to describe market regimes in indian stocks."
    )

    with st.sidebar:
        st.header("controls")
        ticker = st.text_input("yahoo finance ticker", value=default_ticker).strip().upper()
        start_date = st.date_input("start date", value=default_start)
        end_date = st.date_input("end date", value=default_end)
        chart_years = st.slider("years visible in price chart", min_value=1, max_value=5, value=2)
        state_options = [2, 3, 4]
        state_index = state_options.index(default_state_count) if default_state_count in state_options else 1
        selected_states = st.selectbox("number of hmm states", options=state_options, index=state_index)
        retrain = st.checkbox("retrain model live with selected states", value=False)
        model_path = st.text_input("model path", value=str(DEFAULT_MODEL_PATH))

        st.markdown("### feature setup")
        include_returns = st.checkbox("use returns", value=feature_settings["include_returns"], disabled=True)
        volatility_window = st.number_input(
            "rolling volatility window",
            min_value=2,
            max_value=252,
            value=int(feature_settings["volatility_window"] or 20),
        )
        rsi_window = st.number_input(
            "rsi window",
            min_value=2,
            max_value=252,
            value=int(feature_settings["rsi_window"] or 14),
        )
        refresh = st.button("refresh analysis", type="primary")

        st.markdown("---")
        st.markdown(
            "this is a concept-first dashboard. the aim is to explain regime behaviour clearly, not to present a live trading system."
        )

    if start_date >= end_date:
        st.error("start date must be earlier than end date.")
        st.stop()

    active_feature_settings = {
        "include_returns": include_returns,
        "volatility_window": int(volatility_window),
        "rsi_window": int(rsi_window),
    }
    active_feature_columns = [
        column
        for column, enabled in [
            ("return", active_feature_settings["include_returns"]),
            ("volatility", bool(active_feature_settings["volatility_window"])),
            ("rsi", bool(active_feature_settings["rsi_window"])),
        ]
        if enabled
    ]

    if refresh:
        fetch_price_data.clear()
        load_saved_model.clear()
        load_project_config.clear()

    try:
        prices = fetch_price_data(ticker, start_date, end_date)
        feature_df = build_feature_frame(prices, active_feature_settings)
    except Exception as exc:
        st.error(f"unable to prepare data for {ticker}: {exc}")
        st.stop()

    if len(feature_df) < max(60, selected_states * 15):
        st.error("not enough usable history after cleaning to run a stable hmm with the selected settings.")
        st.stop()

    try:
        model, model_metadata = choose_model(
            feature_df,
            active_feature_columns,
            selected_states,
            retrain,
            model_path,
            config,
        )
        model_feature_columns = resolve_model_feature_columns(model_metadata, active_feature_columns)
        states, probabilities = decode_with_model(model, feature_df, model_feature_columns)
    except Exception as exc:
        st.error(f"model step failed: {exc}")
        st.stop()

    display_df = feature_df.copy()
    display_df["state"] = states
    label_map, color_map, state_stat_frame = infer_regime_descriptions(display_df, states, color_sequence)
    display_df["Regime"] = display_df["state"].map(label_map)
    display_df["Max Probability"] = probabilities.max(axis=1)
    display_df["Posterior Probability"] = [probabilities[idx, state] for idx, state in enumerate(states)]
    display_df["Top Two Gap"] = top_two_gap(probabilities)
    display_df["Uncertainty"] = normalized_entropy(probabilities)
    display_df["Close"] = prices.reindex(display_df.index)

    latest_row = display_df.iloc[-1]
    current_state = int(latest_row["state"])
    current_probability = float(probabilities[-1, current_state])
    current_gap = float(display_df.iloc[-1]["Top Two Gap"])
    current_uncertainty = float(display_df.iloc[-1]["Uncertainty"])

    top1, top2, top3, top4, top5 = st.columns(5)
    top1.metric("ticker", ticker)
    top2.metric("latest close", f"{latest_row['Close']:.2f}")
    top3.metric("model states", getattr(model, "n_components", selected_states))
    top4.metric("usable observations", f"{len(display_df):,}")
    top5.metric("top-two gap", f"{current_gap:.1%}")

    render_regime_strip(label_map[current_state], color_map[current_state], current_probability)

    source_text = model_metadata.get("source", "loaded model")
    st.caption(
        f"model source: {source_text} | features used: {', '.join(model_feature_columns)} | "
        f"vol window: {active_feature_settings['volatility_window']} | rsi window: {active_feature_settings['rsi_window']} | "
        f"uncertainty: {current_uncertainty:.1%}"
    )

    info_col1, info_col2 = st.columns(2)
    with info_col1:
        render_card(
            "what this dashboard is showing",
            "it fetches price data, builds returns, rolling volatility and rsi, and then uses the hmm to map that behaviour into recurring market regimes.",
        )
    with info_col2:
        latest_return = latest_row["return"] if "return" in latest_row else np.nan
        latest_vol = latest_row["volatility"] if "volatility" in latest_row else np.nan
        render_card(
            "latest market reading",
            f"latest return is {latest_return:.2f}% and rolling volatility is {latest_vol:.2f}. posterior probability is {current_probability:.1%}, while uncertainty is {current_uncertainty:.1%}.",
        )

    recent_cutoff = display_df.index.max() - pd.DateOffset(years=chart_years)
    recent_frame = display_df.loc[display_df.index >= recent_cutoff, ["Close", "state"]].copy()

    tab1, tab2, tab3 = st.tabs(["market view", "feature view", "stats and flow"])

    with tab1:
        st.markdown(
            '<div class="section-note">this is the main market story: price with regime shading on the left, and model confidence on the right.</div>',
            unsafe_allow_html=True,
        )
        chart_col, prob_col = st.columns([1.7, 1.0])
        with chart_col:
            st.plotly_chart(price_chart(recent_frame, color_map), use_container_width=True)
        with prob_col:
            probability_frame = display_df.loc[display_df.index >= recent_cutoff].copy()
            st.plotly_chart(probability_chart(probability_frame), use_container_width=True)

    with tab2:
        st.markdown(
            '<div class="section-note">this section connects the regime call back to the feature space so the model output feels visible and intuitive.</div>',
            unsafe_allow_html=True,
        )
        try:
            st.plotly_chart(regime_scatter(display_df, color_map, label_map), use_container_width=True)
        except ValueError as exc:
            st.info(str(exc))

        with st.expander("latest feature snapshot"):
            snapshot_columns = [
                column
                for column in ["Close", "return", "volatility", "rsi", "Regime", "Posterior Probability", "Top Two Gap", "Uncertainty"]
                if column in display_df.columns
            ]
            snapshot = display_df[snapshot_columns].tail(12).copy()
            for column in ["Posterior Probability", "Top Two Gap", "Uncertainty"]:
                if column in snapshot.columns:
                    snapshot[column] = snapshot[column].map(lambda value: f"{value:.1%}")
            st.dataframe(snapshot, use_container_width=True)

    with tab3:
        st.markdown(
            '<div class="section-note">this section summarises what each regime has looked like in the selected period and what the fitted state statistics are saying.</div>',
            unsafe_allow_html=True,
        )
        stats_df = regime_statistics(display_df, states, label_map)
        st.subheader("regime statistics")
        st.dataframe(
            stats_df.style.format(
                {
                    column: fmt
                    for column, fmt in {
                        "Mean Return (%)": "{:.2f}",
                        "Mean Volatility": "{:.2f}",
                        "Avg RSI": "{:.1f}",
                        "Share of Days (%)": "{:.1f}",
                        "Avg Duration (days)": "{:.1f}",
                    }.items()
                    if column in stats_df.columns
                }
            ),
            use_container_width=True,
        )

        st.subheader("model diagnostics")
        diagnostics = state_stat_frame.copy()
        diagnostics["Regime"] = diagnostics["state"].map(label_map)
        st.dataframe(diagnostics, use_container_width=True)

        st.subheader("how to read this")
        st.markdown(
            "- **returns** show day-to-day direction.\n"
            "- **volatility** gives the recent instability of price action.\n"
            "- **rsi** adds a simple momentum lens.\n"
            "- the hmm groups similar combinations of those features into recurring market states.\n"
            "- **posterior probability** shows how much probability mass the model assigns to the selected state.\n"
            "- **top-two gap** shows how far ahead the chosen state is from the runner-up.\n"
            "- **uncertainty** is based on normalized entropy, so higher values mean the model is less decisive."
        )


if __name__ == "__main__":
    main()
