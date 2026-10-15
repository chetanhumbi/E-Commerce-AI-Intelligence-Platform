import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="E-Commerce AI | Intelligence Platform", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"

CSS = """
<style>
:root {color-scheme: dark;}
.stApp {background: radial-gradient(circle at 85% 0%, #17204a 0%, #0b1020 39%, #080d19 100%); color: #eaf0ff;}
[data-testid="stSidebar"] {background: #0d1428; border-right: 1px solid #26304d;}
[data-testid="stSidebar"] * {color: #e7edff;}
/* Hide the default Streamlit header, Deploy control, and top decoration. */
header[data-testid="stHeader"] {display: none !important; height: 0 !important;}
[data-testid="stToolbar"], [data-testid="stDecoration"], .stDeployButton {display: none !important;}
/* Remove the blank space normally reserved for the header. */
[data-testid="stAppViewContainer"] > .main {padding-top: 0 !important;}
.block-container {padding-top: 1.2rem !important; padding-bottom: 3rem; max-width: 1400px;}
h1,h2,h3 {color: #f5f7ff !important; letter-spacing: -.025em;}
p, label, [data-testid="stCaptionContainer"] {color: #b6c3dd;}
[data-testid="stMetric"] {background: linear-gradient(135deg,#18223e,#111a30); padding: 1.15rem; border: 1px solid #2a385c; border-radius: 16px;}
[data-testid="stMetricLabel"] {color: #aebfe2;}
[data-testid="stMetricValue"] {color: #f2f6ff;}
[data-testid="stForm"], [data-testid="stExpander"] {background: #111a30; border: 1px solid #2a385c; border-radius: 16px; padding: 1rem;}
[data-testid="stAlert"] {border-radius: 12px;}
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] button {background: linear-gradient(90deg,#6957e9,#477bdf); color: white; border: 0; border-radius: 10px; font-weight: 700;}
.stButton > button {border-radius: 10px;}
.hero {background: linear-gradient(110deg,#1b2650,#182142 55%,#1a2141); border: 1px solid #36457b; border-radius: 22px; padding: 2rem 2.2rem; margin-bottom: 1.4rem;}
.eyebrow {font-size: .77rem; font-weight: 800; color: #91a8ff; letter-spacing: .15em; text-transform: uppercase;}
.hero h1 {font-size: clamp(1.8rem,3vw,2.8rem); margin: .5rem 0;}
.hero p {color: #bac9e8; max-width: 850px;}
.module-card {background: linear-gradient(140deg,#17233e,#121b31); border: 1px solid #2c3b61; border-radius: 18px; padding: 1.5rem; min-height: 170px; margin-bottom: 1rem;}
.module-card h3 {font-size: 1.22rem; margin: .35rem 0 .65rem;}
.module-card p {font-size: .95rem; line-height: 1.65;}
.section-tag {color: #8eabff; font-weight: 700; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase;}
.result-card {background: linear-gradient(115deg,#192850,#14203c); border: 1px solid #4763a5; padding: 1.4rem; border-radius: 16px; margin: 1rem 0;}
.result-card .value {font-size: 2.7rem; font-weight: 800; color: #fff;}
.result-card .small {color: #aebee2; font-size: .9rem;}
.rank-card {background: #141f38; border: 1px solid #2a3b61; border-radius: 14px; padding: 1rem; margin: .3rem 0 .7rem;}
.rank-card .rank {font-size: .78rem; color: #92abff; font-weight: 700;}
.rank-card .id {font-size: 1.25rem; font-weight: 800; color: #f2f5ff;}
.rank-card .meta {font-size: .82rem; color: #b6c4de;}
.rank-card .score {color: #6ce0cc; font-weight: 800;}
[data-testid="stDataFrame"] {border: 1px solid #2a385c; border-radius: 12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading purchase-intent model...")
def load_ann():
    import tensorflow as tf
    model = tf.keras.models.load_model(MODEL_DIR / "purchase_intent_ann.keras", compile=False)
    scaler = joblib.load(MODEL_DIR / "purchase_intent_scaler.pkl")
    with (MODEL_DIR / "purchase_intent_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    return model, scaler, config


@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_recommender():
    neighbors = joblib.load(MODEL_DIR / "recommender_neighbors.pkl")
    metadata = joblib.load(MODEL_DIR / "product_metadata.pkl")
    with (MODEL_DIR / "recommender_config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    return neighbors, metadata, config


def hybrid_recommend(item_ids, neighbors, product_meta, config, top_k=10):
    history = set(item_ids)
    scores = {}
    for source_item in item_ids:
        source_meta = product_meta.get(source_item, {})
        source_category = source_meta.get("categoryid")
        source_parent = source_meta.get("parentid")
        for candidate, count in neighbors.get(source_item, []):
            if candidate in history:
                continue
            candidate_meta = product_meta.get(candidate)
            if candidate_meta is None or candidate_meta.get("latest_available") != 1:
                continue
            category = candidate_meta.get("categoryid")
            parent = candidate_meta.get("parentid")
            score = float(count)
            if pd.notna(source_category) and pd.notna(category) and source_category == category:
                score *= config["category_bonus"]
            elif pd.notna(source_parent) and pd.notna(parent) and source_parent == parent:
                score *= config["parent_bonus"]
            scores[candidate] = scores.get(candidate, 0.0) + score
    ranked = sorted(scores.items(), key=lambda entry: (-entry[1], entry[0]))[:top_k]
    return pd.DataFrame([
        {"Product ID": int(item), "Category ID": product_meta[item].get("categoryid"),
         "Parent Category": product_meta[item].get("parentid"), "Recommendation Score": round(score, 2)}
        for item, score in ranked
    ], columns=["Product ID", "Category ID", "Parent Category", "Recommendation Score"])


def hero(tag, title, description):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{tag}</div><h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def module_card(icon, title, description):
    st.markdown(
        f'<div class="module-card"><div style="font-size:1.8rem">{icon}</div><h3>{title}</h3><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


st.sidebar.markdown("## 🛍️ E-Commerce AI")
st.sidebar.caption("INTELLIGENCE PLATFORM · CP2")
st.sidebar.divider()
page = st.sidebar.radio("Explore", ["🏠 Dashboard", "🧠 Purchase Intent", "🛒 Recommendations", "📊 Model Performance"])
st.sidebar.divider()
st.sidebar.caption("RetailRocket · Historical e-commerce events")
st.sidebar.caption("Python · TensorFlow · Streamlit")

if page == "🏠 Dashboard":
    hero("Capstone project 02", "E-Commerce AI Intelligence Platform", "Predict next-24-hour purchase intent and discover relevant products through behavioral patterns and category-aware ranking.")
    cols = st.columns(4)
    for col, (label, value) in zip(cols, [
        ("Customer events", "2.75M+"), ("Unique visitors", "1.40M+"),
        ("Unique event products", "235K+"), ("ANN test ROC-AUC", "0.8833")
    ]):
        col.metric(label, value)
    st.write("")
    st.markdown("### AI capabilities")
    left, right = st.columns(2)
    with left:
        module_card("🧠", "Purchase intent prediction", "A trained neural network estimates whether a session will be followed by a purchase within 24 hours, using 12 behavioral and temporal features.")
    with right:
        module_card("🛒", "Hybrid product recommendations", "Training-derived co-occurrence candidates are reranked using category hierarchy, with latest-known availability filtering for the app.")
    st.markdown("### How it works")
    a, b, c = st.columns(3)
    a.info("**01 · Behavioral data**\n\nRetailRocket events, session histories and product metadata")
    b.info("**02 · AI inference**\n\nANN purchase-intent scoring and hybrid product ranking")
    c.info("**03 · Decision support**\n\nIntent classification, ranked product IDs and model evaluation")
    st.caption("Dataset is historical (2015). Product IDs and categories are anonymized; product names and images are not available.")

elif page == "🧠 Purchase Intent":
    hero("Neural network · 12 features", "Purchase Intent Prediction", "Enter a session profile to obtain the model's score and decision at the validation-selected threshold of 0.94.")
    try:
        model, scaler, config = load_ann()
    except Exception as error:
        st.error(f"Could not load ANN artifacts: {error}")
        st.stop()
    m1, m2, m3 = st.columns(3)
    m1.metric("Input features", str(len(config["features"])))
    m2.metric("Decision threshold", f'{config["threshold"]:.2f}')
    m3.metric("Test ROC-AUC", "0.8833")
    st.write("")
    st.markdown("### Session profile")
    with st.form("intent_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Session activity**")
            total_events = st.number_input("Total events", min_value=0, value=5)
            views = st.number_input("Product views", min_value=0, value=4)
            add_to_cart = st.number_input("Add-to-cart events", min_value=0, value=1)
            unique_products = st.number_input("Unique products", min_value=0, value=3)
        with c2:
            st.markdown("**Time & duration**")
            session_duration_minutes = st.number_input("Session duration (minutes)", min_value=0.0, value=10.0)
            session_hour = st.number_input("Session hour (0–23)", min_value=0, max_value=23, value=14)
            day_of_week = st.selectbox("Day of week", options=list(range(7)), format_func=lambda day: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day])
            month = st.number_input("Month", min_value=1, max_value=12, value=7)
        with c3:
            st.markdown("**Previous customer history**")
            previous_session_count = st.number_input("Previous session count", min_value=0, value=2)
            previous_purchase_count = st.number_input("Previous purchase count", min_value=0, value=0)
            hours_since_previous_session = st.number_input("Hours since previous session", min_value=0.0, value=24.0)
        submitted = st.form_submit_button("✨ Predict purchase intent", type="primary", use_container_width=True)
    if submitted:
        if total_events < views + add_to_cart:
            st.error("Total events must be at least views + add-to-cart events.")
        elif unique_products > total_events or unique_products > views + add_to_cart:
            st.error("Unique products cannot exceed the number of product-interaction events.")
        elif previous_purchase_count > previous_session_count:
            st.error("Previous purchase count cannot exceed previous session count.")
        elif previous_session_count == 0 and hours_since_previous_session != 0:
            st.error("Hours since previous session should be 0 when there is no previous session.")
        else:
            features = {
                "total_events": total_events, "views": views, "add_to_cart": add_to_cart,
                "unique_products": unique_products, "session_duration_minutes": session_duration_minutes,
                "session_hour": session_hour, "day_of_week": day_of_week,
                "is_weekend": int(day_of_week >= 5), "month": month,
                "previous_session_count": previous_session_count,
                "previous_purchase_count": previous_purchase_count,
                "hours_since_previous_session": hours_since_previous_session,
            }
            input_frame = pd.DataFrame([features])[config["features"]]
            try:
                transformed = scaler.transform(input_frame)
                score = float(model.predict(transformed, verbose=0).reshape(-1)[0])
                score = float(np.clip(score, 0, 1))
                threshold = float(config["threshold"])
                positive = score >= threshold
                st.markdown("### Prediction result")
                st.markdown(
                    f'<div class="result-card"><div class="small">Model output · sigmoid score</div>'
                    f'<div class="value">{score * 100:.2f}%</div>'
                    f'<div class="small">Decision cutoff: {threshold:.2f} · '
                    f'{"Purchase predicted" if positive else "Purchase not predicted"}</div></div>',
                    unsafe_allow_html=True,
                )
                st.progress(score)
                if positive:
                    st.success("High purchase intent: the model predicts a transaction within 24 hours.")
                else:
                    st.info("Low purchase intent: the model does not predict a transaction within 24 hours.")
                st.caption("The output is a model score, not a calibrated real-world probability or guarantee. The model was trained on 2015 data.")
                with st.expander("Inspect model inputs"):
                    st.dataframe(input_frame, use_container_width=True, hide_index=True)
            except Exception as error:
                st.error(f"Prediction failed: {error}")

elif page == "🛒 Recommendations":
    hero("Category-aware · Behavioral ranking", "Product Recommendations", "Generate recommendations from one or more RetailRocket product IDs using the saved training-based hybrid engine.")
    try:
        neighbors, metadata, rec_config = load_recommender()
    except Exception as error:
        st.error(f"Could not load recommender artifacts: {error}")
        st.stop()
    m1, m2, m3 = st.columns(3)
    m1.metric("Products with neighbors", f"{len(neighbors):,}")
    m2.metric("Products with metadata", f"{len(metadata):,}")
    m3.metric("Offline Hit Rate@10", "33.21%")
    st.write("")
    with st.form("recommend_form"):
        product_input = st.text_input("Product ID(s)", value="461686", help="Separate multiple numeric IDs with commas.")
        top_k = st.slider("Number of recommendations", 1, 20, int(rec_config.get("default_top_k", 10)))
        generate = st.form_submit_button("✨ Generate recommendations", type="primary", use_container_width=True)
    if generate:
        try:
            item_ids = list(dict.fromkeys(int(part.strip()) for part in product_input.split(",") if part.strip()))
            if not item_ids:
                st.warning("Enter at least one product ID.")
            else:
                unknown = [item for item in item_ids if item not in neighbors]
                if unknown:
                    st.caption("No training neighbors for these input IDs: " + ", ".join(map(str, unknown[:15])))
                recommendations = hybrid_recommend(item_ids, neighbors, metadata, rec_config, top_k)
                if recommendations.empty:
                    st.warning("No available recommendations found for the supplied product IDs.")
                else:
                    st.markdown("### Ranked recommendations")
                    st.caption("Scores represent relative ranking strength, not probabilities. Availability reflects the latest-known 2015 snapshot, not current retail inventory.")
                    grid = st.columns(2)
                    for index, row in recommendations.iterrows():
                        cat = row["Category ID"]
                        parent = row["Parent Category"]
                        cat_text = "Unknown" if pd.isna(cat) else str(int(cat))
                        parent_text = "Unknown" if pd.isna(parent) else str(int(parent))
                        with grid[index % 2]:
                            st.markdown(
                                f'<div class="rank-card"><div class="rank">RANK #{index + 1}</div>'
                                f'<div class="id">Product {int(row["Product ID"])}</div>'
                                f'<div class="meta">Category {cat_text} · Parent {parent_text}</div>'
                                f'<div class="score">Score {row["Recommendation Score"]:.2f}</div></div>',
                                unsafe_allow_html=True,
                            )
                    with st.expander("View recommendation table / export CSV"):
                        st.dataframe(recommendations, hide_index=True, use_container_width=True)
                        st.download_button("Download recommendations CSV", recommendations.to_csv(index=False).encode("utf-8"), file_name="recommendations.csv", mime="text/csv")
        except ValueError:
            st.error("Please use numeric product IDs separated by commas (example: 461686, 218794).")
        except Exception as error:
            st.error(f"Recommendation generation failed: {error}")

elif page == "📊 Model Performance":
    hero("Evaluation & model transparency", "Model Performance", "Review the held-out ANN test metrics and the chronological offline recommender comparison.")
    st.markdown("### ANN · Purchase intent")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Test ROC-AUC", "0.8833")
    a2.metric("Test PR-AUC", "0.2254")
    a3.metric("Test F1", "0.3265")
    a4.metric("Decision threshold", "0.94")
    ann_metrics = pd.DataFrame({"Metric": ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"], "Score": [0.8833, 0.2254, 0.2915, 0.3712, 0.3265]})
    left, right = st.columns([1.7, 1])
    with left:
        st.bar_chart(ann_metrics.set_index("Metric"), color="#8b83ff")
    with right:
        st.dataframe(ann_metrics, use_container_width=True, hide_index=True)
        st.caption("Precision: 29.15% · Recall: 37.12% · F1: 0.3265. Severe class imbalance makes accuracy less informative.")
    st.divider()
    st.markdown("### Recommender · Offline comparison")
    r1, r2, r3 = st.columns(3)
    r1.metric("Hybrid Hit Rate@5", "26.19%", "+0.71 pp vs baseline")
    r2.metric("Hybrid Hit Rate@10", "33.21%", "+0.94 pp vs baseline")
    r3.metric("Evaluation coverage", "88.22%")
    rec_metrics = pd.DataFrame({
        "Metric": ["Hit Rate@5", "Hit Rate@10", "MRR@10", "Coverage"],
        "Behavioral": [0.2548, 0.3227, 0.1682, 0.8822],
        "Hybrid": [0.2619, 0.3321, 0.1721, 0.8822],
    })
    st.bar_chart(rec_metrics.set_index("Metric"))
    st.dataframe(rec_metrics, use_container_width=True, hide_index=True)
    st.caption("Recommender: 52,433 chronological test sessions; 46,258 generated candidates. Category bonuses were compared on this period, so treat it as offline evaluation/tuning rather than an untouched final holdout.")
    with st.expander("Metric definitions"):
        st.markdown("**Hit Rate@K:** Share of evaluable sessions where the hidden final item appears in the first K recommendations.\n\n**MRR@10:** Mean reciprocal rank of the hidden item, counting ranks beyond 10 as zero.\n\n**Coverage:** Share of test sessions for which the model could generate at least one candidate.\n\n**PR-AUC:** Precision-recall area, useful when positive outcomes are rare.")

st.divider()
st.caption("E-Commerce AI Intelligence Platform · Capstone Project 2 · Historical RetailRocket dataset")
