# E-Commerce AI Intelligence Platform

**Purchase Intent Prediction | Hybrid Product Recommendations | Interactive Streamlit Dashboard**

An end-to-end machine learning project built on the **RetailRocket** e-commerce interaction dataset. The platform analyzes browsing behavior, estimates the likelihood of a customer's next purchase using an artificial neural network (ANN), and supports product recommendations through an interactive Streamlit dashboard.

## Project overview

Online retailers generate large volumes of customer interaction data. This project explores how those events can be transformed into actionable insights and machine learning features for purchase-intent prediction and recommendations.

**Main components**
- Exploratory data analysis and session-based feature engineering
- Comparison of Logistic Regression, Random Forest, and ANN classifiers
- ANN-based prediction of whether a visitor purchases within the next **24 hours**
- Hybrid product recommendation functionality
- Streamlit interface for interactive predictions, recommendations, and model insights

## Dataset

**Source:** [RetailRocket e-commerce dataset (Kaggle)](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

Files used in the analysis:
- `events.csv`
- `item_properties_part1.csv`
- `item_properties_part2.csv`
- `category_tree.csv`

The events file contains **2,756,101 interactions** with event types such as `view`, `addtocart`, and `transaction`.

> The raw dataset is not included in this repository. Download it from the original source and follow the notebook's data-path configuration.

## Methodology

1. **Data preparation:** clean event records, convert timestamps, and inspect customer interaction patterns.
2. **Sessionization:** group visitor events into sessions using a 30-minute inactivity gap.
3. **Target creation:** identify whether a visitor makes a transaction within 24 hours after the end of an eligible session.
4. **Feature engineering:** create 12 session and history features, including event counts, cart activity, session duration, time indicators, and previous visitor activity.
5. **Time-based splitting:** separate train, validation, and test data chronologically to reduce leakage.
6. **Modeling:** compare Logistic Regression, Random Forest, and an ANN implemented with TensorFlow/Keras.
7. **Recommendation system:** provide product suggestions using the project's hybrid recommender logic.
8. **Application:** expose the trained models and recommendations through a Streamlit dashboard.

## Modeling details

The final purchase-prediction dataset contains **1,740,684 eligible sessions**, with **6,015 positive examples** (approximately **0.35%**). Because purchases are rare, model evaluation considers metrics such as **ROC-AUC, PR-AUC, precision, recall, and F1-score**, rather than accuracy alone.

The ANN uses fully connected layers with dropout and a sigmoid output for binary purchase prediction. Preprocessing artifacts must be consistent between training and inference.

### Validation results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Logistic Regression | 0.8636 | 0.1602 |
| Random Forest | 0.8652 | 0.1953 |
| ANN | See training notebook for final validation/test evaluation | — |

*The ANN reached approximately 0.8872 validation ROC-AUC by epoch 6 in recorded training results; refer to the saved evaluation output for the final selected checkpoint and test metrics.*

## Technology stack

**Python**, Pandas, NumPy, scikit-learn, TensorFlow/Keras, Streamlit, Joblib, Matplotlib, and Seaborn. Additional packages may be required depending on the exact application and notebook imports.

## Repository structure

Arrange your existing files as applicable; names below are illustrative and should match your actual upload:

```text
E-Commerce-AI-Intelligence-Platform/
├── app.py                  # Streamlit application (or app/app.py)
├── notebooks/              # EDA, feature engineering, model training
├── screenshots/            # Dashboard screenshots
├── reports/                # Capstone presentation or project report
├── models/                 # Optional small, redistributable model artifacts
├── requirements.txt
└── README.md
```

## Run locally

1. Clone the repository and open its folder:

   ```bash
   git clone https://github.com/chetanhumbi/E-Commerce-AI-Intelligence-Platform
   cd E-Commerce-AI-Intelligence-Platform
   ```

2. Create and activate a virtual environment on Windows:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Place the required **trained ANN, scaler, recommender, and metadata files** in the locations expected by `app.py`. If these artifacts are not included, train/export them using the project notebook first.

5. Start the dashboard:

   ```bash
   streamlit run app.py
   ```

   If your app is inside an `app/` folder, use `streamlit run app/app.py` instead.

> **Reproducibility note:** Check the paths and filenames in `app.py` before running. This repository may not be runnable until the trained artifacts and any required item metadata are supplied. For large files, use an appropriate artifact storage method rather than committing them directly to GitHub.


## Key learnings

- Session-based behavioral signals can help identify purchase intent.
- Chronological evaluation is important for event-stream prediction tasks.
- Rare positive outcomes require careful metric selection and threshold evaluation.
- A user-facing dashboard helps make model predictions and recommendations easier to explore.

## Notes

- Do not commit API keys, secrets, virtual environments, raw datasets without redistribution permission, or very large model artifacts.
- This project is an educational capstone and is not presented as a production-deployed service.
