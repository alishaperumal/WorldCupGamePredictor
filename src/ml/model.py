import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.feature_engineering import FEATURE_COLS, FIFA_RANKINGS

OUTCOME_LABELS = {0: "Home Win", 1: "Draw", 2: "Away Win"}
MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "model.pkl"


def train(feature_df):
    """
    Train the Random Forest on historical match features.
    We split 80/20 — the model never sees the test set during training.
    """
    X = feature_df[FEATURE_COLS].values
    y = feature_df["outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced",
    )

    # Cross-validation gives a more reliable accuracy estimate than one split
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "log_loss":  round(log_loss(y_test, y_prob), 4),
        "cv_mean":   round(cv_scores.mean(), 4),
        "cv_std":    round(cv_scores.std(), 4),
        "n_train":   len(X_train),
        "n_test":    len(X_test),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(zip(FEATURE_COLS, model.feature_importances_.tolist())),
    }

    return model, metrics


def save_model(model, metrics):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "metrics": metrics}, f)

    metrics_path = MODEL_PATH.parent / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Run train_model.py first.")
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["metrics"]


def predict_match(home_team, away_team, tournament="FIFA World Cup", model=None, feature_df=None):
    """
    Predict win/draw/loss probabilities for a single matchup.
    """
    if model is None:
        model, _ = load_model()

    from ml.feature_engineering import calculate_team_form, calculate_head_to_head

    tournament_weights = {
        "FIFA World Cup": 1.0, "UEFA Euro": 0.85,
        "Copa America": 0.80, "Qualifier": 0.60, "Friendly": 0.30,
    }

    if feature_df is not None:
        form = calculate_team_form(feature_df)
        h2h  = calculate_head_to_head(feature_df)

        home_rows = form[(form["team"] == home_team) & (form["is_home"] == True)].tail(5)
        away_rows = form[(form["team"] == away_team) & (form["is_home"] == False)].tail(5)

        def safe_mean(rows, col, default):
            if len(rows) > 0 and col in rows.columns:
                v = rows[col].dropna()
                return float(v.mean()) if len(v) > 0 else default
            return default

        home_win_rate  = safe_mean(home_rows, "form_win_rate", 0.5)
        away_win_rate  = safe_mean(away_rows, "form_win_rate", 0.5)
        home_goal_rate = safe_mean(home_rows, "form_goal_rate", 1.2)
        away_goal_rate = safe_mean(away_rows, "form_goal_rate", 1.2)
        home_concede   = safe_mean(home_rows, "form_concede_rate", 1.0)
        away_concede   = safe_mean(away_rows, "form_concede_rate", 1.0)

        key = tuple(sorted([home_team, away_team]))
        h2h_rec = h2h.get(key, {"wins_A": 0, "wins_B": 0, "draws": 0, "team_A": key[0]})
        total = h2h_rec["wins_A"] + h2h_rec["wins_B"] + h2h_rec["draws"]
        h2h_rate = (h2h_rec["wins_A"] if h2h_rec["team_A"] == home_team else h2h_rec["wins_B"]) / total if total > 0 else 0.5
    else:
        home_win_rate = away_win_rate = 0.5
        home_goal_rate = away_goal_rate = 1.2
        home_concede = away_concede = 1.0
        h2h_rate = 0.5

    home_rank = FIFA_RANKINGS.get(home_team, 50)
    away_rank = FIFA_RANKINGS.get(away_team, 50)

    features = np.array([[
        away_rank - home_rank,
        home_win_rate,
        away_win_rate,
        home_goal_rate,
        away_goal_rate,
        home_concede,
        away_concede,
        h2h_rate,
        tournament_weights.get(tournament, 0.5),
        1,
    ]])

    probs = model.predict_proba(features)[0]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win":  round(float(probs[0]), 4),
        "draw":      round(float(probs[1]), 4),
        "away_win":  round(float(probs[2]), 4),
        "predicted": OUTCOME_LABELS[int(np.argmax(probs))],
    }