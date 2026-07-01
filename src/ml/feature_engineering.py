import pandas as pd
import numpy as np

FIFA_RANKINGS = {
    "Brazil": 1, "France": 2, "Argentina": 3, "Germany": 4,
    "England": 5, "Portugal": 6, "Spain": 7, "Netherlands": 8,
    "Belgium": 9, "Italy": 10, "Uruguay": 11, "Croatia": 12,
    "Morocco": 13, "Denmark": 13, "Switzerland": 14, "USA": 21,
    "Mexico": 18, "Senegal": 20, "Japan": 22, "South Korea": 25,
}

FEATURE_COLS = [
    "rank_diff",
    "home_form_win_rate",
    "away_form_win_rate",
    "home_form_goal_rate",
    "away_form_goal_rate",
    "home_form_concede",
    "away_form_concede",
    "h2h_home_win_rate",
    "tournament_weight",
    "is_neutral",
]


def calculate_team_form(df, n_games=5):
    """
    For each match, compute each team's stats from their last N games.
    We shift by 1 so we never include the current match's result — 
    that would be cheating (data leakage).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Reshape so each game appears once per team
    home_view = df[["date", "home_team", "away_team", "home_score", "away_score"]].copy()
    home_view.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    home_view["is_home"] = True

    away_view = df[["date", "away_team", "home_team", "away_score", "home_score"]].copy()
    away_view.columns = ["date", "team", "opponent", "goals_for", "goals_against"]
    away_view["is_home"] = False

    team_games = pd.concat([home_view, away_view]).sort_values("date").reset_index(drop=True)
    team_games["win"] = (team_games["goals_for"] > team_games["goals_against"]).astype(int)

    def rolling_stats(group):
        group = group.sort_values("date")
        shifted = group.shift(1)
        group["form_win_rate"]     = shifted["win"].rolling(n_games, min_periods=1).mean()
        group["form_goal_rate"]    = shifted["goals_for"].rolling(n_games, min_periods=1).mean()
        group["form_concede_rate"] = shifted["goals_against"].rolling(n_games, min_periods=1).mean()
        return group

    team_games["team_key"] = team_games["team"]
    result = team_games.groupby("team_key", group_keys=False).apply(rolling_stats)
    result["team"] = result["team_key"]
    return result.reset_index(drop=True)


def calculate_head_to_head(df):
    """
    For each pair of teams, count historical wins/draws/losses.
    """
    h2h = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        key = tuple(sorted([h, a]))
        if key not in h2h:
            h2h[key] = {"wins_A": 0, "wins_B": 0, "draws": 0, "team_A": key[0]}
        if row["home_score"] > row["away_score"]:
            winner = h
        elif row["away_score"] > row["home_score"]:
            winner = a
        else:
            winner = "draw"

        if winner == key[0]:
            h2h[key]["wins_A"] += 1
        elif winner == key[1]:
            h2h[key]["wins_B"] += 1
        else:
            h2h[key]["draws"] += 1
    return h2h


def build_features(df, n_form_games=5):
    """
    Build the full feature matrix. Each row = one match.
    Each column = one numeric signal the model learns from.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    form_data = calculate_team_form(df, n_games=n_form_games)
    h2h_data  = calculate_head_to_head(df)

    tournament_weights = {
        "FIFA World Cup": 1.0,
        "UEFA Euro":      0.85,
        "Copa America":   0.80,
        "Qualifier":      0.60,
        "Friendly":       0.30,
    }

    rows = []

    for _, match in df.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]

        home_form = form_data[(form_data["team"] == home) & (form_data["date"] == date) & (form_data["is_home"] == True)]
        away_form = form_data[(form_data["team"] == away) & (form_data["date"] == date) & (form_data["is_home"] == False)]

        def get_form(form_df, col, default=0.5):
            if len(form_df) > 0:
                val = form_df[col].values[0]
                return float(val) if not pd.isna(val) else default
            return default

        home_rank = FIFA_RANKINGS.get(home, 50)
        away_rank = FIFA_RANKINGS.get(away, 50)

        key = tuple(sorted([home, away]))
        h2h = h2h_data.get(key, {"wins_A": 0, "wins_B": 0, "draws": 0, "team_A": key[0]})
        total = h2h["wins_A"] + h2h["wins_B"] + h2h["draws"]
        if total > 0:
            h2h_rate = (h2h["wins_A"] if h2h["team_A"] == home else h2h["wins_B"]) / total
        else:
            h2h_rate = 0.5

        if match["home_score"] > match["away_score"]:
            outcome = 0  # Home win
        elif match["home_score"] == match["away_score"]:
            outcome = 1  # Draw
        else:
            outcome = 2  # Away win

        rows.append({
            "rank_diff":           away_rank - home_rank,
            "home_form_win_rate":  get_form(home_form, "form_win_rate"),
            "away_form_win_rate":  get_form(away_form, "form_win_rate"),
            "home_form_goal_rate": get_form(home_form, "form_goal_rate", 1.2),
            "away_form_goal_rate": get_form(away_form, "form_goal_rate", 1.2),
            "home_form_concede":   get_form(home_form, "form_concede_rate", 1.0),
            "away_form_concede":   get_form(away_form, "form_concede_rate", 1.0),
            "h2h_home_win_rate":   h2h_rate,
            "tournament_weight":   tournament_weights.get(match.get("tournament", "Friendly"), 0.5),
            "is_neutral":          int(match.get("neutral", False)),
            "home_team":           home,
            "away_team":           away,
            "date":                str(date.date()),
            "home_score":          match["home_score"],
            "away_score":          match["away_score"],
            "outcome":             outcome,
        })

    return pd.DataFrame(rows)