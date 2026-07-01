import pandas as pd
import numpy as np
import os

np.random.seed(42)

TEAMS = {
    "Brazil":       {"attack": 85, "defense": 80},
    "France":       {"attack": 83, "defense": 82},
    "England":      {"attack": 80, "defense": 79},
    "Germany":      {"attack": 79, "defense": 81},
    "Argentina":    {"attack": 84, "defense": 78},
    "Spain":        {"attack": 81, "defense": 80},
    "Portugal":     {"attack": 82, "defense": 75},
    "Netherlands":  {"attack": 78, "defense": 76},
    "Belgium":      {"attack": 77, "defense": 74},
    "Italy":        {"attack": 74, "defense": 80},
    "Croatia":      {"attack": 73, "defense": 72},
    "Uruguay":      {"attack": 75, "defense": 73},
    "Denmark":      {"attack": 72, "defense": 74},
    "Switzerland":  {"attack": 70, "defense": 73},
    "Morocco":      {"attack": 69, "defense": 72},
    "Senegal":      {"attack": 68, "defense": 67},
    "Japan":        {"attack": 66, "defense": 68},
    "South Korea":  {"attack": 65, "defense": 65},
    "USA":          {"attack": 67, "defense": 66},
    "Mexico":       {"attack": 68, "defense": 65},
}

TOURNAMENT_TYPES = ["FIFA World Cup", "UEFA Euro", "Copa America", "Friendly", "Qualifier"]


def simulate_match(home_team, away_team):
    h = TEAMS[home_team]
    a = TEAMS[away_team]

    home_advantage = 0.08
    h_score_rate = (h["attack"] / 100) * 2.2 + home_advantage
    a_score_rate = (a["attack"] / 100) * 1.9

    h_defense_factor = 1 - (h["defense"] / 100) * 0.5
    a_defense_factor = 1 - (a["defense"] / 100) * 0.5

    home_goals = max(0, round(np.random.poisson(h_score_rate * a_defense_factor)))
    away_goals = max(0, round(np.random.poisson(a_score_rate * h_defense_factor)))

    return home_goals, away_goals


def generate_matches(n=2000):
    teams = list(TEAMS.keys())
    records = []
    base_date = pd.Timestamp("2010-01-01")

    for _ in range(n):
        home, away = np.random.choice(teams, 2, replace=False)
        days_offset = np.random.randint(0, 4800)
        date = base_date + pd.Timedelta(days=days_offset)
        tournament = np.random.choice(TOURNAMENT_TYPES, p=[0.15, 0.15, 0.10, 0.45, 0.15])
        neutral = tournament == "FIFA World Cup" or np.random.rand() < 0.2

        home_goals, away_goals = simulate_match(home, away)

        records.append({
            "date":       date.strftime("%Y-%m-%d"),
            "home_team":  home,
            "away_team":  away,
            "home_score": home_goals,
            "away_score": away_goals,
            "tournament": tournament,
            "neutral":    neutral,
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_matches(2000)
    df.to_csv("data/results.csv", index=False)
    print(f"Generated {len(df)} matches")
    print(df.head())