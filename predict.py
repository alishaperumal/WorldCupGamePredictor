import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml.model import predict_match, load_model
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Predict a World Cup match outcome.")
    parser.add_argument("home", help="Home team (e.g. 'Brazil')")
    parser.add_argument("away", help="Away team (e.g. 'Germany')")
    parser.add_argument(
        "--tournament",
        default="FIFA World Cup",
        choices=["FIFA World Cup", "UEFA Euro", "Copa America", "Qualifier", "Friendly"],
        help="Tournament type"
    )
    args = parser.parse_args()

    print(f"\n Predicting: {args.home} vs {args.away} ({args.tournament})")
    print("-" * 50)

    model, metrics = load_model()
    print(f"Model accuracy: {metrics['accuracy']:.1%}")

    feature_df = pd.read_csv("data/results.csv") if Path("data/results.csv").exists() else None

    result = predict_match(
        args.home,
        args.away,
        args.tournament,
        model=model,
        feature_df=feature_df
    )

    home_pct = result["home_win"] * 100
    draw_pct  = result["draw"]    * 100
    away_pct  = result["away_win"] * 100

    bar_home = "█" * int(home_pct / 3)
    bar_draw = "█" * int(draw_pct / 3)
    bar_away = "█" * int(away_pct / 3)

    print(f"\n  {args.home:<20} {bar_home:<20} {home_pct:.1f}%")
    print(f"  {'Draw':<20} {bar_draw:<20} {draw_pct:.1f}%")
    print(f"  {args.away:<20} {bar_away:<20} {away_pct:.1f}%")
    print(f"\n  Predicted outcome: {result['predicted']}")


if __name__ == "__main__":
    main()