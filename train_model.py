import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.generate_sample_data import generate_matches
from ml.feature_engineering import build_features, FEATURE_COLS
from ml.model import train, save_model


def main():
    print("=" * 60)
    print("  World Cup Match Predictor — Model Training")
    print("=" * 60)

    print("\n[1/4] Generating match dataset...")
    raw_df = generate_matches(n=2000)
    raw_df.to_csv("data/results.csv", index=False)
    print(f"      {len(raw_df)} matches generated.")

    print("\n[2/4] Engineering features...")
    feature_df = build_features(raw_df)
    feature_df.to_csv("data/features.csv", index=False)
    print(f"      Feature matrix: {feature_df.shape[0]} rows x {len(FEATURE_COLS)} features")

    outcome_dist = feature_df["outcome"].value_counts()
    print(f"      Outcomes — Home Win: {outcome_dist.get(0,0)}, Draw: {outcome_dist.get(1,0)}, Away Win: {outcome_dist.get(2,0)}")

    print("\n[3/4] Training Random Forest model...")
    model, metrics = train(feature_df)

    print("\n[4/4] Evaluation Results:")
    print(f"      Accuracy:      {metrics['accuracy']:.1%}")
    print(f"      Log Loss:      {metrics['log_loss']:.4f}")
    print(f"      CV Score:      {metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
    print(f"      Train samples: {metrics['n_train']}")
    print(f"      Test samples:  {metrics['n_test']}")

    print("\n      Feature Importance:")
    for feat, imp in sorted(metrics["feature_importance"].items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"      {feat:<30} {bar} {imp:.3f}")

    save_model(model, metrics)

    print("\n✅ Training complete!")
    print("   → python predict.py 'Brazil' 'Germany'")
    print("   → python simulate_tournament.py")


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    main()