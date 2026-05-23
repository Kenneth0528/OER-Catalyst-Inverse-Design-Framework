"""Train and evaluate the top-feature XGBoost model used by the GA."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oer_inverse_design.features import TOP_FEATURES


DATA = ROOT / "data" / "processed" / "TRUE_PSEUDO_featData.csv"


def main() -> None:
    df = pd.read_csv(DATA)
    data = df[TOP_FEATURES + ["ReactionEnergy"]].dropna()
    x = data[TOP_FEATURES]
    y = data["ReactionEnergy"]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        random_state=42,
        n_jobs=1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        x,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=True,
        n_jobs=1,
    )

    print(f"samples: {len(data)}")
    print(f"train_r2: {scores['train_r2'].mean():.4f} +/- {scores['train_r2'].std():.4f}")
    print(f"test_r2:  {scores['test_r2'].mean():.4f} +/- {scores['test_r2'].std():.4f}")
    print(f"test_mae: {-scores['test_mae'].mean():.4f} +/- {scores['test_mae'].std():.4f} eV")
    print(f"test_rmse:{-scores['test_rmse'].mean():.4f} +/- {scores['test_rmse'].std():.4f} eV")


if __name__ == "__main__":
    main()
