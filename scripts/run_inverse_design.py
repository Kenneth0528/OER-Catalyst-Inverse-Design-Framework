"""Example inverse-design run using XGBoost predictions and a genetic algorithm."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oer_inverse_design.features import TOP_FEATURES, load_element_table
from oer_inverse_design.genetic_algo import GeneticSearchConfig, MaterialInverseDesignGA


CATALYST_METAL_POOL = [
    "Ag",
    "Au",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Ir",
    "Mn",
    "Mo",
    "Ni",
    "Pd",
    "Pt",
    "Rh",
    "Ru",
    "Sc",
    "Ti",
    "V",
    "W",
    "Y",
    "Zn",
    "Zr",
]


def main() -> None:
    training = pd.read_csv(ROOT / "data" / "processed" / "TRUE_PSEUDO_featData.csv")
    element_table = load_element_table(ROOT / "data" / "processed" / "PubChemElements_all.csv")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        random_state=42,
        n_jobs=1,
    )
    model.fit(training[TOP_FEATURES], training["ReactionEnergy"])

    config = GeneticSearchConfig(
        target_energy=1.20,
        population_size=120,
        generations=25,
        min_elements=2,
        random_state=7,
    )
    ga = MaterialInverseDesignGA(model, element_table, TOP_FEATURES, CATALYST_METAL_POOL, config)
    results = ga.run()

    output = ROOT / "results" / "example_ga_candidates.csv"
    results.head(20).to_csv(output, index=False)
    print(results.head(10).to_string(index=False))
    print(f"\nSaved top candidates to {output}")


if __name__ == "__main__":
    main()
