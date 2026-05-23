"""Composition featurization helpers used by the XGBoost and GA workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TOP_FEATURES = [
    "meanGroup",
    "CoordinationNumber",
    "meanDcount",
    "meanMeltingP",
    "minGroup",
    "maxAtomicRadius",
    "meanAtomicRadius",
    "minDensity",
    "meanElectronegativity",
]


def load_element_table(path: str | Path) -> pd.DataFrame:
    """Load elemental descriptors indexed by chemical symbol."""
    table = pd.read_csv(path)
    if "Symbol" not in table.columns:
        raise ValueError("Element table must contain a Symbol column.")
    table = table.rename(
        columns={
            "Group ": "Group",
            "MeltingPoint": "MeltingP",
            "IonizationEnergy": "Ion",
            "ElectronAffinity": "Affinity",
            "Outermost_p_count": "Pcount",
            "Outermost_d_count": "Dcount",
            "EnthalpyFusion": "Enthalpy",
        }
    )
    return table.set_index("Symbol")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a simple chemical formula into element counts."""
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if not tokens:
        raise ValueError(f"Could not parse formula: {formula}")
    counts: dict[str, int] = {}
    for symbol, raw_count in tokens:
        counts[symbol] = counts.get(symbol, 0) + int(raw_count or 1)
    return counts


def _stat_features(
    counts: dict[str, int],
    element_table: pd.DataFrame,
    property_name: str,
) -> dict[str, float]:
    total_atoms = sum(counts.values())
    values = []
    weights = []
    for symbol, count in counts.items():
        if symbol not in element_table.index:
            raise ValueError(f"Missing elemental descriptors for {symbol}.")
        values.append(float(element_table.loc[symbol, property_name]))
        weights.append(count / total_atoms)

    values_arr = np.asarray(values, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)
    mean = float(np.dot(values_arr, weights_arr))
    variance = float(np.dot((values_arr - mean) ** 2, weights_arr))
    return {
        f"mean{property_name}": mean,
        f"max{property_name}": float(values_arr.max()),
        f"min{property_name}": float(values_arr.min()),
        f"std{property_name}": float(np.sqrt(variance)),
    }


def composition_feature_vector(
    formula: str,
    element_table: pd.DataFrame,
    feature_names: Iterable[str],
    coordination_number: int | float = 4,
) -> pd.DataFrame:
    """Build one feature row for a formula in the order expected by a model."""
    counts = parse_formula(formula)
    row: dict[str, float] = {}
    cache: dict[str, dict[str, float]] = {}
    non_oxygen_elements = [symbol for symbol in counts if symbol != "O"]
    total_atoms = sum(counts.values())

    for feature in feature_names:
        if feature == "CoordinationNumber":
            row[feature] = float(coordination_number)
            continue
        if feature == "NumElementsSurface":
            row[feature] = float(len(non_oxygen_elements))
            continue
        if feature == "PseudoLabelled":
            row[feature] = 0.0
            continue
        if feature == "formula_atom_count":
            row[feature] = float(total_atoms)
            continue
        if feature == "formula_unique_elements":
            row[feature] = float(len(counts))
            continue
        if feature.startswith("frac_"):
            symbol = feature.removeprefix("frac_")
            row[feature] = float(counts.get(symbol, 0) / total_atoms)
            continue

        match = re.match(r"(mean|max|min|std)([A-Z].*)", feature)
        if not match:
            row[feature] = 0.0
            continue

        _, property_name = match.groups()
        if property_name not in cache:
            cache[property_name] = _stat_features(counts, element_table, property_name)
        row[feature] = cache[property_name][feature]

    return pd.DataFrame([row], columns=list(feature_names))
