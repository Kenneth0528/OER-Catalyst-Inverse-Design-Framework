"""Genetic algorithm for XGBoost-guided catalyst composition search."""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd
from typing import Iterable

import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

from .features import composition_feature_vector


def normalize_formula(formula: str) -> str:
    """Return a reduced formula string with no whitespace."""
    counts = {el: int(amount) for el, amount in Composition(formula).as_dict().items()}
    return formula_from_counts(counts)


def formula_from_counts(counts: dict[str, int]) -> str:
    """Create a reduced formula from integer element counts."""
    positive = {el: int(count) for el, count in counts.items() if count > 0}
    if not positive:
        raise ValueError("Composition cannot be empty.")
    divisor = reduce(gcd, positive.values())
    reduced = {el: count // divisor for el, count in positive.items()}
    return "".join(f"{el}{n if n > 1 else ''}" for el, n in sorted(reduced.items()))


@dataclass
class GeneticSearchConfig:
    """Configuration for inverse design."""

    target_energy: float = 1.20
    population_size: int = 200
    generations: int = 50
    mutation_rate: float = 0.30
    crossover_rate: float = 0.70
    elite_fraction: float = 0.10
    min_elements: int = 1
    max_elements: int = 4
    max_stoichiometric_count: int = 5
    random_state: int = 42


class MaterialInverseDesignGA:
    """Search formula space for compositions whose model prediction matches a target."""

    def __init__(
        self,
        model,
        element_table: pd.DataFrame,
        feature_names: Iterable[str],
        elements: Iterable[str],
        config: GeneticSearchConfig | None = None,
    ) -> None:
        self.model = model
        self.element_table = element_table
        self.feature_names = list(feature_names)
        self.elements = [el for el in dict.fromkeys(elements) if el in element_table.index]
        self.config = config or GeneticSearchConfig()
        self.rng = np.random.default_rng(self.config.random_state)
        self.history: list[dict[str, float | str | int]] = []

        if not self.elements:
            raise ValueError("No searchable elements are present in the element descriptor table.")

    @classmethod
    def metallic_search(
        cls,
        model,
        element_table: pd.DataFrame,
        feature_names: Iterable[str],
        config: GeneticSearchConfig | None = None,
    ) -> "MaterialInverseDesignGA":
        elements = [
            symbol
            for symbol in element_table.index
            if Element(symbol).is_metal and Element(symbol).Z <= 83
        ]
        return cls(model, element_table, feature_names, elements, config)

    def _random_formula(self) -> str:
        min_elements = max(1, min(self.config.min_elements, len(self.elements)))
        max_elements = max(min_elements, min(self.config.max_elements, len(self.elements)))
        n_elements = int(self.rng.integers(min_elements, max_elements + 1))
        chosen = self.rng.choice(self.elements, size=n_elements, replace=False)
        counts = {
            str(el): int(self.rng.integers(1, self.config.max_stoichiometric_count + 1))
            for el in chosen
        }
        return formula_from_counts(counts)

    def _predict(self, formula: str, coordination_number: int) -> float:
        features = composition_feature_vector(
            formula,
            self.element_table,
            self.feature_names,
            coordination_number=coordination_number,
        )
        return float(self.model.predict(features)[0])

    def evaluate_formula(self, formula: str) -> dict[str, float | str | int]:
        """Evaluate all coordination numbers and keep the closest target match."""
        reduced = normalize_formula(formula)
        rows = []
        for cn in [1, 2, 3, 4]:
            prediction = self._predict(reduced, cn)
            rows.append(
                {
                    "formula": reduced,
                    "CN": cn,
                    "predicted_energy": prediction,
                    "error": abs(prediction - self.config.target_energy),
                }
            )
        return min(rows, key=lambda row: row["error"])

    def _fitness(self, formula: str) -> float:
        try:
            return -float(self.evaluate_formula(formula)["error"])
        except Exception:
            return -1e9

    def _mutate(self, formula: str) -> str:
        counts = {el: int(amount) for el, amount in Composition(formula).as_dict().items()}
        action = self.rng.choice(["count", "swap", "add_remove"], p=[0.45, 0.35, 0.20])

        if action == "count" and counts:
            el = str(self.rng.choice(list(counts)))
            delta = int(self.rng.choice([-2, -1, 1, 2]))
            counts[el] = max(1, min(self.config.max_stoichiometric_count, counts[el] + delta))
        elif action == "swap" and counts:
            old = str(self.rng.choice(list(counts)))
            available = [el for el in self.elements if el not in counts]
            if available:
                counts[str(self.rng.choice(available))] = counts.pop(old)
        elif len(counts) < self.config.max_elements:
            available = [el for el in self.elements if el not in counts]
            if available:
                counts[str(self.rng.choice(available))] = int(
                    self.rng.integers(1, self.config.max_stoichiometric_count + 1)
                )
        elif len(counts) > self.config.min_elements:
            counts.pop(str(self.rng.choice(list(counts))))

        return formula_from_counts(counts)

    def _crossover(self, parent_a: str, parent_b: str) -> str:
        comp_a = {el: int(amount) for el, amount in Composition(parent_a).as_dict().items()}
        comp_b = {el: int(amount) for el, amount in Composition(parent_b).as_dict().items()}
        child: dict[str, int] = {}

        for el in sorted(set(comp_a) | set(comp_b)):
            source = comp_a if self.rng.random() < 0.5 else comp_b
            if el in source:
                child[el] = source[el]

        if not child:
            return self._random_formula()
        while len(child) > self.config.max_elements:
            child.pop(str(self.rng.choice(list(child))))
        while len(child) < self.config.min_elements:
            available = [el for el in self.elements if el not in child]
            if not available:
                break
            child[str(self.rng.choice(available))] = int(
                self.rng.integers(1, self.config.max_stoichiometric_count + 1)
            )
        return formula_from_counts(child)

    def run(self) -> pd.DataFrame:
        """Run the search and return unique candidates sorted by target error."""
        population = [self._random_formula() for _ in range(self.config.population_size)]
        elite_count = max(1, int(self.config.population_size * self.config.elite_fraction))
        candidates: dict[tuple[str, int], dict[str, float | str | int]] = {}

        for generation in range(self.config.generations):
            fitness = np.asarray([self._fitness(formula) for formula in population])
            ranked_idx = np.argsort(fitness)[::-1]
            best = self.evaluate_formula(population[int(ranked_idx[0])])
            best["generation"] = generation
            self.history.append(best)

            for formula in population:
                try:
                    row = self.evaluate_formula(formula)
                    key = (str(row["formula"]), int(row["CN"]))
                    if key not in candidates or float(row["error"]) < float(candidates[key]["error"]):
                        candidates[key] = row
                except Exception:
                    continue

            selected = [population[int(i)] for i in ranked_idx[: max(2, self.config.population_size // 2)]]
            next_population = [population[int(i)] for i in ranked_idx[:elite_count]]
            while len(next_population) < self.config.population_size:
                if self.rng.random() < self.config.crossover_rate and len(selected) >= 2:
                    parent_a, parent_b = self.rng.choice(selected, size=2, replace=False)
                    child = self._crossover(str(parent_a), str(parent_b))
                else:
                    child = str(self.rng.choice(selected))
                if self.rng.random() < self.config.mutation_rate:
                    child = self._mutate(child)
                next_population.append(child)
            population = next_population

        return pd.DataFrame(candidates.values()).sort_values("error").reset_index(drop=True)
