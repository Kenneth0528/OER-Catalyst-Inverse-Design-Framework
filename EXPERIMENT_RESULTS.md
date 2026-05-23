# Model Improvement Experiments

This document summarizes the additional improvement experiments run after the initial GitHub-ready cleanup.

## Baseline

Original cleaned benchmark:

| Dataset | Features | Model | Test R² | Test MAE |
| --- | --- | --- | ---: | ---: |
| All rows | Top 9 descriptors | XGBoost | 0.8205 ± 0.0372 | 0.2069 ± 0.0103 eV |

## Descriptor and Model Expansion

The most useful non-leaky improvements were:

| Dataset | Features | Model | Test R² | Test MAE |
| --- | --- | --- | ---: | ---: |
| All rows | Numeric descriptors + formula element fractions + facet/site family | Voting ensemble | 0.8251 | 0.2026 eV |
| All rows | Numeric descriptors + formula element fractions + facet/site family | XGBoost | 0.8250 | 0.2064 eV |
| All rows | Numeric descriptors + formula element fractions | XGBoost | 0.8240 | 0.2067 eV |

The near-perfect diagnostic result obtained when including `O*_energy` and `OH*_energy` was intentionally treated as leakage, not as a fair improvement, because those labels are not known for new generated compositions.

## Metallic/Alloy-Only Subset

Because the generated candidates are metallic/alloy compositions, a metallic-only training subset was evaluated.

| Dataset | Features | Model | Test R² | Test MAE |
| --- | --- | --- | ---: | ---: |
| Metallic-only | Top 9 descriptors | XGBoost | 0.7767 | 0.2076 eV |
| Metallic-only | Numeric descriptors + formula fractions + facet/site family | XGBoost | 0.7957 | 0.1993 eV |
| Metallic-only | Numeric descriptors + formula fractions + facet/site family | Voting ensemble | 0.7916 | 0.1982 eV |

This is a stronger within-domain improvement for metallic/alloy candidate generation.

## Composition Search Results

All generated target errors are surrogate-model target errors:

```text
abs(predicted ReactionEnergy - 1.20 eV)
```

They are not DFT validation errors.

| Search/model | Best candidate | CN | Predicted energy | Target error |
| --- | --- | ---: | ---: | ---: |
| Original cleaned GA example | Ag4Rh | 1 | 1.199200 eV | 0.000800 eV |
| Expanded descriptor GA | Au4Pt2RhZn | 1 | 1.199842 eV | 0.000158 eV |
| Formula-fraction all-row GA | Ag4Ru | 2 | 1.200901 eV | 0.000901 eV |
| Metallic-only formula-fraction GA | Cu2Ni | 1 | 1.199867 eV | 0.000133 eV |
| Bayesian-style acquisition | Pt3Rh | 1 | 1.201109 eV | 0.001109 eV |

The best current generated candidate is therefore `Cu2Ni` from the metallic-only formula-fraction model. It is more consistent with the generated composition class than an oxide model would be.

## Interpretation

- Expanding descriptors systematically helped, but only modestly on the full dataset.
- Formula fractions added useful compositional identity beyond aggregated elemental statistics.
- Ensemble models improved MAE more than R² in the all-row setting.
- A metallic-only model is more appropriate for the current generated candidate class.
- Bayesian-style acquisition was implemented, but in this deterministic surrogate objective the GA found closer target matches in this run.

## Next Practical Improvements

1. Add a separate oxide workflow with charge neutrality and oxidation-state constraints.
2. Evaluate candidates by uncertainty, not only target closeness.
3. Run DFT validation for top candidates from `expanded_ga_candidates.csv` and `metallic_only_ga_candidates.csv`.
4. Add stability filters: formation energy, phase separation, and Pourbaix stability.
5. Use grouped cross-validation by reduced composition to test extrapolation more honestly.
