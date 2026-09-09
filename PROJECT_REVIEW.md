# Project Review

## Complexity

The project combines several layers of materials-informatics work:

- data extraction and merging from reaction energy records;
- feature engineering from elemental properties and coordination environments;
- supervised regression for OER-relevant adsorption/reaction energy;
- pseudo-labelling to expand a small labelled dataset;
- inverse design with a genetic algorithm over discrete composition space.

The scientific complexity is moderate to high because OER catalyst activity depends strongly on surface structure, adsorption site, coordination, stability, and reaction environment. The current code captures composition and coordination trends, but it does not yet model full surface structures or electrochemical operating conditions.

The software complexity was originally high because most work lived in notebooks and one large `GA.py` script. The showcase copy separates reusable source code, scripts, data, results, assets, and exploratory notebooks.

## Accuracy

The cleaned benchmark script reproduces the main XGBoost result on the top feature set:

| Dataset | Samples | Test R² | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| True + pseudo-labelled | 1,406 | 0.8205 ± 0.0372 | 0.2069 ± 0.0103 eV | 0.2996 ± 0.0203 eV |
| True-only | 569 | 0.7335 ± 0.0369 | 0.2813 ± 0.0282 eV | 0.3903 ± 0.0417 eV |

This is a respectable result for a small project dataset. However, MAE around 0.2 eV is still large relative to the energy differences that often matter in catalyst ranking, so predicted candidates should be treated as screening hypotheses.

## Key Risks

- **Small-data regime:** many chemical spaces are underrepresented.
- **Pseudo-label bias:** pseudo-labels improve sample count but can reinforce the model used to generate them.
- **Candidate validity:** a formula close to the target prediction is not automatically stable, synthesizable, or active.
- **Surface representation:** elemental averages lose site-specific geometry and local atomic arrangement.
- **Model exploitation:** the GA can discover formulas that exploit model extrapolation rather than real chemistry.

## Recommended Next Work

1. Add DFT validation for the top 10-20 GA candidates, including O*, OH*, and ideally OOH* adsorption energies.
2. Add stability filters: formation energy, decomposition energy, Pourbaix stability, and phase-separation checks.
3. Expand the dataset with larger catalyst-surface sources such as OC20/OC22-style data or targeted DFT calculations.
4. Track uncertainty with XGBoost ensembles or conformal intervals and penalize high-uncertainty GA candidates.
5. Move from composition averages toward graph/structure descriptors for slabs and adsorbate-site environments.
6. Add automated report generation: parity plot, SHAP feature importance, candidate table, and convergence plot.

## GitHub Showcase Recommendation

For a personal GitHub project, the strongest narrative is:

> "I built an end-to-end catalyst screening workflow that goes from raw OER reaction-energy data to ML prediction and inverse composition search, then critically evaluated the limits of the surrogate model."

That framing is more credible than presenting the GA candidates as final discovered catalysts.
