# Robust Nonparametric Regression for Missing Data
> **Implementation of SLMS and ILMS estimators for robust nonparmetric regression under MAR mechanism.**
> Based on my Master's thesis and the work of Boente et al. (2009).

## overview
two estimators are implemented:
- **SLMS**: uses only observed data with robust M-estimation.
- **ILMS**: Imputes missing values using SLMS, then re-smooths.

Designed to accutately estimate a regression function when:
- Data is **missing at random (MAR)**.
- the response variable contains **outliers (contamination)**.
- the relationship between variables is **nonlinerar**.
## methodology 
### SLMS (Simpligied Local M-Smother)
Uses only observed data with robust M-estimation and kernel weight to resist outliers.

### ILMS (Imputed Local M-Smother)
First, imputes missing responses using SLMS, then applies a second robust smoother on the completed dataset.

Both methods are consistent and asymptotically normal under MAR and contamination.

## Simulation Design

The simulation will reproduce the setup from chapter 4 of my thesis:
- $ y_i = 0.25 \sin(\pi x_i) + \epsilon_i $
- $ \epsilion_i $: mixture of $ \mathcal{N}(0, \sigma^2) $ and $ \mathcal{N}(0, 25\sigma^2) $
- missingness mechanism: $ p(x) = 0.3 + 0.5 \sin^2(5(x + 0.2)) $ (MAR)

Performance will be evaluated using **MISE** (Mean Integrated Squared Error) at contamination levels $ \eta = 0\%, 10\%, 20\%$.

## References
- **Affane, I. (2024)**.
  *Estimation non paramétique robuste de la régression pour des données manquantes*.
  [Download thesis (PDF).](<docs/Estimation non paramétrique robuste de la régression pour des données manquantes.pdf>)

- **Boente, G., González-Manteiga, w., & perez-González, A. (2009)**.
  *Robust nonparametric estimation with missing data*.
  Journal of Statistical Planning and Inference, 139(2), 571-592.
  [DOI:10.1016/j.jspi.2008.02.019](https://doi.org/10.1016/j.jspi.2008.02.019)

## Project Structure
RobustNonparametricRegression/
├── src/
│ ├── data_simulation.py # Generate data and MAR mechanism
│ ├── slms.py # Simplified Local M-Smoother
│ ├── ilms.py # Imputed Local M-Smoother
│ └── evaluation.py # Compute MISE, generate plots
├── notebooks/
│ └── full_simulation.ipynb # End-to-end simulation
├── results/
│ ├── plots/ # Estimation curves, MISE comparison
│ └── metrics.csv # Performance metrics
├── docs/
│ └── Estimation non paramétrique robuste de la régression pour des données manquantes.pdf
├── README.md
└── requirements.txt

## Results

After 200 replications, the Mean Integrated Squared Error (MISE) is:

| Estimator | MISE |
|---------|------|
| SLMS | 0.0171 |
| ILMS | 0.0203 |
| Nadaraya-Watson | 0.0429 |

> **Robust estimators (SLMS/ILMS) outperform the classical Nadaraya-Watson under contamination and MAR.**
> **SLMS achieves the best performance.**

![Comparison](results/plots/comparison_all.png)