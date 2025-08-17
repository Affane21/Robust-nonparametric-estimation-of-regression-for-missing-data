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

## References
- **Affane, I. (2024)**.
  *Estimation non paramétique robuste de la régression pour des données manquantes*.
  [Download thesis (PDF).](<docs/Estimation non paramétrique robuste de la régression pour des données manquantes.pdf>)

- **Boente, G., González-Manteiga, w., & perez-González, A. (2009)**.
  *Robust nonparametric estimation with missing data*.
  Journal of Statistical Planning and Inference, 139(2), 571-592.
  [DOI:10.1016/j.jspi.2008.02.019](https://doi.org/10.1016/j.jspi.2008.02.019)