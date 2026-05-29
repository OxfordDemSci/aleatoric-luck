# Aleatoric Luck

> **Oxford Leverhulme Centre for Demographic Science**

This repository contains the research code for the **aleatoric luck project** — an investigation into the role of irreducible randomness in shaping life outcomes. We replicate predictive models from Zheng & Cheng (2025), decompose their prediction error using learning curves and power-law fitting, and interpret the resulting aleatoric error component as an empirical measure of *luck* in predicting social and demographic outcomes.

---

## Overview

How much of what happens to people is genuinely unpredictable — not merely unmeasured, but fundamentally unknowable in advance? This project approaches that question empirically.

We replicate the machine learning models of **Zheng & Cheng (2025)**, who used large-scale survey data to predict individual life outcomes. We then fit **learning curves** — tracking how model error changes as training data grows — and apply **power-law fitting** to extrapolate the irreducible floor of prediction error as sample size approaches infinity. This limiting error, which cannot be reduced by adding more data or improving the model, is the *aleatoric* component: the portion of outcome variance that is structurally unpredictable given the available feature space.

We argue this aleatoric error is a principled, empirically grounded measure of **luck** — the genuine role of chance in determining outcomes such as educational attainment, occupational status, income, and other life course variables.

### Key research questions

- What is the irreducible lower bound of prediction error for a given life outcome, and how is it estimated?
- How does aleatoric error vary across outcome domains (e.g. education, income, health)?
- Which features account for the predictable component, and how does their contribution change with sample size?
- What does the residual aleatoric error tell us about the role of luck in social stratification?

---

## Methodology

The analysis proceeds in three stages:

**1. Model replication**
We replicate the predictive modelling framework of Zheng & Cheng (2025), training gradient-boosted models (XGBoost, LightGBM) to predict life outcomes from individual-level covariates.

**2. Learning curve estimation**
Models are retrained across a range of training set sizes, and prediction error (e.g. MSE or R²) is recorded at each size to construct a learning curve.

**3. Power-law decomposition**
The learning curve is fit to a power-law function of the form:

$$E(n) = E_{\infty} + A \cdot n^{-\alpha}$$

where $E(n)$ is error at training size $n$, $E_{\infty}$ is the asymptotic irreducible error (aleatoric), $A$ is a scaling constant, and $\alpha$ is the learning rate exponent. The fitted $E_{\infty}$ is our empirical estimate of aleatoric luck.

**SHAP analysis** is additionally used to characterise the predictable component — identifying which features drive model performance and how feature importance shifts across domains and training regimes.

---

## Repository Structure

```
aleatoric-luck/
├── notebooks/          # Exploratory and analytical Jupyter notebooks
├── src/                # Core Python source scripts
│   └── SHAP_vals.py    # Computes SHAP values from trained models
├── run_SHAP.sh         # SLURM batch script for SHAP computation (HPC)
├── run_SHAP_exp.sh     # SLURM batch script for expanded SHAP experiments
├── run_domain.sh       # SLURM batch script for domain-level analysis
├── run_feature.sh      # SLURM batch script for feature-level analysis
├── requirements.txt    # Python dependencies
└── output.png          # Example output figure
```

---

## Getting Started

### Prerequisites

Python 3.11 is recommended. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Running Locally

Exploratory analyses and visualisations are in the `notebooks/` directory and can be run interactively via Jupyter:

```bash
jupyter notebook notebooks/
```

The core SHAP value computation can be run directly:

```bash
python src/SHAP_vals.py
```

### Running on an HPC Cluster (SLURM)

For computationally intensive jobs, several SLURM submission scripts are provided. Set up a virtual environment first:

```bash
python -m venv ~/venvs/aleatoric-luck
source ~/venvs/aleatoric-luck/bin/activate
pip install -r requirements.txt
```

Then submit jobs as needed:

```bash
sbatch run_SHAP.sh          # Compute SHAP values
sbatch run_SHAP_exp.sh      # Expanded SHAP experiments
sbatch run_domain.sh        # Domain-level analysis
sbatch run_feature.sh       # Feature-level analysis
```

---

## Dependencies

Key packages (pinned versions in `requirements.txt`):

| Package | Version | Purpose |
|---|---|---|
| `xgboost` | 3.2.0 | Gradient boosted tree models |
| `lightgbm` | 4.6.0 | Fast gradient boosting |
| `scikit-learn` | 1.8.0 | ML utilities, learning curve tooling |
| `shap` | 0.51.0 | Feature attribution / interpretability |
| `scipy` | 1.17.1 | Power-law curve fitting |
| `pandas` | 3.0.2 | Data manipulation |
| `numpy` | 2.4.4 | Numerical computation |
| `matplotlib` / `seaborn` | 3.10.9 / 0.13.2 | Visualisation |

---

## Reference

> Zheng, H., & Cheng, S. (2025). *[Title]*. [Journal]. [DOI]

---

## Organisation

This project is maintained by the [Leverhulme Centre for Demographic Science](https://github.com/OxfordDemSci) at the University of Oxford.

---

## Contributing

This is an active research repository. If you have questions or suggestions, please open an issue on GitHub.

---

## Licence

Please contact the authors regarding reuse and licensing of this research code.
