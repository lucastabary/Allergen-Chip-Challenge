# Allergen Chip Challenge: allergy severity factors

TELECOM Nancy hackathon project on the data.gouv.fr challenge
[*Identifier et prévoir les facteurs de sévérité des allergies*](https://defis.data.gouv.fr/defis/identifier-et-prevoir-les-facteurs-de-severite-des-allergies)
(Health Data Hub × Société Française d'Allergologie).

**Question:** how can the IgE profiles and clinical data of the
[Allergen Chip Challenge (ACC) database](https://www.data.gouv.fr/datasets/allergen-chip-challenge)
be used to define allergic profiles, identify severity factors, and relate specific allergens to allergic diseases?

## Data

4,271 patients (mostly hospitalised, often with severe food allergies) × 256 columns:

- 15 demographic and clinical columns (age, sex, region, chip type, treatments, symptoms, cofactors);
- 241 specific IgE levels, one per molecular allergen (WHO/IUIS names such as `Ara_h_2`, `Bet_v_1`).

Three chips were used (ISAC v1, ISAC v2, ALEX²). They do not test the same allergens, so comparisons must take `Chip_Type` into account.
Data under [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/). It is not stored in this repository.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync               # create .venv and install dependencies
uv run acc-download   # fetch the CSV and the two dictionaries into data/raw/
uv run pytest         # sanity checks on the loaded data
```
Then, in Python or a notebook:

```python
from acc.data import load_raw, split_columns

df = load_raw()
clinical_cols, ige_cols = split_columns(df)
```

## Interactive page

`reports/atlas/index.html` (built with `uv run acc-atlas`, about 10 minutes) is a self-contained page
that follows the three parts of the challenge question:

1. **Allergic profiles.** K-means (k = 8, stable: ARI 0.75) on the IgE profile only: for each of the
   18 allergen groups, positive or not and its intensity (max IgE on the patient's own chip: low
   0.3-1, moderate 1-15, high >= 15). Each profile is described by its allergens (heatmap) and by its
   pathologies and severity, with odds ratios against the "barely sensitized" profile at equal
   centre, age and sex (Benjamini-Hochberg).
2. **From the IgE profile to pathologies.** Pick an intensity per group (or load a real patient and
   compare with what was observed): one logistic regression per pathology (asthma, rhinitis, atopic
   dermatitis treated or severe, food allergy) returns a probability and the nearest profile. Each
   row shows three F1: stratified CV (same centres), leave-one-centre-out (a new centre), and the
   age/sex/chip-only baseline.
3. **Severity factors.** Centre-adjusted odds ratios (fixed effects) of a severe form or a treatment,
   three views: one group at a time, all 18 groups in one penalized model (bootstrap CI), and by
   intensity level (dose-response).
4. **What clinical answers alone can estimate** (secondary): the earlier simulator from age,
   symptoms and treatments to the probability of sensitization per group, with the severity degree
   of the diseases left unanswered.
5. **Model scores** of that simulator, including the new-centre F1 and the ceiling experiments.
6. **Allergens.** The 241 IgE columns with their group and positive rate per chip; 235 are grouped,
   6 are excluded with a reason.

The models run in the browser from exported coefficients. There is no severity label: severity is
the treatment step (GINA, ARIA, systemic treatment for atopic dermatitis). The allergy diagnosis
labels used by the 2023 competition (winner F1 about 0.8) are not in the open file, so our F1 are on
other targets and not comparable.

## Structure

```
docs/           challenge summary
src/acc/        shared code (loading, allergen names and families, features, models, profiles, page)
notebooks/      exploration and analyses
data/raw/       downloaded files (git-ignored)
data/processed/ derived tables (git-ignored)
reports/        figures and final deliverables (reports/atlas/index.html)
tests/          tests
```
