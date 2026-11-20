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
meant to present the project. It is a narrative in steps; each step shows one algorithm on a small
example you can play with, then its result on the 4,271 patients:

0. **Data processing.** 241 IgE columns, positivity threshold 0.3 (manufacturer's, in each chip's
   unit), 235 allergens in 18 biological families (6 excluded), family intensity = highest IgE of
   the family on the patient's chip, in 4 classes (ISAC bounds 0.3 / 1 / 15).
1. **K-means** (unsupervised). Animated 2D demo, then the 8 real profiles (36 variables per patient,
   ARI 0.75 over 5 seeds) and their diseases with centre-adjusted odds ratios.
2. **Logistic regression.** The real "treated for asthma" model: checked cases, weights, score,
   sigmoid, threshold. Then F1 for 7 outcomes, same centres and new centre (leave-one-centre-out),
   against an age/sex/chip baseline also tested on a new centre.
3. **Multinomial regression.** The real asthma treatment-level model: 4 scores, softmax. The level
   models now use the chip result (54 intensity cases) plus the other clinical answers; macro F1 is
   compared with clinical only, chip only, age/sex/chip and the majority level.
4. **Fixed effects by centre.** Two fictitious hospitals show how mixing centres fakes an effect;
   then the forest plot of severity factors (one family, all families jointly, by intensity).
5. **Severity simulator.** Chip result + clinical answers give the nearest profile, the probability
   of each disease, the most probable treatment level of unanswered diseases, and the significant
   severity factors of the positive families.

Folded annexes: the 241 allergens, the clinical-only sensitization simulator with its scores and
the ceiling experiments (region, month, C, gradient boosting), method and limits.

There is no severity label: severity is the treatment step (GINA, ARIA, systemic treatment for
atopic dermatitis). The allergy diagnosis labels used by the 2023 competition (winner F1 about 0.8)
are not in the open file, so our F1 are on other targets and not comparable.

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
