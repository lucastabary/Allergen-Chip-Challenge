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

## Structure

```
docs/           challenge summary
src/acc/        shared code (loading, allergen names and families, features)
notebooks/      exploration and analyses
data/raw/       downloaded files (git-ignored)
data/processed/ derived tables (git-ignored)
reports/        figures and final deliverables
tests/          tests
```
