# CLAUDE.md

## Purpose

Team project (3 people) for the TELECOM Nancy hackathon on the data.gouv.fr challenge
"Identifier et prévoir les facteurs de sévérité des allergies"
(https://defis.data.gouv.fr/defis/identifier-et-prevoir-les-facteurs-de-severite-des-allergies).
The goal is to use the Allergen Chip Challenge (ACC) database to build allergic profiles, find severity
factors, and link IgE profiles to clinical outcomes. The deliverable is open: clustering, PCA, correlation
analyses, dataviz, and interpretable statistical models are all suggested. Results may be published as a
"réutilisation" on data.gouv.fr. Interpretability matters more than leaderboard performance.

Background: the original ACC competition (2023) winner reached F1 ≈ 0.79 for allergy presence, but
overestimated severity and did poorly on food and venom allergies.

## Environment and commands

- Python 3.12, managed with `uv` (`pyproject.toml` + `uv.lock`, both committed).
- `uv sync`: install everything (dev group included: ipykernel, pytest, ruff).
- `uv run acc-download`: download the 3 raw files from data.gouv.fr into `data/raw/` (SHA-1 checked).
- `uv run pytest`: tests (data tests are skipped if the data is not downloaded).
- `uv run ruff check . && uv run ruff format .`: lint and format.

## Structure

- `src/acc/`: installable package with the shared code. Everything that several notebooks need
  (loading, cleaning, feature engineering) goes here so the three of us share one pipeline.
  - `data.py`: download, `load_raw()`, `split_columns()`.
- `notebooks/`: exploration. Name them `NN_initials_topic.ipynb` (e.g. `01_lt_eda.ipynb`).
- `data/raw/`, `data/processed/`: git-ignored, never commit data.
- `reports/figures/`: exported figures and deliverables.
- `tests/`: pytest.

## Data facts (verified)

- CSV is French-style: `;` separator, `,` decimal, UTF-8 with BOM. `load_raw()` handles this.
- 4,271 rows × 256 columns: 15 clinical columns, then 241 IgE columns (float, kUA/L for ALEX, ISU for ISAC).
- Clinical columns are read as **strings** because some are multi-coded (`"1,3"` = codes 1 and 3,
  `"6, 7"` with a space also occurs). Never parse them with a global decimal conversion.
- 52 IgE values are exactly `-1` (36 patients, 35 on ISAC v2): impossible concentrations, so `load_raw()`
  turns them into NaN. These are the only NaN inside a chip's own allergen set.
- Code `9` means "not provided" (`Non renseigné`) in the clinical categorical columns.
- `Chip_Type`: ISAC_V1 (2,351), ALEX (1,139), ISAC_V2 (781). ISAC chips cover 112 allergens, ALEX 223,
  so a NaN in an IgE column is usually structural (allergen not on that chip), not a missing measurement.
- Codebooks: `data/raw/acc-dictionnaire-final.xls` (French, category codes) and
  `data/raw/dictionnaire-acc-english.pdf` (only the yellow variables exist in the data; blue diagnostic
  variables were removed).

## Analysis conventions

- No severity label exists: any severity target is a constructed proxy (treatments, cofactors, symptoms).
  Document how it is built.
- Always account for `Chip_Type` (stratify or include it); do not impute across chips naively.
- IgE levels are zero-inflated and heavy-tailed: use `log1p` or positivity thresholds
  (0.3 kUA/L for ALEX, 0.3 ISU for ISAC) before PCA or clustering.
- Correct for multiple testing (BH/FDR) when screening many allergens.
- Population is hospital-based with severe food allergies: no claim about the general population.
- Report results over several seeds or with confidence intervals, and use sensible baselines.

## Conventions

- Code, comments, docs and commits in English. Conventional Commits (`feat:`, `fix:`, `docs:`...).
- Feature branches (`feat/topic`) for non-trivial work; keep notebooks light (clear large outputs).
