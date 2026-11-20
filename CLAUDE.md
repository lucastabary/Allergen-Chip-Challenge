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

Full summary of the challenge (context, question, data warnings, suggested deliverables):
@docs/challenge.md

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
  - `allergens.py`: IgE column -> source family (`family_of()`, `sort_by_family()`: pollen, food...)
    and WHO/IUIS info (`iuis_table()`, `display_name()`, `protein_family_of()`, `describe()`).
    Protein families (PR-10, nsLTP, 2S albumin, tropomyosin...) are derived by regex from the IUIS
    biochemical names; PCA showed they drive co-sensitization. Whole extracts get `unknown`.
  - `resources/iuis_allergens.tsv`: extract of the WHO/IUIS database (allergen.org) for our 241 columns,
    with an `origin` column (`iuis`, `iuis_species`, `manual`). allergen.org blocks scripted downloads,
    so this extract is committed; cite https://www.allergen.org when using it.
  - `features.py`: `common_allergens()` (the 91 columns on all chips), `positivity()`, the clinical
    codebook (`CLINICAL_FIELDS`, French labels), `encode_clinical()` (44 one-hot inputs) and the 18
    `TARGET_GROUPS`: `allergen_group()` maps 235 of the 241 columns to a group (food split by protein
    family), `EXCLUDED_ALLERGENS` lists the other 6 with a reason, `sensitization_targets()` uses all
    the group's allergens on each patient's own chip (NaN when the chip has none: venoms on ISAC v2).
  - `features.py` also has `group_intensity()` (level 0-3 per group: max IgE on the own chip, bounds
    `INTENSITY_BOUNDS` 0.3/1/15), `food_allergy()` and `encode_intensity()` (68 inputs: 18 groups x 3
    levels, age band, sex, chip).
  - `modeling.py`: shared logistic-regression helpers (F1-max threshold, nested stratified CV
    `evaluate_group()`, leave-one-centre-out `evaluate_new_centre()`).
  - `profiles.py`: the IgE-first analyses. `allergic_profiles()` (k-means k=8 on flags + intensity,
    ARI stability, pathologies with centre-adjusted ORs vs the barely-sensitized profile, and
    `profile_allergen_rates()`: positive rate of each common allergen per profile, drawn as bars),
    `severity_factors()` (per group, joint penalized model with bootstrap CI, dose by intensity
    level), `fit_profile_models()` (IgE profile -> 7 pathology outcomes, F1 same centres and new
    centre, age/sex baseline).
  - `atlas.py`: builds the deliverable, a self-contained HTML page organized as a narrative by
    algorithm (data processing, k-means, logistic regression, multinomial regression, centre fixed
    effects, then a severity simulator; annexes: allergen list, clinical-only simulator, its scores
    and ceiling experiments). `fit_degree_target()` is a multinomial model of a disease's treatment
    level from the chip result (54 intensity cases) and the OTHER clinical answers + age/sex/chip,
    also scored with clinical only, chip only and age/sex/chip. Clinical-only sensitization models
    (`fit_group`) and `ceiling_experiments()` feed the annexes. Run `uv run acc-atlas` (about 10 min).
  - `resources/atlas_template.html`: page template (French UI); data is inlined at the
    `/*__ATLAS_DATA__*/null` marker. Edit the template, never the generated page.
  - `predict.py`: clinical targets built from the coded columns (`TARGETS`, with a positive control
    `adult` and a negative control `shuffled`), nested feature sets (`demo`, `pfam` = positives per
    protein family, `summary`, `ige` = 91 allergens common to the 3 chips), repeated stratified CV with
    logistic regression and gradient boosting, Nadeau-Bengio corrected t-tests + BH.
    `python -m acc.predict run --out runs/x --repeats 10` then `report runs/x`.
  - `experiments.py`: `screen` (one adjusted logistic per target x allergen, with and without
    adjustment for polysensitization), `importance` (permutation importance under CV), `transport`
    (leave-one-region-out).
  - `score.py`: storage-protein score for food anaphylaxis (number of positive 2S albumins + 11S
    legumins on the common panel): risk by level, threshold metrics with bootstrap CIs, CV calibration
    (intercept, slope, ECE), leave-one-region-out calibration. `python -m acc.score --out runs/score`.
- `relay.toml` + `scripts/chav.sh`: run the heavy jobs on Chav, a remote 12-CPU machine driven by the
  `relay` agent of Lucas's priism project (reference: `C:\Users\lucas\Projects\priism-cleave\docs\chav-agent.md`).
  `bash scripts/chav.sh push <branch>` ships a committed branch (signed bundle) and runs the tests there;
  `bash scripts/chav.sh job MAXMIN runs/x acc/predict run --out runs/x ...` submits a CPU job and pulls
  its results into `data/processed/runs/x`. Jobs run one at a time; never push while an acc job runs.
  Only the commands declared in `relay.toml` can run. The data is downloaded on the machine.
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
- `General_cofactors` has no code 9: `0` mixes "none known" and "not provided". Its code `10`
  ("pet rodent" in the codebook) is carried by 918 patients, almost all from four centres: treated as
  a centre artefact and kept out of the models.
- Treatment-based severity is centre-driven: 83 of the 96 severe atopic dermatitis patients come
  from centre XPW (`centre_of()`: letters of Patient_ID). Always check a severity result with
  leave-one-centre-out validation or centre fixed effects.
- `Age_of_onsets` (age at food-allergy onset) is binned in the source data, the last bin being
  "after 20 years" (150 training patients, aged 21 to 84): it cannot be refined.
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
