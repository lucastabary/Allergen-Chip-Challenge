# The challenge

Summary of the data.gouv.fr challenge this project answers. Source (in French):
https://defis.data.gouv.fr/defis/identifier-et-prevoir-les-facteurs-de-severite-des-allergies
(season 4, led by the Health Data Hub and the Société Française d'Allergologie).

## Context

- Allergy is the 4th chronic disease worldwide (WHO). About 1 in 3 people in France is allergic, possibly
  50 % by 2050. Food allergies have doubled in ten years (3.5 % of adults, 8 % of children).
- Diagnosis is slow (about 7 years on average) and allergists are scarce.
- Multiplex IgE chips measure specific IgE against hundreds of allergen molecules from one blood sample,
  but the resulting profile is hard to interpret. AI could help.

## The Allergen Chip Challenge (ACC), 2021–2023

- Led by the SFA with the Health Data Hub, funded by Bpifrance. Published in January 2026 in the
  Journal of Allergy and Clinical Immunology ("Allergen Chip Challenge: A nationwide open database
  supporting allergy prediction algorithms").
- 4,271 patients from 12 French hospital allergy labs (AllergoBioNet network): Bordeaux, Lille, Lyon,
  Marseille, Toulouse, Trousseau, Bichat, Caen, Reims, Clermont-Ferrand, Dijon, Nîmes. Retrospective,
  2012–2023. Mostly hospitalised patients, many with severe food allergy.
- Original competition (May–July 2023): 292 participants, 3,135 submissions, predict the presence of an
  allergic disease and its severity from the IgE profile. The winner reached F1 ≈ 0.79 (good on
  respiratory allergy) but overestimated severity and did poorly on food and venom allergies.

## The question of this challenge

How can the ACC data be used to **define allergic profiles**, **identify severity factors** (intensity,
associations, patient groups) and understand **how IgE profiles help predict severity** (link specific
allergens to the associated allergic diseases)?

Expected impact: better care for complex patients, visibility for the ACC database, new ways of reading it.

## Data notes from the organisers

- One CSV, 4,271 rows × 256 columns: 15 demographic/clinical columns, then 241 IgE columns (one per
  allergen, kUA/L; high value = sensitization). Allergen names follow WHO/IUIS (https://www.allergen.org).
  Six source families: pollens (trees, grasses, weeds), mites and cockroaches, pets, foods, molds and
  yeasts, hymenoptera venoms.
- Dictionaries: `dictionnaire-acc-english.pdf` (only yellow variables are in the data) and
  `acc-dictionnaire-final.xls` (category codes, e.g. `General_cofactors`).
- Warnings from the organisers:
  - about 1,300 patients (30 %) have partial or missing clinical data;
  - the 3 chip types do not measure the same allergens, so comparisons must include `Chip_Type`;
  - detailed clinical diagnoses (light blue variables) are not in the data.
- Clinical and biological data do not have to be used together.

## Suggested deliverables (free choice)

Four data-analysis tracks and one modelling track:

1. **Clustering** of patients on IgE and clinical data (k-means, DBSCAN, hierarchical clustering).
   Examples: a mono/poly-sensitization index on pollen markers (grasses `Phl_p_1`, `Phl_p_5`, `Lol_p_1`;
   trees `Bet_v_1`, `Ole_e_1`, `Cup_a_1`; weeds `Art_v_1`, `Amb_a_1`, `Par_j_2`); clusters on mites
   (`Der_p_1`, `Der_p_2`, `Der_f_1`, `Blo_t_5`) and molds (`Alt_a_1`, `Asp_f_1`, `Cla_h_8`) crossed with
   `Treatment_of_athsma`.
2. **PCA** of the 241 IgE variables with clinical variables as supplementary variables. Examples: color
   patients by `French_Region`; project `Skin_Symptoms`, `Treatment_of_rhinitis`, `Treatment_of_athsma`.
3. **Correlation matrices**: co-sensitization between IgE (e.g. within mites: `Der_p_1`, `Der_p_2`,
   `Der_p_23`, `Der_f_1`, `Blo_t_5`, `Lep_d_2`) and IgE–clinical associations (e.g. bee `Api_m_1`,
   `Api_m_4` vs wasp `Ves_v_5`, `Pol_d_5` double sensitization crossed with `General_cofactors`).
4. **Simple cross-tabulations and dataviz**: e.g. molds or high-pollen living areas vs skin allergy;
   less conjunctivitis but more skin allergy; regional map of pollen sensitization.
5. **Statistical modelling (econometrics)**, transparent and explanatory. Examples: confirmed
   sensitization ~ `Age` + `Gender` + `French_Region` + `Chip_Type`; presence of at least one cofactor ~
   venom IgE (`Api_m_1`, `Ves_v_5`) + major food allergens (`Ara_h_2`, `Jug_r_1`) to explore
   anaphylaxis risk profiles.

## Publishing

Results can be published as a "réutilisation" on data.gouv.fr and may be showcased during the
Mois de l'open data. Contacts at the Health Data Hub are listed on the challenge page.
