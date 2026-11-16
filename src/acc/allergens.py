"""Map IgE columns (WHO/IUIS allergen names) to their source and allergen family.

Column names start with the source species code, e.g. `Der_p_2` -> `Der_p`
(Dermatophagoides pteronyssinus, a house dust mite). Columns without a number
(`Cyn_d`, `Fra_e`...) are whole extracts measured on ALEX.

Reference for the codes: https://www.allergen.org
"""

from __future__ import annotations

import re

GRASS = "grass pollen"
TREE = "tree pollen"
WEED = "weed pollen"
MITE = "mites"
INSECT = "insects"
ANIMAL = "animal dander"
MOLD = "molds / yeasts"
VENOM = "venoms"
LATEX = "latex"
FOOD_PLANT = "plant food"
FOOD_ANIMAL = "animal food"
OTHER = "other"

FAMILIES = [
    GRASS,
    TREE,
    WEED,
    MITE,
    INSECT,
    ANIMAL,
    MOLD,
    VENOM,
    LATEX,
    FOOD_PLANT,
    FOOD_ANIMAL,
    OTHER,
]

# source code -> (common name, family)
SOURCES: dict[str, tuple[str, str]] = {
    # grass pollen
    "Phl_p": ("timothy grass", GRASS),
    "Lol_p": ("ryegrass", GRASS),
    "Lol_spp": ("ryegrass", GRASS),
    "Cyn_d": ("Bermuda grass", GRASS),
    # tree pollen
    "Bet_v": ("birch", TREE),
    "Aln_g": ("alder", TREE),
    "Fag_s": ("beech", TREE),
    "Cry_j": ("Japanese cedar", TREE),
    "Cup_a": ("Arizona cypress", TREE),
    "Cup_s": ("Italian cypress", TREE),
    "Jun_a": ("mountain cedar", TREE),
    "Ole_e": ("olive tree", TREE),
    "Fra_e": ("ash tree", TREE),
    "Pla_a": ("plane tree", TREE),
    "Mor_r": ("mulberry", TREE),
    "Bro_pa": ("paper mulberry", TREE),
    "Ail_a": ("tree of heaven", TREE),
    # weed pollen
    "Amb_a": ("ragweed", WEED),
    "Art_v": ("mugwort", WEED),
    "Par_j": ("wall pellitory", WEED),
    "Pla_l": ("plantain", WEED),
    "Sal_k": ("saltwort", WEED),
    "Che_a": ("goosefoot", WEED),
    "Ama_r": ("pigweed", WEED),
    "Mer_a": ("annual mercury", WEED),
    # mites
    "Der_p": ("house dust mite (D. pteronyssinus)", MITE),
    "Der_f": ("house dust mite (D. farinae)", MITE),
    "Lep_d": ("storage mite (Lepidoglyphus)", MITE),
    "Blo_t": ("storage mite (Blomia)", MITE),
    "Aca_s": ("storage mite (Acarus)", MITE),
    "Gly_d": ("storage mite (Glycyphagus)", MITE),
    # insects
    "Bla_g": ("cockroach", INSECT),
    "Loc_m": ("locust", INSECT),
    "Ach_d": ("house cricket", INSECT),
    "Chi_spp": ("chironomid midge", INSECT),
    "Arg_r": ("pigeon tick", INSECT),
    # animal dander
    "Fel_d": ("cat", ANIMAL),
    "Can_f": ("dog", ANIMAL),
    "Equ_c": ("horse", ANIMAL),
    "Mus_m": ("mouse", ANIMAL),
    "Cav_p": ("guinea pig", ANIMAL),
    "Ory_c": ("rabbit", ANIMAL),
    "Mes_a": ("hamster", ANIMAL),
    "Cap_h": ("goat", ANIMAL),
    "Cam_d": ("camel", ANIMAL),
    # molds / yeasts
    "Alt_a": ("Alternaria", MOLD),
    "Asp_f": ("Aspergillus", MOLD),
    "Cla_h": ("Cladosporium", MOLD),
    "Mala_s": ("Malassezia", MOLD),
    # venoms
    "Api_m": ("honey bee", VENOM),
    "Ves_v": ("common wasp", VENOM),
    "Pol_d": ("paper wasp", VENOM),
    "Dol_spp": ("hornet (Dolichovespula)", VENOM),
    # latex
    "Hev_b": ("latex", LATEX),
    # plant food
    "Ara_h": ("peanut", FOOD_PLANT),
    "Gly_m": ("soy", FOOD_PLANT),
    "Jug_r": ("walnut", FOOD_PLANT),
    "Cor_a": ("hazelnut", FOOD_PLANT),
    "Ana_o": ("cashew", FOOD_PLANT),
    "Ber_e": ("Brazil nut", FOOD_PLANT),
    "Car_i": ("pecan", FOOD_PLANT),
    "Mac_i": ("macadamia", FOOD_PLANT),
    "Mac_inte": ("macadamia", FOOD_PLANT),
    "Ses_i": ("sesame", FOOD_PLANT),
    "Fag_e": ("buckwheat", FOOD_PLANT),
    "Tri_a": ("wheat", FOOD_PLANT),
    "Hor_v": ("barley", FOOD_PLANT),
    "Ave_s": ("oat", FOOD_PLANT),
    "Act_d": ("kiwi", FOOD_PLANT),
    "Api_g": ("celery", FOOD_PLANT),
    "Mal_d": ("apple", FOOD_PLANT),
    "Pru_p": ("peach", FOOD_PLANT),
    "Fra_a": ("strawberry", FOOD_PLANT),
    "Dau_c": ("carrot", FOOD_PLANT),
    "Cuc_m": ("melon", FOOD_PLANT),
    "Cuc_p": ("pumpkin seed", FOOD_PLANT),
    "Cit_s": ("orange", FOOD_PLANT),
    "Mus_a": ("banana", FOOD_PLANT),
    "Man_i": ("mango", FOOD_PLANT),
    "Car_p": ("papaya", FOOD_PLANT),
    "Cap_a": ("bell pepper", FOOD_PLANT),
    "All_c": ("onion", FOOD_PLANT),
    "All_s": ("garlic", FOOD_PLANT),
    "Len_c": ("lentil", FOOD_PLANT),
    "Cic_a": ("chickpea", FOOD_PLANT),
    "Lup_a": ("lupine", FOOD_PLANT),
    "Che_q": ("quinoa", FOOD_PLANT),
    "Hel_a": ("sunflower seed", FOOD_PLANT),
    "Ori_v": ("oregano", FOOD_PLANT),
    "Fic_c": ("fig", FOOD_PLANT),
    # animal food
    "Gal_d": ("chicken egg", FOOD_ANIMAL),
    "Bos_d": ("cow's milk", FOOD_ANIMAL),
    "Gad_c": ("cod", FOOD_ANIMAL),
    "Gad_m": ("cod", FOOD_ANIMAL),
    "Cyp_c": ("carp", FOOD_ANIMAL),
    "Clu_h": ("herring", FOOD_ANIMAL),
    "Pen_m": ("shrimp", FOOD_ANIMAL),
    "Lit_s": ("shrimp", FOOD_ANIMAL),
    "Cra_c": ("shrimp", FOOD_ANIMAL),
    "Hom_g": ("lobster", FOOD_ANIMAL),
    "Car_c": ("crab", FOOD_ANIMAL),  # assumed Carcinus maenas
    "Myt_e": ("mussel", FOOD_ANIMAL),
    "Ani_s": ("Anisakis (fish parasite)", FOOD_ANIMAL),
    "Mel_g": ("turkey meat", FOOD_ANIMAL),
    "Alpha-Gal": ("alpha-gal (mammalian meat)", FOOD_ANIMAL),
    # other
    "MUXF3": ("CCD (cross-reactive carbohydrate)", OTHER),
    "Hom_s": ("human lactoferrin", OTHER),
    "Fic_b": ("weeping fig (plant)", OTHER),
    "Can_s": ("hemp", OTHER),
    "Aca_m": ("unidentified", OTHER),
}

# Columns whose family differs from their source's default.
OVERRIDES: dict[str, str] = {
    "Cor_a_1.0101": TREE,  # hazel pollen PR-10
    "Cor_a_1.0103": TREE,
    "Cor_a_pollen": TREE,
    "Cor_a_12_RUO": FOOD_PLANT,
    "Jug_r_pollen": TREE,
    "Equ_c_meat": FOOD_ANIMAL,
    "Equ_c_milk": FOOD_ANIMAL,
    "Cap_h_milk": FOOD_ANIMAL,
    "Bos_d_2": ANIMAL,  # cow dander
}


def source_of(column: str) -> str:
    """Source code of an IgE column, e.g. 'Der_p_2' -> 'Der_p'."""
    if column in SOURCES:
        return column
    match = re.match(r"^([A-Za-z]+_[a-z]+)", column)
    return match.group(1) if match else column


def family_of(column: str) -> str:
    """Allergen family of an IgE column (one of FAMILIES)."""
    if column in OVERRIDES:
        return OVERRIDES[column]
    return SOURCES[source_of(column)][1]


def sort_by_family(columns: list[str]) -> list[str]:
    """Order columns by family (FAMILIES order), then by source and name."""
    return sorted(columns, key=lambda c: (FAMILIES.index(family_of(c)), source_of(c), c))
