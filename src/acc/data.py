"""Download and load the ACC dataset published on data.gouv.fr.

Usage:
    uv run acc-download          # fetch the raw files into data/raw/
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

_BASE = "https://static.data.gouv.fr/resources/allergen-chip-challenge"

# file name -> (url, sha1) as published by the data.gouv.fr API
# (dataset id 669f768780d3cf0ea54d518b, last updated 2024-07-26).
FILES: dict[str, tuple[str, str]] = {
    "allergenchipchallenge-data-corrected-final-hdh-sfa.csv": (
        f"{_BASE}/20240726-152416/allergenchipchallenge-data-corrected-final-hdh-sfa.csv",
        "ef46a4962c85c14f6f2939ced3ed0693d0a7584f",
    ),
    "dictionnaire-acc-english.pdf": (
        f"{_BASE}/20240726-134542/dictionnaire-acc-english.pdf",
        "365c386a425f1bfe35a7420d2683967bc23e1a1a",
    ),
    "acc-dictionnaire-final.xls": (
        f"{_BASE}/20240726-134443/acc-dictionnaire-final.xls",
        "b8c744c327032ff1ff40832f4921d1527862c2f4",
    ),
}

DATA_FILE = RAW_DIR / "allergenchipchallenge-data-corrected-final-hdh-sfa.csv"

# The first 15 columns are demographic/clinical, the remaining 241 are IgE levels.
N_CLINICAL_COLUMNS = 15


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def download(force: bool = False) -> None:
    """Download the raw files into data/raw/ and check their SHA-1 checksums."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, (url, sha1) in FILES.items():
        target = RAW_DIR / name
        if target.exists() and not force and _sha1(target) == sha1:
            print(f"ok       {name}")
            continue
        print(f"download {name}")
        urllib.request.urlretrieve(url, target)
        if _sha1(target) != sha1:
            raise RuntimeError(f"Checksum mismatch for {name}: the published file has changed.")


def load_raw() -> pd.DataFrame:
    """Load the ACC table as is (one row per patient)."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"{DATA_FILE} is missing, run `uv run acc-download` first.")
    # French-style CSV: ';' separator, ',' decimal mark, UTF-8 with BOM.
    # Everything is read as text first: some clinical columns are multi-coded
    # ("1,3" = categories 1 and 3), which a global decimal="," would turn into 1.3.
    df = pd.read_csv(DATA_FILE, sep=";", encoding="utf-8-sig", dtype=str)
    clinical, ige = split_columns(df)
    df[ige] = df[ige].apply(lambda s: pd.to_numeric(s.str.replace(",", ".")))
    df["Age"] = pd.to_numeric(df["Age"].str.replace(",", "."))
    return df.copy()  # defragment after the column-wise conversions


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (clinical columns, IgE columns)."""
    cols = list(df.columns)
    return cols[:N_CLINICAL_COLUMNS], cols[N_CLINICAL_COLUMNS:]


def main() -> None:
    download()


if __name__ == "__main__":
    main()
