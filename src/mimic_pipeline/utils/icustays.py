from pathlib import Path

import pandas as pd


def load_icustays(mimic_root: Path) -> pd.DataFrame:
    """
    Load the icu stays from the MIMIC IV 2.2 dataset.

    Parameters
    ----------
    mimic_root : Path
        The path to the root of the mimic dataset.

    Returns
    -------
    pd.Dataframe
        The icu stays after filtering invalid stays.
    """
    icustays_df = pd.read_csv(
        mimic_root / "icu" / "icustays.csv.gz", compression="gzip"
    )
    print(f"Loaded {len(icustays_df)} icu stays\n")

    # Process time columns
    icustays_df["intime"] = pd.to_datetime(icustays_df["intime"])
    icustays_df["outtime"] = pd.to_datetime(icustays_df["outtime"])
    icustays_df["icu_year"] = icustays_df["intime"].dt.year

    # Filter stays that are not between 12h and 10 days
    return filter_invalid_stays(icustays_df, 0.5, 10)


def filter_invalid_stays(icustays_df: pd.DataFrame, min_days: float, max_days: float):
    """
    Filter icustays dataframe to remove invalid length stays.

    Parameters
    ----------
    icustays_df : pd.DataFrame
        The path to the root of the mimic dataset.
    min_days : float
        Minimum length of the icu_stay in days.
    max_days : float
        Maximum length of the icu_stay in days.

    Returns
    -------
    pd.Dataframe
        The icu stays filtered invalid stays
        of less than 12h or more than 10 days.
    """
    print("Filtering patients with invalid stays...")
    icustays_df = icustays_df[
        (icustays_df["los"] >= min_days) & (icustays_df["los"] <= max_days)
    ]
    print(f"Filtered. {len(icustays_df)} rows left\n")
    return icustays_df
