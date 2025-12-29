from pathlib import Path

import pandas as pd

MIN_AGE = 15


def load_patients(eicu_root: Path):
    """
    Get the patients icu stays.

    Parameters
    ----------
    eicu_root : Path
        The path to the root of the eicu dataset.

    Returns
    -------
    pd.Dataframe
        The patients features.
    """
    icustays_df = pd.read_csv(eicu_root / "patient.csv.gz", compression="gzip")
    print(f"Loaded {len(icustays_df)} icu stays\n")

    variables = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "gender",
        "age",
        "ethnicity",
        "admissionweight",
        "hospitaladmitoffset",
        "hospitaldischargeoffset",
        "unitadmittime24",
        "unitdischargetime24",
    ]

    icustays_df = icustays_df[variables]

    # Convert gender
    icustays_df["gender"] = icustays_df["gender"].apply(
        lambda x: "F" if x == "Female" else "M"
    )

    # Convert age, cap at 89
    icustays_df["age"] = pd.to_numeric(
        icustays_df["age"].apply(lambda x: 89 if x == "> 89" else x)
    )

    # Filter out pediatric
    icustays_df = icustays_df[icustays_df["age"] >= MIN_AGE]

    icustays_df["unitadmittime24"] = pd.to_timedelta(icustays_df["unitadmittime24"])
    icustays_df["unitdischargetime24"] = pd.to_timedelta(
        icustays_df["unitdischargetime24"]
    )

    return icustays_df

    # Calculate ICU length of stay in days
    icustays_df["los"] = (
        icustays_df["unitdischargetime24"] - icustays_df["unitadmittime24"]
    ).dt.total_seconds() / (24 * 3600)

    # Filter less than 12 h or more than 10 d
    return filter_invalid_stays(icustays_df, 0.5, 10)


def filter_invalid_stays(icustays_df, min_days: float, max_days: float):
    """
    Filter icustays dataframe to remove invalid length stays.

    Parameters
    ----------
    icustays_df : pd.DataFrame
        The ICU stays dataframe.
    min_days : float
        Minimum length of the ICU stay in days.
    max_days : float
        Maximum length of the ICU stay in days.

    Returns
    -------
    pd.DataFrame
        The ICU stays filtered for invalid lengths.
    """
    print("Filtering patients with invalid stays...")
    icustays_df = icustays_df[
        (icustays_df["los"] >= min_days) & (icustays_df["los"] <= max_days)
    ]
    print(f"Filtered. {len(icustays_df)} rows left\n")
    return icustays_df
