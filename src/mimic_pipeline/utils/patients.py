from pathlib import Path

import pandas as pd

MIN_AGE = 15


def add_patient_features(
    mimic_root: Path, icustays_df: pd.DataFrame, rm_early_die=False
) -> pd.DataFrame:
    """
    Add patients features to the icu stays dataframe.

    Parameters
    ----------
    mimic_root : Path
        The path to the root of the mimic dataset.
    icustays_df : pd.Dataframe
        The icu stays dataframe.
    rm_early_die : bool
        To remove patients with death time during the stay.

    Returns
    -------
    pd.Dataframe
        The icu stays with patient features and filtering
        rows with age under 15, and rows with death time during the stay.
    """
    # Load patients info
    patients_df = pd.read_csv(
        mimic_root / "hosp" / "patients.csv.gz", compression="gzip"
    )

    # Get the info we care about from patients_df
    merged_df = icustays_df.merge(
        patients_df[["subject_id", "anchor_age", "anchor_year", "gender"]],
        on="subject_id",
        how="left",
    )
    del patients_df

    # Only keep patients that are at least 15 at the moment of admission to icu
    merged_df = filter_age_under_15(merged_df)

    # Add the patient features
    admission_df = pd.read_csv(
        mimic_root / "hosp" / "admissions.csv.gz", compression="gzip"
    )

    # Get the info we care about from admission_df
    merged_df = merged_df.merge(
        admission_df[["hadm_id", "deathtime", "marital_status", "race"]],
        on="hadm_id",
        how="left",
    )
    del admission_df

    # Filter those who died in the icu stay
    if rm_early_die:
        merged_df = filter_death_during_stay(merged_df=merged_df)
        # Filter those with multiple icu_stays in this admission
        merged_df = merged_df.drop(columns=["outtime", "deathtime"], axis=1)
        merged_df = filter_multiple_icu_stay_per_admission(merged_df=merged_df)
        return merged_df

    else:
        # Filter those with multiple icu_stays in this admission
        merged_df = filter_multiple_icu_stay_per_admission(merged_df=merged_df)

        # compute time to death
        merged_df = time_to_death(merged_df)
        merged_df = merged_df.drop(columns=["outtime", "deathtime"], axis=1)

        return merged_df


def filter_age_under_15(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where patient is under 15 years old.

    Parameters
    ----------
    merged_df : pd.Dataframe
        The icu stays dataframe with the patients features.

    Returns
    -------
    pd.Dataframe
        The icu stays with patient features and filtering
        rows with age under 15.
    """
    print("Filtering patients under 15...")

    # Calculate the actual age at ICU stay
    merged_df["icu_age"] = merged_df["anchor_age"] + (
        merged_df["icu_year"] - merged_df["anchor_year"]
    )

    # Filter out patients under 15 at ICU admission
    merged_df = merged_df[merged_df["icu_age"] >= MIN_AGE]

    # Remove columns we are not going to use any longer
    merged_df = merged_df.drop(columns=["icu_year", "anchor_age", "anchor_year"])

    print(f"Filtered. {len(merged_df)} rows left\n")
    return merged_df


def filter_death_during_stay(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where row death time was during the icu stay.

    Parameters
    ----------
    merged_df : pd.Dataframe
        The icu stays dataframe with the patients features.

    Returns
    -------
    pd.Dataframe
        The icu stays with patient features and filtering
        rows with death time during the icu stay.
    """
    print("Filtering patients that with time of death during stay...")

    # Filter out patients where death time is present and is between in time and out time
    mask = (
        (merged_df["deathtime"].notna())
        & (merged_df["intime"] <= merged_df["deathtime"])
        & (merged_df["deathtime"] <= merged_df["outtime"])
    )
    merged_df = merged_df[~mask]

    print(f"Filtered. {len(merged_df)} rows left\n")
    return merged_df


def time_to_death(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time to death for each patient.

    Remove patients where time of death is before admission.

    Parameters
    ----------
    merged_df : pd.Dataframe
        The icu stays dataframe with the patients features.

    Returns
    -------
    pd.Dataframe
        The icu stays with patient features and the
        time_to_death compute or 'No death'.
    """
    print("Compute time to death from ICU admission")

    # Remove rows where deathtime is before intime
    valid_df = merged_df[
        (pd.isna(merged_df["deathtime"]))
        | (merged_df["deathtime"] >= merged_df["intime"])
    ].copy()

    # Calculate time to death
    valid_df["Time_to_death_(h)"] = (
        pd.to_datetime(valid_df["deathtime"]) - valid_df["intime"]
    ).dt.total_seconds() / 3600

    # Set type as object
    valid_df["Time_to_death_(h)"] = (
        valid_df["Time_to_death_(h)"]
        .where(pd.notnull(valid_df["Time_to_death_(h)"]), "No death")
        .astype("object")
    )

    no_deaths = valid_df["Time_to_death_(h)"].value_counts().loc["No death"]
    print(f"Number of Survival patients: {no_deaths}")
    print("\n")

    return valid_df


def filter_multiple_icu_stay_per_admission(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows where the admission had more than one icu_stay.

    This is necessary because MIMICIV 2.2 gives diagnosis at admission
    level, not per icu_stay.

    Parameters
    ----------
    merged_df : pd.Dataframe
        The icu stays dataframe with the patients features.

    Returns
    -------
    pd.Dataframe
        The icu stays with patient features and filtering
        admissions with more than one icu_stay.
    """
    print("Filtering admissions with more than one icu_stay...")

    # Filter those that have more than one icu_stay in this admission,
    # since diagnosis are made at admission level, not icu_stay level
    unique_count = merged_df.groupby("hadm_id")["stay_id"].nunique()
    valid_groups = unique_count[unique_count == 1].index
    merged_df = merged_df[merged_df["hadm_id"].isin(valid_groups)]
    print(f"Filtered. {len(merged_df)} rows left\n")
    return merged_df
