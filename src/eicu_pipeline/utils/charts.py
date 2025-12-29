from pathlib import Path

import pandas as pd
from tqdm import tqdm

from extra.charts import create_chart_features


def add_charts_features(eicu_root: Path, icustays_df: pd.DataFrame, cutoff_h=3):
    """
    Add charts features to the icu stays dataframe.

    For all icu stays add all charts features present in the dataset.
    For each feature we keep:
        - The value that is the one closest from bellow to the cutoff time.
        - The max value before cutoff.
        - The min value before cutoff.
        - The mean value before cutoff.
        - The median value before cutoff.

    Each feature value will be returned as a column.

    Parameters
    ----------
    eicu_root : Path
        The path to the root of the dataset.
    icustays_df : pd.DataFrame
        The icu stay dataframe from the pipeline.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.Dataframe
        The icu stays with the feature columns.
    """
    # Load respiratory charting data
    respiratory_charts_df = load_respiratory_charting(
        eicu_root=eicu_root, icustays_df=icustays_df, cutoff_h=cutoff_h
    )

    # Make all values numeric
    respiratory_charts_df["respchartvalue"] = pd.to_numeric(
        respiratory_charts_df["respchartvalue"], errors="coerce"
    )

    # Sort
    respiratory_charts_df = respiratory_charts_df.sort_values(
        ["patientunitstayid", "respchartvaluelabel", "respchartoffset"]
    )

    # Get the last, max, min, mean, median value per patient and respchartvaluelabel
    respiratory_charts_df = create_chart_features(
        charts_df=respiratory_charts_df,
        stay_col="patientunitstayid",
        variable_col="respchartvaluelabel",
        value_col="respchartvalue",
    )

    # Merge back with icustays_df
    icustays_df = icustays_df.merge(
        respiratory_charts_df, on="patientunitstayid", how="left"
    )

    # Load nurse charts features
    nurse_charts_df = load_nurse_charting(
        eicu_root=eicu_root, icustays_df=icustays_df, cutoff_h=cutoff_h
    )

    # Make all values numeric
    nurse_charts_df["nursingchartvalue"] = pd.to_numeric(
        nurse_charts_df["nursingchartvalue"], errors="coerce"
    )

    # Sort
    nurse_charts_df = nurse_charts_df.sort_values(
        ["patientunitstayid", "nursingchartcelltypevallabel", "nursingchartoffset"]
    )

    # Get the last, max, min, mean, median value per patient and nursingchartcelltypevallabel
    nurse_charts_df = create_chart_features(
        charts_df=nurse_charts_df,
        stay_col="patientunitstayid",
        variable_col="nursingchartcelltypevallabel",
        value_col="nursingchartvalue",
    )

    # Merge back with icustays_df
    icustays_df = icustays_df.merge(nurse_charts_df, on="patientunitstayid", how="left")

    # Add vital periodic features
    vital_periodic_df = load_vital_periodic(
        eicu_root=eicu_root, icustays_df=icustays_df, cutoff_h=cutoff_h
    )

    vital_columns = [
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "cvp",
        "etco2",
        "systemicsystolic",
        "systemicdiastolic",
        "systemicmean",
        "pasystolic",
        "padiastolic",
        "pamean",
    ]
    vital_periodic_df = vital_periodic_df.dropna(subset=vital_columns, how="all")

    # Melt the dataframe to long format so each vital is treated as a variable
    vital_periodic_df = vital_periodic_df.melt(
        id_vars=["patientunitstayid", "observationoffset"],
        value_vars=vital_columns,
        var_name="vital",
        value_name="value",
    ).dropna(subset=["value"])

    # Sort
    vital_periodic_df = vital_periodic_df.sort_values(
        ["patientunitstayid", "vital", "observationoffset"]
    )

    # Get the last, max, min, mean, median value per patient and nursingchartcelltypevallabel
    vital_periodic_df = create_chart_features(
        vital_periodic_df,
        stay_col="patientunitstayid",
        variable_col="vital",
        value_col="value",
    )

    # Merge back with icustays_df
    icustays_df = icustays_df.merge(
        vital_periodic_df, on="patientunitstayid", how="left"
    )

    # Add virtual aperiodic features
    vital_aperiodic_df = load_vital_aperiodic(
        eicu_root=eicu_root, icustays_df=icustays_df, cutoff_h=cutoff_h
    )

    vital_columns = [
        "noninvasivesystolic",
        "noninvasivediastolic",
        "noninvasivemean",
        "paop",
        "cardiacoutput",
        "cardiacinput",
        "svr",
        "svri",
        "pvr",
        "pvri",
    ]

    vital_aperiodic_df = vital_aperiodic_df.dropna(subset=vital_columns, how="all")

    # Melt the dataframe to long format so each vital is treated as a variable
    vital_aperiodic_df = vital_aperiodic_df.melt(
        id_vars=["patientunitstayid", "observationoffset"],
        value_vars=vital_columns,
        var_name="vital",
        value_name="value",
    ).dropna(subset=["value"])

    # Sort
    vital_aperiodic_df = vital_aperiodic_df.sort_values(
        ["patientunitstayid", "vital", "observationoffset"]
    )

    # Get the last, max, min, mean, median value per patient and nursingchartcelltypevallabel
    vital_aperiodic_df = create_chart_features(
        vital_aperiodic_df,
        stay_col="patientunitstayid",
        variable_col="vital",
        value_col="value",
    )

    # Merge back with icustays_df
    return icustays_df.merge(vital_aperiodic_df, on="patientunitstayid", how="left")


def load_vital_periodic(
    eicu_root: Path,
    icustays_df: pd.DataFrame,
    cutoff_h=3,
):
    """
    Load vital periodic features from the EICU 2.0 dataset.

    Modified from https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    eicu_root : str
        The path to the root of the eicu dataset.
    icustays_df : pd.DataFrame
        The icu stay dataframe from the pipeline.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.DataFrame
        The chart dataframe with each row having a combination of icu stay_id and feature.
    """
    print("Loading vital periodic charts in chunks...")
    results = []

    for chunk in tqdm(
        pd.read_csv(
            eicu_root / "vitalPeriodic.csv.gz",
            compression="gzip",
            chunksize=10_000_000,
            dtype=None,
        )
    ):
        # Keep only patients in filtered ICU stays
        chunk = chunk[chunk["patientunitstayid"].isin(icustays_df["patientunitstayid"])]

        # Keep only observations before cutoff
        chunk = chunk[chunk["observationoffset"] <= cutoff_h * 60]

        # Convert temperature to Fahrenheit if available
        if "temperature" in chunk.columns:
            chunk["Temperature Fahrenheit"] = chunk["temperature"] * 9 / 5 + 32

        results.append(chunk)

    # Concatenate all chunks
    return pd.concat(results, ignore_index=True)


def load_vital_aperiodic(
    eicu_root: Path,
    icustays_df: pd.DataFrame,
    cutoff_h=3,
):
    """
    Load vital Aperiodic features from the EICU 2.0 dataset.

    Modified from https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    eicu_root : str
        The path to the root of the eicu dataset.
    icustays_df : pd.DataFrame
        The icu stay dataframe from the pipeline.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.DataFrame
        The chart dataframe with each row having a combination of icu stay_id and feature.
    """
    print("Loading vital aperiodic charts in chunks...")
    results = []

    for chunk in tqdm(
        pd.read_csv(
            eicu_root / "vitalAperiodic.csv.gz",
            compression="gzip",
            chunksize=10_000_000,
            dtype=None,
        )
    ):
        # Keep only patients in filtered ICU stays
        chunk = chunk[chunk["patientunitstayid"].isin(icustays_df["patientunitstayid"])]

        # Keep only observations before cutoff
        chunk = chunk[chunk["observationoffset"] <= cutoff_h * 60]

        results.append(chunk)

    # Concatenate all chunks
    return pd.concat(results, ignore_index=True)


def load_nurse_charting(eicu_root, icustays_df, cutoff_h):
    """
    Load nurse chart features from the EICU 2.0 dataset.

    Modified from https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    eicu_root : str
        The path to the root of the eicu dataset.
    icustays_df : pd.DataFrame
        The icu stay dataframe from the pipeline.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.DataFrame
        The chart dataframe with each row having a combination of icu stay_id and feature.
    """
    print("Loading nurseCharting in chunks...")

    cut_off = cutoff_h * 60
    chunks = []

    for chunk in tqdm(
        pd.read_csv(
            eicu_root / "nurseCharting.csv.gz", compression="gzip", chunksize=5_000_000
        )
    ):
        # Keep only patients in filtered ICU stays
        chunk = chunk[chunk["patientunitstayid"].isin(icustays_df["patientunitstayid"])]

        # Keep only observations before cutoff
        chunk = chunk[(chunk["nursingchartoffset"] <= cut_off)]

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def load_respiratory_charting(eicu_root, icustays_df, cutoff_h):
    """
    Load respiratory chart features from the eICU 2.0 dataset.

    Modified from https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    eicu_root : str
        The path to the root of the eicu dataset.
    icustays_df : pd.DataFrame
        The icu stay dataframe from the pipeline.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.DataFrame
        The chart dataframe with each row having a combination of icu stay_id and feature.
    """
    print("Loading respiratoryCharting in chunks...")

    cut_off = cutoff_h * 60
    chunks = []

    for chunk in tqdm(
        pd.read_csv(
            eicu_root / "respiratoryCharting.csv.gz",
            compression="gzip",
            chunksize=5_000_000,
        )
    ):
        # Keep only patients in filtered ICU stays
        chunk = chunk[chunk["patientunitstayid"].isin(icustays_df["patientunitstayid"])]

        # Only observations before cutoff
        chunk = chunk[chunk["respchartoffset"] <= cut_off]

        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)
