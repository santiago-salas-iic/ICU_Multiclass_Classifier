from pathlib import Path

import pandas as pd
from tqdm import tqdm

from extra.charts import create_chart_features


def add_charts_features(
    mimic_root: Path, icustays_df: pd.DataFrame, valid_items: list, cutoff_h=3
):
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
    mimic_root : Path
        The path to the root of the mimic dataset.
    icustays_df : pd.Dataframe
        The icu stays dataframe.
    valid_items : list | None
        List of allowed feature codes.
    cutoff_h : float (default 24)
        Number of hours since admission in which the features will be considered.

    Returns
    -------
    pd.Dataframe
        The icu stays with the feature columns.
    """
    # Load values
    charts_df = load_charts(
        mimic_root=mimic_root,
        icustays_df=icustays_df,
        valid_items=valid_items,
        cutoff_h=cutoff_h,
    )

    # Sort values
    charts_df = charts_df.sort_values(["stay_id", "itemid", "event_time_from_admit"])

    # Get the last, max, min, mean, median value per patient and itemid
    charts_df = create_chart_features(
        charts_df=charts_df,
        stay_col="stay_id",
        variable_col="itemid",
        value_col="valuenum",
    )

    # Join the icustays_df with the features
    return icustays_df.merge(charts_df, on="stay_id", how="left")


def load_charts(
    mimic_root: Path,
    icustays_df: pd.DataFrame,
    valid_items: list,
    cutoff_h=3,
):
    """
    Load chart features from the Mimic IV 2.2 dataset.

    Modified from https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    mimic_root : str
        The path to the root of the mimic dataset.
    icustays_df : pd.DataFrame
        The icu stay dataframe from the pipeline.
    valid_items : list, None
        List or valid features to load, None means that all features will be loaded.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.DataFrame
        The chart dataframe with each row having a combination of icu stay_id and feature.
    """
    count = 0
    results = []

    print("Loading charts...")
    for chunk in tqdm(
        pd.read_csv(
            mimic_root / "icu" / "chartevents.csv.gz",
            compression="gzip",
            usecols=["stay_id", "charttime", "itemid", "valuenum", "valueuom"],
            dtype=None,
            chunksize=10_000_000,
        )
    ):
        count = count + 1

        # Filter out nan to save space
        filtered_chunk = chunk.dropna(subset=["valuenum"])

        # Only keep the features
        if valid_items is not None:
            filtered_chunk = filtered_chunk[filtered_chunk["itemid"].isin(valid_items)]

        # Filter out entries for patients not present in icustays_df after the filters
        chunk_merged = filtered_chunk.merge(
            icustays_df[["stay_id", "intime"]],
            how="inner",
            left_on="stay_id",
            right_on="stay_id",
        )

        # Filter out test after cutoff
        chunk_merged["charttime"] = pd.to_datetime(chunk_merged["charttime"])
        chunk_merged["event_time_from_admit"] = (
            chunk_merged["charttime"] - chunk_merged["intime"]
        )
        chunk_merged = chunk_merged[
            chunk_merged["event_time_from_admit"] < pd.Timedelta(hours=cutoff_h)
        ]

        del chunk_merged["charttime"]
        del chunk_merged["intime"]

        # Drop duplicates
        chunk_merged = chunk_merged.dropna(subset=["valuenum"])
        chunk_merged = chunk_merged.drop_duplicates()

        # Store chunk
        results.append(chunk_merged)

    df = pd.concat(results, ignore_index=True)

    return df


def change_itemid_to_item_name(mimic_root: Path, df: pd.DataFrame):
    """
    Auxiliary function to translate the feature column names from its itemid to its label.

    If a column has a name that its not an itemid, its name will not be changed.

    Parameters
    ----------
    mimic_root : Path
        The path to the root of the mimic dataset.
    df : pd.DataFrame
        The icu stay dataframe with each feature as a column.

    Returns
    -------
    pd.DataFrame
        The icu stay dataframe with each feature column name changed to the feature label.
    """
    # Load itemid data
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz", compression="gzip")

    # Change number to label or keep it as it is
    mapping = dict(zip(d_items["itemid"].astype(str), d_items["label"], strict=False))

    # Build new column names list
    new_columns = []

    for col in df.columns:
        # Skip columns that do not contain an underscore
        if "_" not in col:
            new_columns.append(col)
            continue

        # Split only on last underscore
        parts = col.rsplit("_", 1)
        metric = parts[0]
        itemid = parts[1]

        # Replace itemid if in mapping
        if itemid in mapping:
            new_col = f"{metric}_{mapping[itemid]}"
        else:
            new_col = col

        new_columns.append(new_col)

    # Apply renaming
    df = df.copy()
    df.columns = new_columns
    return df
