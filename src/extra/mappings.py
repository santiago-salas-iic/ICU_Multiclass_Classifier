from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_icd_mapping(map_path: str) -> pd.DataFrame:
    """
    Read mapping table to convert ICD9 to ICD10 codes.

    link: https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    map_path : str
        The mapping csv.

    Returns
    -------
    pd.Dataframe
        The dataframe with ICD9 to ICD10 codes.
    """
    mapping = pd.read_csv(map_path, header=0, delimiter="\t")
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


def map_icd_to_css(icustays_df: pd.DataFrame, map_path: str) -> pd.DataFrame:
    """
    Map ICD-10 codes to 'CCSR CATEGORY 1' codes.

    Parameters
    ----------
    icustays_df : pd.Dataframe
        The icu stays dataframe, with the diagnosis as icd-10.
    map_path : str
        The path to the mapping csv.

    Returns
    -------
    pd.Dataframe
        The icu stays with the CSS code of the diagnosis.
    """
    # Read mapping
    mapping = _read_css_mapping(map_path=map_path)

    # Merge CSSR values to the icu_stays
    icustays_df = icustays_df.merge(
        mapping, left_on="icd10_code", right_on="ICD-10-CM CODE", how="left"
    )

    # Replace empty value with NaN
    icustays_df["CCSR CATEGORY 1"] = icustays_df["CCSR CATEGORY 1"].replace("", np.nan)
    icustays_df["CCSR CATEGORY 1 DESCRIPTION"] = icustays_df[
        "CCSR CATEGORY 1 DESCRIPTION"
    ].replace("", np.nan)

    return icustays_df.drop(columns=["ICD-10-CM CODE", "icd10_code"])


def _read_css_mapping(map_path: str) -> pd.DataFrame:
    """
    Read the mapping table to convert ICD10 to CSS codes.

    Parameters
    ----------
    map_path : str
        The path to the mapping csv.

    Returns
    -------
    pd.Dataframe
        The mappings as a dataframe with one row per icd-10 code and its corresponding mapping.
    """
    # Load your dataset containing ICD-10 codes
    ccs_mapping = pd.read_csv(map_path)  # assume a column 'ICD10'

    ccs_mapping.columns = (
        ccs_mapping.columns.str.strip()
        .str.replace("'", "", regex=False)  # Remove \
        .str.upper()
    )

    for col in ccs_mapping.columns:
        ccs_mapping[col] = ccs_mapping[col].str.strip("'")

    important_col_reduced = [
        "ICD-10-CM CODE",
        "CCSR CATEGORY 1",
        "CCSR CATEGORY 1 DESCRIPTION",
        "CCSR CATEGORY 2",
        "CCSR CATEGORY 2 DESCRIPTION",
    ]

    return ccs_mapping[important_col_reduced]


def _map_eicu_data_to_mimic(mapping: dict, eicu_data: pd.DataFrame):
    """
    Map columns between eICU and MIMIC datasets.

    Parameters
    ----------
    mapping : dict
        The loaded mapping.
    eicu_data : pd.DataFrame
        The eicu dataframe extracted using the pipeline.

    Returns
    -------
    pd.DataFrame
        The dataframe with renamed columns.
    """
    # Create reverse mapping: eicu to mimic
    reverse_mapping = {
        old_name: new_name
        for new_name, old_names in mapping.items()
        for old_name in old_names
    }

    # Keep only columns present in the mapping
    cols_to_keep = [col for col in eicu_data.columns if col in reverse_mapping]
    eicu_data = eicu_data[cols_to_keep]

    # Rename columns to the new standardized names
    eicu_data = eicu_data.rename(columns=reverse_mapping)

    # Some columns may be duplicated
    # Unify them by getting the mean
    duplicated_cols = eicu_data.columns[eicu_data.columns.duplicated()].unique()
    if not duplicated_cols.empty:
        numeric_means = {}
        for col in duplicated_cols:
            cols = [c for c in eicu_data.columns if c == col]

            numeric_eicu_data = eicu_data[cols].apply(pd.to_numeric, errors="coerce")

            numeric_means[col] = numeric_eicu_data.mean(axis=1)

        eicu_data = eicu_data.drop(columns=duplicated_cols)

        for col, series in numeric_means.items():
            eicu_data[col] = series

    return eicu_data


def equate_columns_mimic_and_eicu(mimic_data, eicu_data):
    """
    Equate eicu columns to mimic columns.

    It uses the mappings/mimic_to_eicu.yaml map to
    transform all eicu columns to the corresponding mimic column.

    All columns not in the mapping will be dropped for both dataframes.

    Columns present in mimic but not in eicu will be added as NaN.

    Parameters
    ----------
    mimic_data : pd.DataFrame
        The mimic dataframe extracted using the pipeline.
    eicu_data : pd.DataFrame
        The eicu dataframe extracted using the pipeline.

    Returns
    -------
    pd.DataFrame
        The processed mimic dataframe.
    pd.DataFrame
        The processed eicu dataframe.
    """
    project_root = Path(__file__).resolve().parents[2]

    # Load the mapping from YAML
    with open(project_root / "mappings" / "mimic_to_eicu.yaml", "r") as file:
        mapping = yaml.safe_load(file)

    # Expand it to add the prefixes
    prefixes = ["last_", "mean_", "median_", "max_", "min_"]

    # Create the new expanded mapping
    expanded_mapping = {}
    for key, values in mapping.items():
        expanded_mapping[key] = values
        for prefix in prefixes:
            new_key = f"{prefix}{key}"
            new_values = [f"{prefix}{v}" for v in values]
            expanded_mapping[new_key] = new_values

    mapping = expanded_mapping

    # Get list of valid new column names from YAML
    valid_columns = list(mapping.keys())

    # Keep only valid columns for mimic
    mimic_data = mimic_data[
        [col for col in mimic_data.columns if col in valid_columns]
    ].copy()

    # Map columns for eicu
    eicu_data = _map_eicu_data_to_mimic(mapping=mapping, eicu_data=eicu_data)

    # Add missing columns as NaN
    missing_cols = [col for col in mimic_data.columns if col not in eicu_data.columns]
    for col in missing_cols:
        eicu_data[col] = np.nan

    # Make columns have the same order
    eicu_data = eicu_data[mimic_data.columns]
    return mimic_data, eicu_data
