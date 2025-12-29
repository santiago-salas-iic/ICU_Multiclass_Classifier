from pathlib import Path

import numpy as np
import pandas as pd

from extra.mappings import map_icd_to_css, read_icd_mapping


def add_diagnosis(
    eicu_root: Path, icustays_df: pd.DataFrame, diagnosis_codes: list | None, cutoff_h=3
):
    """
    Add primary diagnosis for each stayid.

    Parameters
    ----------
    eicu_root : str
        The path to the root of the dataset.
    icustays_df : pd.Dataframe
        The icu stays dataframe.
    diagnosis_codes : list
        CCSR diagnosis list.
    cutoff_h : float
        The number of hours since admission of which feature values will be loaded.

    Returns
    -------
    pd.Dataframe
        The CCSR diagnosis.
    """
    cut_off = cutoff_h * 60

    project_root = Path(__file__).resolve().parents[3]

    print("Loading diagnosis as ICD 10 and removing stays with no diagnosis")
    icd_map = read_icd_mapping(project_root / "mappings" / "ICD9_to_ICD10_mapping.txt")
    diag_df = pd.read_csv(
        eicu_root / "diagnosis.csv.gz",
        compression="gzip",
        usecols=[
            "patientunitstayid",
            "icd9code",
            "diagnosisoffset",
            "diagnosispriority",
        ],
    )

    # Keep only primary diagnosis
    diag_df = diag_df[
        (diag_df["diagnosispriority"] == "Primary")
        & (diag_df["diagnosisoffset"] <= cut_off)
    ]

    # Keep only the last diagnosis
    diag_df = diag_df.loc[
        diag_df.groupby("patientunitstayid")["diagnosisoffset"].idxmax()
    ]

    # Drop no icd9code
    diag_df.dropna(subset=["icd9code"], inplace=True)
    diag_df["icd9code"] = (
        diag_df["icd9code"].astype(str).str.split(",").str[0].str.strip()
    )
    # Trasform to icd10
    _standardize_icd(icd_map, diag_df)

    # Merge with icustays
    icustays_df = icustays_df.merge(
        diag_df[["patientunitstayid", "icd10_code"]], on="patientunitstayid", how="left"
    )

    # Drop stays with no diagnosis
    icustays_df = icustays_df.dropna(subset=["icd10_code"])

    # Change diagnosis from ICD-10 to CSS
    print("Changing ICD 10 code to CSS")
    icustays_df = map_icd_to_css(
        icustays_df=icustays_df,
        map_path=project_root / "mappings" / "DXCCSR_v2025-1.csv",
    )
    if diagnosis_codes:
        icustays_df = icustays_df[
            icustays_df["CCSR CATEGORY 1"].isin(diagnosis_codes)
        ].reset_index(drop=True)

    print(f"Loaded diagnosis, {len(icustays_df)} rows left\n")

    return icustays_df


def _standardize_icd(mapping, df):
    """
    Add column with converted ICD10.

    link: https://github.com/healthylaife/MIMIC-IV-Data-Pipeline/blob/main/utils/icu_preprocess_util.py

    Parameters
    ----------
    mapping : pd.DataFrame
        The mapping DataFrame.
    df : pd.DataFrame
        The DataFrame to add ICD10 column.
    """

    def icd_9to10(icd):
        icd = icd[:3]
        try:
            # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
            return mapping.loc[mapping.diagnosis_code == icd].icd10cm.iloc[0]
        except IndexError:
            return np.nan

    # Create new column with original codes as default
    col_name = "icd10_code"
    df[col_name] = df["icd9code"].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    for code, group in df.groupby(by="icd9code"):
        new_code = icd_9to10(code)

        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            df.at[idx, col_name] = new_code
