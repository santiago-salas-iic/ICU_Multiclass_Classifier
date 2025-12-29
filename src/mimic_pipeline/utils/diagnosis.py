from pathlib import Path

import numpy as np
import pandas as pd

from extra.mappings import map_icd_to_css, read_icd_mapping


def add_diagnosis(
    mimic_root: str, icustays_df: pd.DataFrame, diagnosis_codes: list | None
):
    """
    Add diagnosis to the icu stays dataframe.

    Only add diagnosis from the list of allowed diagnosis codes.

    Parameters
    ----------
    mimic_root : str
        The path to the root of the mimic dataset.
    icustays_df : pd.Dataframe
        The icu stays dataframe.
    diagnosis_codes : list | None
        List of allowed diagnosis CSS codes. If None al CSS codes will be allowed.

    Returns
    -------
    pd.Dataframe
        The icu stays with the CSS code of the diagnosis.
        One icu stay can only have one diagnosis.
    """
    print("Loading diagnosis as ICD 10 and removing stays with no diagnosis")

    diag_df = pd.read_csv(
        f"{mimic_root}/hosp/diagnoses_icd.csv.gz",
        compression="gzip",
        usecols=["hadm_id", "icd_code", "seq_num", "icd_version"],
    )

    project_root = Path(__file__).resolve().parents[3]
    icd_map = read_icd_mapping(project_root / "mappings" / "ICD9_to_ICD10_mapping.txt")
    _standardize_icd(icd_map, diag_df)
    diag_df.dropna(subset=["icd10_code"], inplace=True)

    # Merge the diagnosis
    icustays_df = icustays_df.merge(
        diag_df[["hadm_id", "seq_num", "icd10_code"]], on="hadm_id", how="left"
    )

    # Drop stays with no diagnosis
    icustays_df = icustays_df.dropna(subset=["icd10_code"])

    # Keep diagnosis of most priority (lowest seq_num)
    icustays_df = icustays_df.loc[icustays_df.groupby(["stay_id"])["seq_num"].idxmin()]

    # Not needed anymore
    icustays_df = icustays_df.drop(columns="seq_num")

    # Change diagnosis from ICD-10 to CSS
    print("Changing ICD 10 code to CSS")
    icustays_df = map_icd_to_css(
        icustays_df=icustays_df,
        map_path=project_root / "mappings" / "DXCCSR_v2025-1.csv",
    )

    # If specific diagnosis codes where selected
    if diagnosis_codes:
        icustays_df = icustays_df[icustays_df["CCSR CATEGORY 1"].isin(diagnosis_codes)]

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
    df[col_name] = df["icd_code"].values

    # Group identical ICD9 codes, then convert all ICD9 codes within a group to ICD10
    ICU_9_CODE = 9
    for code, group in df.loc[df.icd_version == ICU_9_CODE].groupby(by="icd_code"):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            df.at[idx, col_name] = new_code
