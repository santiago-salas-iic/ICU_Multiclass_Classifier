from unittest.mock import patch

import numpy as np
import pandas as pd

from mimic_pipeline.utils.diagnosis import (
    _standardize_icd,
    add_diagnosis,
)


def test__standardize_icd():
    """
    Check that `_standardize_icd` correctly changes ic9 to ic10.
    """
    mock_mapping = pd.DataFrame(
        {"diagnosis_code": ["996", "730"], "icd10cm": ["T8384XA", "M86169"]}
    )

    mock_icustays = pd.DataFrame(
        {
            "icd_code": ["996.76", "730.06", "999.99", "A001"],
            "icd_version": [9, 9, 9, 10],
        }
    )

    _standardize_icd(mock_mapping, mock_icustays)

    expected_icd10 = ["T8384XA", "M86169", np.nan, "A001"]
    assert mock_icustays["icd10_code"].tolist() == expected_icd10


def test_add_diagnosis_default():
    """
    Check that `add_diagnosis` adds diagnoses correctly and filters invalid stays.

    The result contains one diagnosis per stay, using the lowest seq_num,
    and maps ICD-9 to ICD-10 and then to CCSR correctly.
    """
    # Mock input
    mock_icustays_df = pd.DataFrame(
        {
            "subject_id": [1, 2],
            "hadm_id": [100, 200],
            "stay_id": [1000, 2000],
        }
    )

    # Mock all csv files
    mock_diagnoses_df = pd.DataFrame(
        {
            "hadm_id": [100, 100, 200],
            "icd_code": ["996.76", "730.06", "999.99"],
            "seq_num": [2, 1, 1],
            "icd_version": [9, 9, 9],
        }
    )

    mock_icd_mapping = pd.DataFrame(
        {
            "diagnosis_code": ["996", "V54", "730"],
            "icd10cm": ["T8384XA", "S52602D", "M86169"],
            "diagnosis_description": [
                "OTHER COMPLICATIONS DUE TO GENITOURINARY DEVICE, IMPLANT, AND GRAFT",
                "AFTERCARE FOR HEALING TRAUMATIC FRACTURE OF LOWER ARM",
                "ACUTE OSTEOMYELITIS INVOLVING LOWER LEG",
            ],
        }
    )

    mock_css_mapping = pd.DataFrame(
        {
            "ICD-10-CM CODE": ["T8384XA", "S52602D", "M86169"],
            "CCSR CATEGORY 1": ["INF001", "SURG002", "INF003"],
            "CCSR CATEGORY 1 DESCRIPTION": [
                "Infections",
                "Surgical conditions",
                "Bacterial infections",
            ],
            "CCSR CATEGORY 2": ["", "", ""],
            "CCSR CATEGORY 2 DESCRIPTION": ["", "", ""],
        }
    )

    # Expected result
    expected = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "stay_id": [1000],
            "CCSR CATEGORY 1": ["INF003"],
            "CCSR CATEGORY 1 DESCRIPTION": ["Bacterial infections"],
            "CCSR CATEGORY 2": [""],
            "CCSR CATEGORY 2 DESCRIPTION": [""],
        }
    )

    with patch(
        "pandas.read_csv",
        side_effect=[mock_diagnoses_df, mock_icd_mapping, mock_css_mapping],
    ):
        out = add_diagnosis(
            mimic_root="mock/path",
            icustays_df=mock_icustays_df,
            diagnosis_codes=["INF001", "SURG002", "INF003"],
        )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        expected,
        check_dtype=False,
    )
