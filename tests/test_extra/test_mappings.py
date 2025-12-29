from unittest.mock import patch

import numpy as np
import pandas as pd

from extra.mappings import (
    _read_css_mapping,
    map_icd_to_css,
    read_icd_mapping,
)


def test_read_icd_mapping():
    """
    Check that `read_icd_mapping` correctly loads the mappings.
    """
    mock_diagnosis = pd.DataFrame(
        {
            "diagnosis_type": ["ICD9", "ICD9", "ICD9"],
            "diagnosis_code": ["996.76", "V54.12", "730.06"],
            "diagnosis_description": [
                "OTHER COMPLICATIONS DUE TO GENITOURINARY DEVICE, IMPLANT, AND GRAFT",
                "AFTERCARE FOR HEALING TRAUMATIC FRACTURE OF LOWER ARM",
                "ACUTE OSTEOMYELITIS INVOLVING LOWER LEG",
            ],
            "icd9cm": ["99676", "V5412", "73006"],
            "icd10cm": ["T8384XA", "S52602D", "M86169"],
            "flags": [10000, 10000, 10000],
        }
    )

    with patch("pandas.read_csv", return_value=mock_diagnosis) as mock_read:
        result = read_icd_mapping("sponge/bob")
        mock_read.assert_called_once_with("sponge/bob", header=0, delimiter="\t")

    expected = pd.DataFrame(
        {
            "diagnosis_type": pd.Series(["ICD9", "ICD9", "ICD9"], dtype="object"),
            "diagnosis_code": pd.Series(["996.76", "V54.12", "730.06"], dtype="object"),
            "diagnosis_description": pd.Series(
                [
                    "other complications due to genitourinary device, implant, and graft",
                    "aftercare for healing traumatic fracture of lower arm",
                    "acute osteomyelitis involving lower leg",
                ],
                dtype="object",
            ),
            "icd9cm": pd.Series(["99676", "V5412", "73006"], dtype="object"),
            "icd10cm": pd.Series(["T8384XA", "S52602D", "M86169"], dtype="object"),
            "flags": pd.Series([10000, 10000, 10000], dtype="int64"),
        }
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=True,
        check_index_type=False,
    )


def test_map_icd_to_css():
    """
    Check that `map_icd_to_css` correctly transforms icd10 to CSSR.
    """
    mock_icustays = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "icd10_code": ["A000", "A009", "JUAN"],
        }
    )

    moc_mapping = pd.DataFrame(
        {
            "ICD-10-CM CODE": ["A000", "A009"],
            "CCSR CATEGORY 1": ["DIG001", ""],
            "CCSR CATEGORY 1 DESCRIPTION": ["Intestinal infection", ""],
        }
    )

    with patch(
        "extra.mappings._read_css_mapping", return_value=moc_mapping
    ) as mock_read:
        out = map_icd_to_css(mock_icustays, map_path="rubber/duck")
        mock_read.assert_called_once_with(map_path="rubber/duck")

    expected = pd.DataFrame(
        {
            "stay_id": [1, 2, 3],
            "CCSR CATEGORY 1": ["DIG001", np.nan, np.nan],
            "CCSR CATEGORY 1 DESCRIPTION": ["Intestinal infection", np.nan, np.nan],
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected, check_dtype=True, check_index_type=False
    )


def test_read_css_mapping():
    """
    Check that `_read_css_mapping` correctly loads the mappings.
    """
    mock_mapping = pd.DataFrame(
        {
            "'ICD-10-CM CODE'": ["'A000'", "'A001'", "'A009'"],
            "'ICD-10-CM CODE DESCRIPTION'": [
                "Cholera due to Vibrio cholerae 01, biovar cholerae",
                "Cholera due to Vibrio cholerae 01, biovar eltor",
                "Cholera, unspecified",
            ],
            "'Default CCSR CATEGORY IP'": ["'DIG001'", "'DIG001'", "'DIG001'"],
            "'Default CCSR CATEGORY DESCRIPTION IP'": [
                "Intestinal infection",
                "Intestinal infection",
                "Intestinal infection",
            ],
            "'Default CCSR CATEGORY OP'": ["'DIG001'", "'DIG001'", "'DIG001'"],
            "'Default CCSR CATEGORY DESCRIPTION OP'": [
                "Intestinal infection",
                "Intestinal infection",
                "Intestinal infection",
            ],
            "'CCSR CATEGORY 1'": ["'DIG001'", "'DIG001'", "'DIG001'"],
            "'CCSR CATEGORY 1 DESCRIPTION'": [
                "Intestinal infection",
                "Intestinal infection",
                "Intestinal infection",
            ],
            "'CCSR CATEGORY 2'": ["'INF003'", "'INF003'", "'INF003'"],
            "'CCSR CATEGORY 2 DESCRIPTION'": [
                "Bacterial infections",
                "Bacterial infections",
                "Bacterial infections",
            ],
            "'CCSR CATEGORY 3'": ["' '", "' '", "' '"],
            "'CCSR CATEGORY 3 DESCRIPTION'": ["", "", ""],
            "'CCSR CATEGORY 4'": ["' '", "' '", "' '"],
            "'CCSR CATEGORY 4 DESCRIPTION'": ["", "", ""],
            "'CCSR CATEGORY 5'": ["' '", "' '", "' '"],
            "'CCSR CATEGORY 5 DESCRIPTION'": ["", "", ""],
            "'CCSR CATEGORY 6'": ["' '", "' '", "' '"],
            "'CCSR CATEGORY 6 DESCRIPTION'": ["", "", ""],
            "'Rationale for Default Assignment'": [
                "06 Infectious conditions",
                "06 Infectious conditions",
                "06 Infectious conditions",
            ],
        }
    )

    with patch("pandas.read_csv", return_value=mock_mapping) as mock_read:
        out = _read_css_mapping("wow")
        mock_read.assert_called_once_with("wow")

    expected = pd.DataFrame(
        {
            "ICD-10-CM CODE": pd.Series(["A000", "A001", "A009"], dtype="object"),
            "CCSR CATEGORY 1": pd.Series(
                ["DIG001", "DIG001", "DIG001"], dtype="object"
            ),
            "CCSR CATEGORY 1 DESCRIPTION": pd.Series(
                ["Intestinal infection"] * 3, dtype="object"
            ),
            "CCSR CATEGORY 2": pd.Series(
                ["INF003", "INF003", "INF003"], dtype="object"
            ),
            "CCSR CATEGORY 2 DESCRIPTION": pd.Series(
                ["Bacterial infections"] * 3, dtype="object"
            ),
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected, check_dtype=True, check_index_type=False
    )
