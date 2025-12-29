from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mimic_pipeline.utils.icustays import filter_invalid_stays, load_icustays


def test_filter_invalid_stays():
    """
    Check that `filter_invalid_stays` correctly filters stays outside given length bounds.
    """
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "los": [0.3, 1.0, 5.0, 12.5],
        }
    )

    out = filter_invalid_stays(df, min_days=0.5, max_days=10.0)

    expected = pd.DataFrame(
        {
            "stay_id": [2, 3],
            "los": [1.0, 5.0],
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True)[expected.columns],
        expected,
        check_dtype=True,
        check_index_type=False,
    )


def test_load_icustays_default():
    """
    Check that `load_icustays` works as expected.

    Given the expected input format it returns the expected output format.
    And calls all of the filters.

    subject_id 1 is valid.
    subject_id 2 is not valid, it has invalid length of stay.
    subject_id 3 is valid.
    subject_id 4 is not valid, it has invalid length of stay.
    """
    mock_icustays_df = pd.DataFrame(
        {
            "subject_id": [1, 2, 3, 4],
            "hadm_id": [100, 200, 300, 400],
            "stay_id": [1000, 2000, 3000, 4000],
            "intime": [
                "2025-04-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 08:00:00",
            ],
            "outtime": [
                "2025-04-05 08:00:00",
                "2025-04-01 14:00:00",
                "2025-05-05 08:00:00",
                "2025-05-01 08:00:00",
            ],
            "los": [
                4.0,
                0.25,
                4.0,
                30.0,
            ],
        }
    )

    expected = pd.DataFrame(
        {
            "subject_id": pd.Series([1, 3]),
            "hadm_id": pd.Series([100, 300]),
            "stay_id": pd.Series([1000, 3000]),
            "intime": [
                pd.Timestamp("2025-04-01 08:00:00"),
                pd.Timestamp("2025-05-01 08:00:00"),
            ],
            "outtime": [
                pd.Timestamp("2025-04-05 08:00:00"),
                pd.Timestamp("2025-05-05 08:00:00"),
            ],
            "los": pd.Series([4.0, 4.0]),
            "icu_year": pd.Series([2025, 2025], dtype="int32"),
        }
    )

    with (
        patch("pandas.read_csv", side_effect=[mock_icustays_df]),
        patch(
            "mimic_pipeline.utils.icustays.filter_invalid_stays",
            wraps=filter_invalid_stays,
        ) as mock_filter_invalid_stays,
    ):
        out = load_icustays(Path("wingardium leviosa"))

        pd.testing.assert_frame_equal(
            out.reset_index(drop=True)[expected.columns],
            expected,
            check_dtype=True,
            check_index_type=False,
        )

        assert mock_filter_invalid_stays.called
