from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from mimic_pipeline.utils.charts import (
    add_charts_features,
    change_itemid_to_item_name,
    load_charts,
)


def test_load_charts():
    """
    Check that "test_load_charts" loads and filters as expected.
    """
    icustays_df = pd.DataFrame(
        {"stay_id": [10], "intime": [pd.Timestamp("2025-04-01 08:00:00")]}
    )

    # Two chunks
    chunk1 = pd.DataFrame(
        {
            "stay_id": [10, 10],
            "charttime": [
                "2025-04-01 09:00:00",  # valid
                "2025-04-01 16:00:00",  # invalid
            ],
            "itemid": [123, 123],
            "valuenum": [98.6, 99.1],
            "valueuom": ["F", "F"],
        }
    )

    chunk2 = pd.DataFrame(
        {
            "stay_id": [10, 10],
            "charttime": [
                "2025-04-01 10:00:00",  # valid
                "2025-04-01 11:00:00",  # invalid
            ],
            "itemid": [123, 123],
            "valuenum": [97.5, np.nan],
            "valueuom": ["F", "F"],
        }
    )

    expected = pd.DataFrame(
        {
            "stay_id": [10, 10],
            "itemid": [123, 123],
            "valuenum": [98.6, 97.5],
            "valueuom": ["F", "F"],
            "event_time_from_admit": [timedelta(hours=1), timedelta(hours=2)],
        }
    )

    with (
        patch("pandas.read_csv", return_value=[chunk1, chunk2]),
    ):
        out = load_charts(
            mimic_root=Path("wow"),
            icustays_df=icustays_df,
            valid_items=[123],
            cutoff_h=6,
        )

        pd.testing.assert_frame_equal(
            out[expected.columns].reset_index(drop=True),
            expected,
            check_dtype=False,
        )


def test_change_itemid_to_item_name():
    """
    Test the `change_itemid_to_item_name` correctly changes itemid to name.
    """
    mock_input_df = pd.DataFrame(
        {
            "last_220045": [1.0, 2.0],
            "mean_220045": [1.0, 2.0],
            "median_220045": [1.0, 2.0],
            "max_220045": [1.0, 2.0],
            "min_220045": [1.0, 2.0],
            "last_220050": [3.0, 4.0],
            "mean_220050": [3.0, 4.0],
            "median_220050": [3.0, 4.0],
            "max_220050": [3.0, 4.0],
            "min_220050": [3.0, 4.0],
            "non_itemid_col": [5.0, 6.0],
            "stayid": [1, 1],
        }
    )

    mock_d_items = pd.DataFrame(
        {
            "itemid": [220045, 220050],
            "label": ["Heart Rate", "SpO2"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "last_Heart Rate": [1.0, 2.0],
            "mean_Heart Rate": [1.0, 2.0],
            "median_Heart Rate": [1.0, 2.0],
            "max_Heart Rate": [1.0, 2.0],
            "min_Heart Rate": [1.0, 2.0],
            "last_SpO2": [3.0, 4.0],
            "mean_SpO2": [3.0, 4.0],
            "median_SpO2": [3.0, 4.0],
            "max_SpO2": [3.0, 4.0],
            "min_SpO2": [3.0, 4.0],
            "non_itemid_col": [5.0, 6.0],
            "stayid": [1, 1],
        }
    )

    with patch("pandas.read_csv", return_value=mock_d_items):
        out = change_itemid_to_item_name(Path("mock/path"), mock_input_df)

    pd.testing.assert_frame_equal(out, expected_df)


def test_add_charts_features():
    """
    Check that `test_add_charts_features` works as expected.

    Given the expected input format it returns the expected output format.
    And calls filters invalid values.
    """
    icustays_df = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "intime": [
                pd.Timestamp("2025-04-01 08:00:00"),
                pd.Timestamp("2025-04-02 08:00:00"),
            ],
        }
    )

    # Four chunks
    chunk1 = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [
                pd.Timestamp("2025-04-01 09:00:00"),
                pd.Timestamp("2025-04-02 09:00:00"),
            ],
            "itemid": [220045, 220045],
            "valuenum": [98.6, 99.5],
            "valueuom": ["{beats}/sec", "{beats}/sec"],
        }
    )

    chunk2 = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [
                pd.Timestamp("2025-04-01 09:00:00"),
                pd.Timestamp("2025-04-02 09:00:00"),
            ],
            "itemid": [220050, 220050],
            "valuenum": [100, 99],
            "valueuom": ["%", "%"],
        }
    )

    # outside cutoff
    chunk3 = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [
                pd.Timestamp("2025-06-01 09:00:00"),
                pd.Timestamp("2025-06-02 09:00:00"),
            ],
            "itemid": [220050, 220050],
            "valuenum": [100, 99],
            "valueuom": ["%", "%"],
        }
    )

    # not in valid_items
    chunk4 = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "charttime": [
                pd.Timestamp("2025-04-01 09:00:00"),
                pd.Timestamp("2025-04-02 09:00:00"),
            ],
            "itemid": [123, 123],
            "valuenum": [100, 99],
            "valueuom": ["%", "%"],
        }
    )

    expected_df = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "intime": [
                pd.Timestamp("2025-04-01 08:00:00"),
                pd.Timestamp("2025-04-02 08:00:00"),
            ],
            "last_220045": [98.6, 99.5],
            "mean_220045": [98.6, 99.5],
            "median_220045": [98.6, 99.5],
            "max_220045": [98.6, 99.5],
            "min_220045": [98.6, 99.5],
            "last_220050": [100, 99],
            "mean_220050": [100, 99],
            "median_220050": [100, 99],
            "max_220050": [100, 99],
            "min_220050": [100, 99],
        }
    )

    with (
        patch("pandas.read_csv", return_value=[chunk1, chunk2, chunk3, chunk4]),
        patch("pandas.DataFrame.to_csv"),
    ):
        out = add_charts_features(
            mimic_root=Path("mock/path"),
            icustays_df=icustays_df,
            valid_items=[220045, 220050],
            cutoff_h=6,
        )

    out = out[expected_df.columns]

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), expected_df, check_dtype=False
    )
