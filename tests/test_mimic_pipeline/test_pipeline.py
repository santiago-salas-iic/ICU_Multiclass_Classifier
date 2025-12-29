from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mimic_pipeline.load_data import MimicLoad


@patch("mimic_pipeline.load_data.change_itemid_to_item_name")
@patch("mimic_pipeline.load_data.add_charts_features")
@patch("mimic_pipeline.load_data.add_diagnosis")
@patch("mimic_pipeline.load_data.add_patient_features")
@patch("mimic_pipeline.load_data.load_icustays")
def test_mimic_preprocess_pipeline(
    mock_load, mock_add_patient, mock_add_diagnosis, mock_add_charts, mock_change
):
    """
    Test the MimicLoad pipeline by mocking all data transformation steps.

    This test ensures that:
    - Each preprocessing function is called exactly once with the correct arguments.
    - The final output `data` attribute of the `MimicLoad` instance
      is set correctly to the output of `change_itemid_to_item_name`.
    - The default attributes such as `target` and `categoric_columns` are properly initialized.

    Parameters
    ----------
    mock_load : MagicMock
        Mocked function for `load_icustays`, returns a dummy ICU stays DataFrame.
    mock_add_patient : MagicMock
        Mocked function for `add_patient_features`, returns a dummy patient feature DataFrame.
    mock_add_diagnosis : MagicMock
        Mocked function for `add_diagnosis`, returns a DataFrame with diagnosis codes.
    mock_add_charts : MagicMock
        Mocked function for `add_charts_features`, returns a DataFrame with charted vitals.
    mock_change : MagicMock
        Mocked function for `change_itemid_to_item_name`, returns final processed DataFrame.
    """
    mimic_root = Path("/fake/path")

    # Prepare dummy DataFrames for each step
    icu_stays_df = pd.DataFrame({"stay_id": [1, 2, 3, 4]})
    mock_load.return_value = icu_stays_df

    patient_df = pd.DataFrame({"stay_id": [1, 2, 3, 4], "gender": ["M", "F", "M", "F"]})
    mock_add_patient.return_value = patient_df

    cssr_df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "gender": ["M", "F", "M", "F"],
            "CCSR CATEGORY 1": ["INF001", "SURG002", "INF001", "INF003"],
            "CCSR CATEGORY 1 DESCRIPTION": [
                "Infections",
                "Surgical conditions",
                "Infections",
                "Bacterial infections",
            ],
        }
    )
    mock_add_diagnosis.return_value = cssr_df

    charts_df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "gender": ["M", "F", "M", "F"],
            "CCSR CATEGORY 1": ["INF001", "SURG002", "INF001", "INF003"],
            "CCSR CATEGORY 1 DESCRIPTION": [
                "Infections",
                "Surgical conditions",
                "Infections",
                "Bacterial infections",
            ],
            220045: [98.6, 99.5, 98.6, 99.5],
            220050: [100, 99, 100, 99],
        }
    )
    mock_add_charts.return_value = charts_df

    final_df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "gender": ["M", "F", "M", "F"],
            "CCSR CATEGORY 1": ["INF001", "SURG002", "INF001", "INF003"],
            "CCSR CATEGORY 1 DESCRIPTION": [
                "Infections",
                "Surgical conditions",
                "Infections",
                "Bacterial infections",
            ],
            "Heart Rate": [98.6, 99.5, 98.6, 99.5],
            "SpO2": [100, 99, 100, 99],
        }
    )
    mock_change.return_value = final_df

    # Instantiate the pipeline
    mp = MimicLoad(mimic_root, diagnosis_codes=None)

    # Verify that each step was called once with the correct arguments
    mock_load.assert_called_once_with(mimic_root=mimic_root)
    mock_add_patient.assert_called_once_with(
        mimic_root=mimic_root, icustays_df=icu_stays_df
    )
    mock_add_diagnosis.assert_called_once_with(
        mimic_root=mimic_root, icustays_df=patient_df, diagnosis_codes=None
    )
    mock_add_charts.assert_called_once_with(
        mimic_root=mimic_root, icustays_df=cssr_df, valid_items=None
    )
    mock_change.assert_called_once_with(mimic_root=mimic_root, df=charts_df)

    pd.testing.assert_frame_equal(mp.data, final_df)

    # Check default attributes
    assert mp.target == "CCSR CATEGORY 1"
