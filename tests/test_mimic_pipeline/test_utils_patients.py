from pathlib import Path
from unittest.mock import patch

import pandas as pd

from mimic_pipeline.utils.patients import (
    add_patient_features,
    filter_age_under_15,
    filter_death_during_stay,
    filter_multiple_icu_stay_per_admission,
    time_to_death,
)


def test_filter_multiple_icu_stay_per_admission():
    """
    Check that `filter_multiple_icu_stay_per_admission` works with repeated hadm_id.
    """
    test_df = pd.DataFrame(
        {
            "hadm_id": [1, 1, 2, 3, 3],
            "stay_id": [10, 11, 20, 30, 31],
        }
    )

    out = filter_multiple_icu_stay_per_admission(merged_df=test_df)

    expected = pd.DataFrame(
        {
            "hadm_id": [2],
            "stay_id": [20],
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        expected,
        check_dtype=True,
        check_index_type=False,
    )


def test_filter_multiple_icu_stay_per_admission_without_repetition():
    """
    Check that `filter_multiple_icu_stay_per_admission` works with no repeated hadm_id.
    """
    test_df = pd.DataFrame(
        {
            "hadm_id": [1, 2, 3, 4, 5],
            "stay_id": [10, 20, 30, 40, 50],
        }
    )

    out = filter_multiple_icu_stay_per_admission(merged_df=test_df)

    pd.testing.assert_frame_equal(
        out,
        test_df,
        check_dtype=True,
        check_index_type=False,
    )


def test_filter_death_during_stay():
    """
    Check that `filter_death_during_stay` works as expected.
    """
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "intime": pd.to_datetime(
                [
                    "2025-04-01 08:00",
                    "2025-04-02 09:00",
                    "2025-04-03 10:00",
                    "2025-04-04 11:00",
                ]
            ),
            "outtime": pd.to_datetime(
                [
                    "2025-04-01 20:00",
                    "2025-04-02 19:00",
                    "2025-04-03 11:00",
                    "2025-04-05 12:00",
                ]
            ),
            "deathtime": pd.to_datetime(
                ["2025-04-01 12:00", None, "2025-04-03 10:30", "2025-04-06 12:00"]
            ),
        }
    )

    out = filter_death_during_stay(df)

    expected = pd.DataFrame(
        {
            "stay_id": [2, 4],
            "intime": pd.to_datetime(["2025-04-02 09:00", "2025-04-04 11:00"]),
            "outtime": pd.to_datetime(
                [
                    "2025-04-02 19:00",
                    "2025-04-05 12:00",
                ]
            ),
            "deathtime": pd.to_datetime([None, "2025-04-06 12:00"]),
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        expected,
        check_dtype=True,
        check_index_type=False,
    )


def test_filter_age_under_15():
    """
    Check that `filter_age_under_15` works as expected.
    """
    df = pd.DataFrame(
        {
            "anchor_age": [10, 20, 30],
            "anchor_year": [2000, 2000, 2000],
            "icu_year": [2004, 2020, 2030],
            "stay_id": [1, 2, 3],
        }
    )

    out = filter_age_under_15(df)

    expected = pd.DataFrame(
        {
            "stay_id": [2, 3],
            "icu_age": [40, 60],
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        expected,
        check_dtype=True,
        check_index_type=False,
    )


def test_time_to_death():
    """
    Check that `time_to_death` computes the time to death correctly.
    """
    df = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "intime": [
                pd.Timestamp("2025-04-01 08:00:00"),
                pd.Timestamp("2025-04-02 08:00:00"),
                pd.Timestamp("2025-04-03 08:00:00"),
                pd.Timestamp("2025-04-04 08:00:00"),
                pd.Timestamp("2025-04-05 08:00:00"),
            ],
            "deathtime": [
                pd.Timestamp("2025-04-01 10:00:00"),
                pd.NaT,  # No death for patient 2
                pd.Timestamp("2025-04-03 12:00:00"),
                pd.Timestamp("2025-04-04 07:00:00"),  # Died before intime
                pd.Timestamp("2025-04-05 08:00:00"),
            ],
        }
    )

    out = time_to_death(df)

    expected = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 5],
            "intime": [
                pd.Timestamp("2025-04-01 08:00:00"),
                pd.Timestamp("2025-04-02 08:00:00"),
                pd.Timestamp("2025-04-03 08:00:00"),
                pd.Timestamp("2025-04-05 08:00:00"),
            ],
            "deathtime": [
                pd.Timestamp("2025-04-01 10:00:00"),
                pd.NaT,
                pd.Timestamp("2025-04-03 12:00:00"),
                pd.Timestamp("2025-04-05 08:00:00"),
            ],
            "Time_to_death_(h)": [2.0, "No death", 4.0, 0.0],
        }
    )

    pd.testing.assert_frame_equal(
        out.reset_index(drop=True),
        expected,
        check_dtype=True,
        check_index_type=False,
    )


def test_add_patient_features_default():
    """
    Check that `add_patient_features` works as expected with default flags.

    Given the expected input format it returns the expected output format.
    And calls all of the filters.

    subject_id 1 is valid.
    subject_id 2 is not valid, it has multiple icu_stays in one admission.
    subject_id 3 is not valid, under 15 years old.
    subject_id 4 is valid.
    subject_id 5 is valid, it has two admissions.
    subject_id 6 is valid and died during icu_stay.
    """
    mock_icustays_df = pd.DataFrame(
        {
            "subject_id": [1, 2, 2, 3, 4, 5, 5, 6],
            "hadm_id": [100, 200, 200, 300, 400, 500, 501, 600],
            "stay_id": [1000, 2000, 2001, 3000, 4000, 5000, 5001, 6000],
            "intime": [
                "2025-04-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 10:00:00",
            ],
            "outtime": [
                "2025-04-05 08:00:00",
                "2025-04-05 08:00:00",
                "2025-05-05 08:00:00",
                "2025-04-04 08:00:00",
                "2025-05-04 08:00:00",
                "2025-04-03 08:00:00",
                "2025-05-05 08:00:00",
                "2025-04-03 01:00:00",
            ],
        }
    )
    mock_icustays_df["intime"] = pd.to_datetime(mock_icustays_df["intime"])
    mock_icustays_df["outtime"] = pd.to_datetime(mock_icustays_df["outtime"])
    mock_icustays_df["icu_year"] = mock_icustays_df["intime"].dt.year

    mock_patients_df = pd.DataFrame(
        {
            "subject_id": [1, 2, 2, 3, 4, 5, 6],
            "anchor_age": [20, 10, 10, 10, 40, 25, 20],
            "anchor_year": [2000, 2000, 2000, 2023, 2020, 2015, 2023],
            "gender": ["M", "M", "M", "F", "F", "M", "F"],
        }
    )

    mock_admissions_df = pd.DataFrame(
        {
            "hadm_id": [100, 200, 200, 300, 400, 500, 501, 600],
            "deathtime": [
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.Timestamp("2025-04-02 10:00:00"),
            ],
            "marital_status": [
                "SINGLE",
                "SINGLE",
                "SINGLE",
                "SINGLE",
                "MARRIED",
                "SINGLE",
                "MARRIED",
                "MARRIED",
            ],
            "race": [
                "WHITE",
                "WHITE",
                "WHITE",
                "LATIN",
                "WHITE",
                "WHITE",
                "WHITE",
                "LATIN",
            ],
        }
    )

    expected = pd.DataFrame(
        {
            "subject_id": [1, 4, 5, 5, 6],
            "hadm_id": [100, 400, 500, 501, 600],
            "stay_id": [1000, 4000, 5000, 5001, 6000],
            "intime": pd.to_datetime(
                [
                    "2025-04-01 08:00:00",
                    "2025-05-01 08:00:00",
                    "2025-04-01 08:00:00",
                    "2025-05-01 08:00:00",
                    "2025-04-01 10:00:00",
                ]
            ),
            "gender": ["M", "F", "M", "M", "F"],
            "icu_age": [45, 45, 35, 35, 22],
            "marital_status": ["SINGLE", "MARRIED", "SINGLE", "MARRIED", "MARRIED"],
            "race": ["WHITE", "WHITE", "WHITE", "WHITE", "LATIN"],
            "Time_to_death_(h)": ["No death", "No death", "No death", "No death", 24.0],
        }
    )

    with (
        patch("pandas.read_csv", side_effect=[mock_patients_df, mock_admissions_df]),
        patch(
            "mimic_pipeline.utils.patients.filter_age_under_15",
            wraps=filter_age_under_15,
        ) as mock_filter_age,
        patch(
            "mimic_pipeline.utils.patients.filter_death_during_stay",
            wraps=filter_death_during_stay,
        ) as mock_filter_death,
        patch(
            "mimic_pipeline.utils.patients.filter_multiple_icu_stay_per_admission",
            wraps=filter_multiple_icu_stay_per_admission,
        ) as mock_filter_multiple,
        patch(
            "mimic_pipeline.utils.patients.time_to_death",
            wraps=time_to_death,
        ) as mock_time_to_death,
    ):
        out = add_patient_features(Path("wingardium leviosa"), mock_icustays_df)

        pd.testing.assert_frame_equal(
            out.reset_index(drop=True),
            expected,
            check_dtype=True,
            check_index_type=False,
        )

        assert mock_filter_age.called
        assert mock_filter_death.assert_not_called
        assert mock_time_to_death.called
        assert mock_filter_multiple.called


def test_add_patient_features_rm_death():
    """
    Check that `add_patient_features` works as expected with rm_early_die=True.

    Given the expected input format it returns the expected output format.
    And calls all of the filters.

    subject_id 1 is valid.
    subject_id 2 is not valid, it has multiple icu_stays in one admission.
    subject_id 3 is not valid, under 15 years old.
    subject_id 4 is valid.
    subject_id 5 is valid, it has two admissions.
    subject_id 6 is not valid died during icu_stay.
    """
    mock_icustays_df = pd.DataFrame(
        {
            "subject_id": [1, 2, 2, 3, 4, 5, 5, 6],
            "hadm_id": [100, 200, 200, 300, 400, 500, 501, 600],
            "stay_id": [1000, 2000, 2001, 3000, 4000, 5000, 5001, 6000],
            "intime": [
                "2025-04-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 08:00:00",
                "2025-05-01 08:00:00",
                "2025-04-01 10:00:00",
            ],
            "outtime": [
                "2025-04-05 08:00:00",
                "2025-04-05 08:00:00",
                "2025-05-05 08:00:00",
                "2025-04-04 08:00:00",
                "2025-05-04 08:00:00",
                "2025-04-03 08:00:00",
                "2025-05-05 08:00:00",
                "2025-04-03 01:00:00",
            ],
        }
    )
    mock_icustays_df["intime"] = pd.to_datetime(mock_icustays_df["intime"])
    mock_icustays_df["outtime"] = pd.to_datetime(mock_icustays_df["outtime"])
    mock_icustays_df["icu_year"] = mock_icustays_df["intime"].dt.year

    mock_patients_df = pd.DataFrame(
        {
            "subject_id": [1, 2, 2, 3, 4, 5, 6],
            "anchor_age": [20, 10, 10, 10, 40, 25, 20],
            "anchor_year": [2000, 2000, 2000, 2023, 2020, 2015, 2023],
            "gender": ["M", "M", "M", "F", "F", "M", "F"],
        }
    )

    mock_admissions_df = pd.DataFrame(
        {
            "hadm_id": [100, 200, 200, 300, 400, 500, 501, 600],
            "deathtime": [
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.Timestamp("2025-04-02 10:00:00"),
            ],
            "marital_status": [
                "SINGLE",
                "SINGLE",
                "SINGLE",
                "SINGLE",
                "MARRIED",
                "SINGLE",
                "MARRIED",
                "MARRIED",
            ],
            "race": [
                "WHITE",
                "WHITE",
                "WHITE",
                "LATIN",
                "WHITE",
                "WHITE",
                "WHITE",
                "LATIN",
            ],
        }
    )

    expected = pd.DataFrame(
        {
            "subject_id": [1, 4, 5, 5],
            "hadm_id": [100, 400, 500, 501],
            "stay_id": [1000, 4000, 5000, 5001],
            "intime": pd.to_datetime(
                [
                    "2025-04-01 08:00:00",
                    "2025-05-01 08:00:00",
                    "2025-04-01 08:00:00",
                    "2025-05-01 08:00:00",
                ]
            ),
            "gender": ["M", "F", "M", "M"],
            "icu_age": [45, 45, 35, 35],
            "marital_status": ["SINGLE", "MARRIED", "SINGLE", "MARRIED"],
            "race": ["WHITE", "WHITE", "WHITE", "WHITE"],
        }
    )

    with (
        patch("pandas.read_csv", side_effect=[mock_patients_df, mock_admissions_df]),
        patch(
            "mimic_pipeline.utils.patients.filter_age_under_15",
            wraps=filter_age_under_15,
        ) as mock_filter_age,
        patch(
            "mimic_pipeline.utils.patients.filter_death_during_stay",
            wraps=filter_death_during_stay,
        ) as mock_filter_death,
        patch(
            "mimic_pipeline.utils.patients.filter_multiple_icu_stay_per_admission",
            wraps=filter_multiple_icu_stay_per_admission,
        ) as mock_filter_multiple,
        patch(
            "mimic_pipeline.utils.patients.time_to_death",
            wraps=time_to_death,
        ) as mock_time_to_death,
    ):
        out = add_patient_features(
            Path("wingardium leviosa"), mock_icustays_df, rm_early_die=True
        )

        pd.testing.assert_frame_equal(
            out.reset_index(drop=True),
            expected,
            check_dtype=True,
            check_index_type=False,
        )

        assert mock_filter_age.called
        assert mock_filter_death.called
        assert mock_time_to_death.assert_not_called
        assert mock_filter_multiple.called
