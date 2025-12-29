import warnings
from pathlib import Path

import pandas as pd

from mimic_pipeline.utils.charts import add_charts_features, change_itemid_to_item_name
from mimic_pipeline.utils.diagnosis import (
    add_diagnosis,
)
from mimic_pipeline.utils.icustays import load_icustays
from mimic_pipeline.utils.patients import add_patient_features

warnings.simplefilter(action="ignore", category=pd.errors.DtypeWarning)


class MimicLoad:
    """
    Extract values from the mimic database and preprocess them.

    Apply a pipeline to format the data for machine learning.

    This pipeline is implemented for MIMIC-IV 2.2.

    The mimic_pipeline will perform the following steps:
        1. Load all icu stays.
        2. Filter out stays of less than 12h or more than 10 days.
        3. Add patient features to the icu stays.
        4. Filter out patients under 15 years old and optionally, when death
            time is during the icu stay.
        5. Filter out patient admissions with more than one icu stay.
        6. Add the diagnosis of with the most priority and transform it to CSSR.
        7. Add chart features.

    Parameters
    ----------
    mimic_root : Path
        The path to the root of the mimic dataset.
    diagnosis_codes : list (default = None)
        List of the CSSR diagnosis code to keep in the dataset.
        If None, all diagnosis codes will be kept.

    Attributes
    ----------
    data : pd.Dataframe
        The processed mimic dataframe containing all of the features.
    target : str
        Target column name.
    diagnosis_codes : list (default = None)
        List of the CSSR diagnosis code to keep in the dataset.
        If None, all diagnosis codes will be kept.
    extra_information : pd.Dataframe
        The description of the diagnosis value and the number of
        admissions for each diagnosis.
    """

    def __init__(self, mimic_root: Path, diagnosis_codes: list):
        """
        Init the instance and load and preprocess the data mimic data.

        Parameters
        ----------
        mimic_root : Path
            The path to the root of the mimic dataset.
        diagnosis_codes : list (default = None)
            List of the CSSR diagnosis code to keep in the dataset.
        """
        self.target = "CCSR CATEGORY 1"
        self.diagnosis_codes = diagnosis_codes

        self._load_preprocessed_data(mimic_root=mimic_root)

    def _load_preprocessed_data(self, mimic_root: Path) -> pd.DataFrame:
        """
        Run the mimic_pipeline and load the data.

        Parameters
        ----------
        mimic_root : Path
            The path to the root of the mimic dataset.
        """
        # Load initial data
        self.data = load_icustays(mimic_root=mimic_root)

        # Add and filter patient features
        self.data = add_patient_features(
            mimic_root=mimic_root,
            icustays_df=self.data,
        )

        # Add and filter diagnosis codes
        self.data = add_diagnosis(
            mimic_root=mimic_root,
            icustays_df=self.data,
            diagnosis_codes=self.diagnosis_codes,
        )

        # Get extra_information = diagnosis description and count
        self.extra_information = (
            self.data.groupby(self.target)
            .agg(
                CCSR_CATEGORY_1_DESCRIPTION=("CCSR CATEGORY 1 DESCRIPTION", "first"),
                ADMISSION_COUNT=(self.target, "size"),
            )
            .reset_index()
        )
        # Sort by count
        self.extra_information = self.extra_information.sort_values(
            by="ADMISSION_COUNT", ascending=False
        )

        self.data = add_charts_features(
            mimic_root=mimic_root,
            icustays_df=self.data,
            valid_items=None,
        )
        # Change feature names
        self.data = change_itemid_to_item_name(mimic_root=mimic_root, df=self.data)
