from pathlib import Path

import pandas as pd

from eicu_pipeline.utils.charts import add_charts_features
from eicu_pipeline.utils.diagnosis import add_diagnosis
from eicu_pipeline.utils.patients import load_patients


class EICULoad:
    """
    Extract values from the eICU database.

    Apply a pipeline to format the data for machine learning.

    This pipeline is implemented for eICU 2.0.

    The pipeline will perform the following steps:
        1. Load all icu stays.
        2. Filter out stays of less than 12h or more than 10 days.
        3. Filter out patients under 15 years.
        4. Add diagnosis with the most priority that is closer to cutoff_h.
        5. Convert diagnosis to CCSR.
        5. Add respiratory charts.
        6. Add nurse charts.
        7. Add vital charts.

    Parameters
    ----------
    eicu_root : Path
        The path to the root of the dataset.
    diagnosis_codes : list (default = None)
        List of the CSSR diagnosis code to keep in the dataset.
        If None, all diagnosis codes will be kept.

    Attributes
    ----------
    data : pd.Dataframe
        The processed eicu dataframe containing all of the features.
    target : str
        Target column name.
    diagnosis_codes : list (default = None)
        List of the CSSR diagnosis code to keep in the dataset.
        If None, all diagnosis codes will be kept.
    extra_information : pd.Dataframe
        The description of the diagnosis value and the number of
        admissions for each diagnosis.
    """

    def __init__(self, eicu_root: Path, diagnosis_codes: list):
        """
        Init the instance and load and preprocess the eICU data.

        Parameters
        ----------
        eicu_root : str
            The path to the root of the dataset.
        diagnosis_codes : list
            CCSR phenotypes.
        """
        self.target = "CCSR CATEGORY 1"
        self.diagnosis_codes = diagnosis_codes

        self._load_preprocessed_data(eicu_root)

    def _load_preprocessed_data(
        self,
        eicu_root: str,
    ) -> pd.DataFrame:
        """
        Run the pipeline and load the data.

        Parameters
        ----------
        eicu_root : str
            The path to the root of the eicu dataset.
        """
        # Load initial data
        self.data = load_patients(eicu_root=eicu_root)

        # Add and filter diagnosis codes
        self.data = add_diagnosis(
            eicu_root=eicu_root,
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
            eicu_root=eicu_root,
            icustays_df=self.data,
        )
