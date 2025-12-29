import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class DataPreprocess:
    """
    Preprocess dataframes for ML training.

    The

    Parameters
    ----------
    main_data : pd.DataFrame
        Dataframe that will be used as main dataset for training.
    external_data : pd.DataFrame
        Dataframe that will be used as external dataset for validation.
    label : str,
        Target label. The target column will be renamed to Target.
    cat_variables : list
        List of categorical columns.
    important_variables : list
        List of columns that cannot be dropped.
    max_nan_percentage : float (default = 10)
        Maximum % of nans that each column has to meet
        in both datasets in order to be dropped.

    Attributes
    ----------
    main_data : pd.DataFrame
        Dataframe that will be used as main dataset for training.
    main_X : pd.DataFrame
        Features of the main dataframe.
    main_y : pd.DataFrame
        Target column in the main dataframe.
    external_data : pd.DataFrame
        Dataframe that will be used as external dataset for validation.
    external_X : pd.DataFrame
        Features of the external dataframe.
    external_y : pd.DataFrame
        Target column in the external dataframe.
    label : str,
        Target label. The target column will be renamed to Target.
    cat_col : list
        List of categorical columns.
    important_variables : list
        List of columns that cannot be dropped.
    target : str
        The name of the target column. Will be 'Target'.
    encoders : dict
        Dict of the encoders used for each column.
    dict_target : dict
        Mapping of the class to the encoded target value.
    """

    def __init__(
        self,
        main_data: pd.DataFrame,
        external_data: pd.DataFrame,
        label: str,
        cat_variables: list,
        important_variables: list,
        max_nan_percentage: float = 10,
    ):
        # Split main data
        self.main_data = main_data
        self.main_X = main_data.drop([label], axis=1)
        self.main_y = pd.DataFrame(main_data[label]).rename(columns={label: "Target"})

        # Split external data
        self.external_data = external_data
        self.external_X = external_data.drop([label], axis=1)
        self.external_y = pd.DataFrame(external_data[label]).rename(
            columns={label: "Target"}
        )

        # Get cat columns
        self.cat_col = cat_variables

        # Get important variables
        self.important_variables = important_variables

        # Target after being renamed
        self.target = "Target"

        # Encoders
        self.encoders = {}

        # Filter columns with more than X% of missing values
        self._feature_filter(max_nan_percentage=max_nan_percentage)

    def _feature_filter(self, max_nan_percentage: float):
        """
        Filter the features.

        Remove non-feature columns and features with a higher percentage
        of than the NaN than the allowed. This percentage must be met
        in the main dataset and in the external dataset in order to be dropped.

        It will never drop columns in the important_columns attribute.

        It also encodes the categorical features.

        Parameters
        ----------
        max_nan_percentage : float
            The maximum percentage of Nan allowed to keep features in the final dataset.
        """
        # Encode target
        le = LabelEncoder()
        self.main_y[self.target] = le.fit_transform(self.main_y[self.target])
        self.external_y[self.target] = le.transform(self.external_y[self.target])
        self.encoders[self.target] = le

        # Create the mapping from the class to the encoded value
        keys = self.encoders[self.target].classes_
        values = self.encoders[self.target].transform(
            self.encoders[self.target].classes_
        )
        dictionary_class = dict(zip(keys, values))
        self.dict_target = dictionary_class

        # Fill nan and do binary encoding
        main_categorical_columns = self.main_X[self.cat_col].copy()
        external_categorical_columns = self.external_X[self.cat_col].copy()

        for column in main_categorical_columns.columns:
            # Impute Nan with constant value
            main_categorical_columns[column] = main_categorical_columns[column].fillna(
                "UNKNOWN"
            )
            external_categorical_columns[column] = external_categorical_columns[
                column
            ].fillna("UNKNOWN")

            # Encode category
            oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            main_categorical_columns[column] = oe.fit_transform(
                main_categorical_columns[[column]]
            ).ravel()

            external_categorical_columns[column] = oe.transform(
                external_categorical_columns[[column]]
            ).ravel()

            self.encoders[column] = oe

        # Drop numerical columns with too many nan values
        self.main_X = self.main_X.drop(columns=main_categorical_columns.columns)
        self.external_X = self.external_X.drop(
            columns=external_categorical_columns.columns
        )

        self._filter_nans(max_nan_percentage=max_nan_percentage)

        # Add categorical features back
        self.main_X[main_categorical_columns.columns] = main_categorical_columns
        self.external_X[external_categorical_columns.columns] = (
            external_categorical_columns
        )

    def _filter_nans(self, max_nan_percentage) -> pd.DataFrame:
        """
        Filter the features due NaN percentage.

        Remove non-feature columns and features with a higher percentage
        of than the NaN than the allowed. This percentage must be met
        in the main dataset and in the external dataset in order to be dropped.

        It will never drop columns in the important_columns attribute.

        Parameters
        ----------
        max_nan_percentage : float
            The maximum percentage of Nan allowed to keep features in the final datasets.
        """
        # Calculate the columns to drop
        drop_nan_columns = []
        for column in self.main_X.columns:
            # If the column is important
            important_columns = column in self.important_variables
            # If the column meets the nan criteria in the internal dataset
            threshold_met_main = (
                self.main_X[column].isna().mean() * 100 >= max_nan_percentage
            )
            # If the columns meets the nan criteria in the external dataset
            threshold_met_external = (
                self.external_X[column].isna().mean() * 100 >= max_nan_percentage
            )
            # Drop unimportant columns that meet the criteria in both datasets
            if (
                (not important_columns)
                & (threshold_met_main)
                & (threshold_met_external)
            ):
                drop_nan_columns.append(column)

        # Drop
        self.main_X = self.main_X.drop(columns=drop_nan_columns)
        self.external_X = self.external_X.drop(columns=drop_nan_columns)
