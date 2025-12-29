def create_chart_features(charts_df, stay_col, variable_col, value_col):
    """
    Create wide-format chart features with last, max, min, mean, median for each variable.

    Parameters
    ----------
    charts_df : pd.DataFrame
        Dataframe containing chart data. Filtered to cutoff.
    stay_col : str
        Column name for ICU stay ID (e.g., "patientunitstayid").
    variable_col : str
        Column name for the variable/measurement label.
    value_col : str
        Column name for the measurement values.

    Returns
    -------
    pd.DataFrame
        Resulting dataframe with columns like last_<var>, max_<var>, min_<var>, mean_<var>, median_<var>.
    """
    # Compute summary statistics per (stay, variable)
    stats = (
        charts_df.groupby([stay_col, variable_col])[value_col]
        .agg(last="last", max="max", min="min", mean="mean", median="median")
        .reset_index()
    )

    # Pivot all metrics into wide format
    wide_df = stats.pivot(index=stay_col, columns=variable_col)[
        ["last", "max", "min", "mean", "median"]
    ]

    # Flatten MultiIndex columns
    wide_df.columns = [f"{metric}_{var}" for metric, var in wide_df.columns]
    wide_df = wide_df.reset_index()

    return wide_df
