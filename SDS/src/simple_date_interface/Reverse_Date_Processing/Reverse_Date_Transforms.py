import pandas as pd
import numpy as np
from datetime import datetime

### General Information
"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

def reverse_split_out_date(dataframe, processed_date_columns):
    """Function to rejoin the string dates with '-'. """

    """
    Parameters
    ----------
    date_columns: list
        This is the list of columns in the dataframe that have date related
        information in them.

    dataframe: pd.DataFrame
        The dataframe containing the real synthetic data. 

    Returns
    -------
    working_df: pd.Dataframe
        Dataframe of synthetic data with day/month/year rejoined with '-'.

    """

    ### Standard practice
    working_df = dataframe.copy()

    for key in processed_date_columns:
        working_df[key] = working_df[processed_date_columns[key][0]].str.cat(
            working_df[processed_date_columns[key][1:3]], sep="-"
        )

        working_df = working_df.drop(columns=processed_date_columns[key])

    return working_df


def nan_test(x):
    """Small function to replace any 9999 values with NaN."""

    """
    Parameters
    ----------
    x: str
        Main input in lambda style function for application in function 
        return_date_nan().

    Returns
    -------
    x: str
        The string containing only the date in format YYYY-MM-DD or NaN.
    """

    ### Basic loop
    if "9999" in x:
        x = "NaN"

    return x


def return_date_nan(dataframe, date_columns):
    """Actual function to create NaN values in dates."""

    """
    Parameters
    ----------
    date_columns: list
        This is the list of columns in the dataframe that have date related
        information in them.

    dataframe: pd.DataFrame
        The dataframe containing the real synthetic data. 

    Returns
    -------
    working_df: pd.Dataframe
        Dataframe of synthetic data with day/month/year rejoined with '-'.

    """

    ### Standard practice
    working_df = dataframe.copy()

    ### Applying the function to various columns
    for column in date_columns:
        working_df[column] = working_df[column].apply(nan_test)

    return working_df
