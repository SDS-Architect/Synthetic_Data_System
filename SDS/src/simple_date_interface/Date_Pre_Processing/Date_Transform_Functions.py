import pandas as pd
import numpy as np
from datetime import datetime

### General Information
"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

def date_strip(x):
    """Small function to convert a date time to needed string and
        extract only the date, leaving time out"""

    """
    Parameters
    ----------
    x: datetime format
        Main input in lambda style function for application in function 
        date_only().

    Returns
    -------
    final: str
        The string containing only the date in format YYYY-MM-DD.
    """

    out = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    final = str(out.date())
    return final


def date_only(dataframe, date_columns):
    """Function to deal with dates by removing any time aspects from the data
        and then splitting the data out into seperate columns. It also deals 
        with NaN values that will be replaced later"""

    """
    Parameters
    ----------

    date_columns: list
        This is the list of columns in the dataframe that have date related
        information in them.

    dataframe: pd.DataFrame
        The dataframe containing the real data. 

    Returns
    -------
    working_df: pd.Dataframe
        Dataframe with NaNs filled and with only dates as strings.

    """

    ### Standard practice
    working_df = dataframe.copy()

    for column in date_columns:
        working_df[column].fillna("9999-12-31 00:00:00", inplace=True)
        working_df[column] = working_df[column].apply(date_strip)

    return working_df


def split_out_date(dataframe, date_columns):
    """Function to split each of the date day/month/year values into their
        own columns for synthesis purposes."""

    """
    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe containing the real data. 

    date_columns: list
        This is the list of columns in the dataframe that have date related
        information in them.

    Returns
    -------
    working_df: pd.Dataframe
        Dataframe with NaNs filled and with only dates as strings.

    """

    ### Standard practice
    working_df = dataframe.copy()

    processed_date_columns_reverse = {}

    working_date_columns = []

    for column in date_columns:

        ### Create Column Names
        column_day = str(column) + "_DAY"
        working_date_columns.append(column_day)
        column_month = str(column) + "_MONTH"
        working_date_columns.append(column_month)
        column_year = str(column) + "_YEAR"
        working_date_columns.append(column_year)

        ### Add values to a dictionary
        processed_date_columns_reverse[column] = [
            column_day,
            column_month,
            column_year,
        ]

        ### Assigning values
        working_df[column_day] = working_df[column].str.slice(start=8, stop=10)
        working_df[column_month] = working_df[column].str.slice(
            start=5, stop=7
        )
        working_df[column_year] = working_df[column].str.slice(start=0, stop=4)

        ### Removing redundant columns
        working_df = working_df.drop(columns=[column])

    return (working_df, processed_date_columns_reverse, working_date_columns)
