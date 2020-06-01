#!/usr/bin/env python
# coding: utf-8

# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Remove non-needed warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This file is for all general utilities for the synthetic data research
project. These functions are designed to do small but repetitive tasks
such as removing missing values or loading files.
"""


def open_file(open_file_name=None):

    """ One-click opening and loading of file.

    Parameters
    ----------
    None.


    Returns
    -------
    main_file: pd.DataFrame
        A pandas dataframe of the selected file.
    """

    if open_file_name is None:
        # Load a Tk Inter instance
        root = Tk()
        root.withdraw()

        # Select the file
        file_path = askopenfilename()

        # Destroy the root/open window
        root.destroy()

        try:
            # Transform into PD DataFrame
            main_file = pd.read_csv(file_path)

            # Transform into PD DataFrame
            main_file = pd.read_csv(file_path)

            # Return the output
            return main_file

        except:
            raise TypeError("ERROR: Select a .CSV file please")

    try:
        # Transform into PD DataFrame
        main_file = pd.read_csv(open_file_name)

        # Return the output
        return main_file

    except:
        raise TypeError("ERROR: Select a .CSV file please")


def string_cut(dataframe=None, length_string=None, col_list=None):

    """Cuts all strings in col_list down to the length in specified by
        length_string.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe to be worked on.

    length_string: integer
        Length of final string.

    col_list: list
        Comma seperated list of columns in dataset to be cut to length desired.


    Returns
    -------
    working_df: pd.DataFrame
        Pandas dataframe that has no missing/NaN values.
    """

    ### Checks if data is useable
    if any([dataframe is None, length_string is None, col_list is None]):

        raise TypeError(
            "One of your values is empty, you can only use"
            + "this function if you have a dataframe, column list and a length value"
        )

    ### Checks all data are correct types
    if all(
        [
            type(dataframe) is pd.DataFrame,
            type(length_string) is int,
            type(col_list) is list,
        ]
    ):

        ### Check column names are correct
        if all(elem in list(dataframe) for elem in col_list):

            # Take a copy of the data
            working_df = dataframe.copy()

            # Iterate over columns
            for column in col_list:
                working_df[column] = working_df[column].str.slice(
                    stop=length_string
                )

            return working_df

        else:
            raise TypeError(
                "The columns in col_list don't match those in"
                + " your dataframe"
            )

    raise TypeError(
        "One of your variables is not the correct type - \
        e.g. check you have an integer for length_string"
    )


def synth_label_create(dataframe, synth_label_cols_stucture, synth_label_cols):

    """ Creates a synthetic vairables to help ML training later. It is based
            on a cut of a categorical (string) variable.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe to be worked on.

    length: integer or list
        Length of final label and how much of original string to be used to
        create label.

    col_list: list
        Comma seperated list of columns in dataset to create synthetic labels.


    Returns
    -------
    working_df: pd.DataFrame
        Pandas dataframe that has no missing/NaN values.

    synth_column_name_list: list
        A list of synthetic label columns that can be removed later.
    """

    # Take a copy of the data
    working_df = dataframe.copy()

    # initialise a number value
    number = 0

    # Return list of synthetic columns
    synth_column_name_list = []

    # Iterate over columns
    for column, length in zip(synth_label_cols, synth_label_cols_stucture):
        new_name = "Synth_Label_" + str(number)
        synth_column_name_list.append(new_name)
        working_df[new_name] = working_df[column].str.slice(stop=length)

        number += 1

    return (working_df, synth_column_name_list)


def NaN_Handle_Cat(dataframe):

    """ Replaces missing/NaN with holding value with '{column_name}
            + _missing'.

    Parameters
    ----------

    dataframe: pd.DataFrame
        Dataframe to have it's missing/NaN values transformed with column
        names.


    Returns
    -------
    miss_value_list: list
        A list of columns that contained missing values to be processed later.

    dataframe: pd.Dataframe
        Pandas dataframe that has no missing/NaN values.
    """

    # Create a list to remove from final synth dataset
    miss_value_list = []

    # Loop and replace values in DF with column name and '_Missing'
    for column in list(dataframe):
        missing_name = str(column + "_Missing")
        miss_value_list.append(missing_name)
        dataframe[column] = dataframe[column].fillna(missing_name)

    # Return list of missing values for later
    return (miss_value_list, dataframe)


def reverse_NaN(dataframe, m_values_list, removal_columns):

    """ Replaces '{column_name} + _missing' back into '', (equivalent to NaN).

    Parameters
    ----------

    dataframe: pd.DataFrame
        Dataframe to have it's '_Missing' transformed to ' '.

    miss_value_list: list
        A list of columns that contained missing values.


    Returns
    -------
    miss_value_list: list
        A list of columns that contained missing values to be processed later.

    dataframe: pd.Dataframe
        Pandas dataframe that has no missing/NaN values.
    """

    # Take a copy of the data
    working_df = dataframe.copy()

    ### Create proper list
    list_removal = []
    for value in removal_columns:
        out = value + "_Missing"
        list_removal.append(out)

    ### Filter out synth columns
    m_values_list = [x for x in m_values_list if x not in list_removal]

    ### Loop to replace values
    for column in m_values_list:
        target_string = str(column)

        target_column = str(target_string.replace("_Missing", ""))

        working_df.loc[
            (working_df[target_column] == target_string), target_column
        ] = " "

    return working_df
