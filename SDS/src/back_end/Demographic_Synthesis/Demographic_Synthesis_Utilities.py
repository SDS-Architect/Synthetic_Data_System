#!/usr/bin/env python
# coding: utf-8

# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy

"""
This file is for setting up groups of similar records in the dataset for
use by later methods. These functions create a 'Combi' column that
combine important columns to make groups and then remove low count groups
and/or records iteratively.
"""

"""
Please cite this system as: 

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""


def separate_low_counts(data, threshold, print_statement):

    """ Iterates over dataframe multiple times to remove low count values.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe to have it's low count instances removed.

    threshold: integer
       Specifices counts of values below or equal to it are removed.


    Returns
    -------
    output_df: pd.DataFrame
        Pandas dataframe that has no counts less than the threshold.
    """

    # take a copy of the data
    output_df = data.copy()
    cols = data.columns.values

    while True:

        iter_shape_list = []

        for col in cols:
            counts = output_df[col].value_counts()
            countsU = counts[counts <= threshold].index.values
            output_df.drop(
                output_df[output_df.loc[:, col].isin(countsU)].index,
                inplace=True,
            )

            iter_shape_list.append(output_df.shape)

            if print_statement == True:
                print(iter_shape_list)

        if iter_shape_list[0] == iter_shape_list[-1]:
            return output_df


def create_filtered_index(
    dataframe, index_cols, remove_small_vals, print_statement
):

    """ Creates a group index of the dataframe for ML training partitioning.
            Also automatically removes low counts variables.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe to be worked on.

    index_cols: list)
        Comma seperated list of columns in dataset to create synthetic labels.
        The FIRST value in list is the MOST IMPORTANT ONE as this is what all
        others link with for form the combi column.

    remove_small_vals: integer
        Specifices counts of values below or equal to it are removed.


    Returns
    -------
        output_df: pd.DataFrame
            Pandas dataframe that has no missing/NaN values.


        process_list: list
            A list of Combi column values to be ordered later in process.
    """

    # Output information
    print("\n")
    print("Making Combi column")
    print("\n")

    # Take a copy of the data
    output_df = dataframe.copy()

    # Create the main link col as 1st value in index_cols
    link_col = index_cols[0]

    # Remove link_col form index_cols
    index_cols.pop(0)

    # Create a Combi column for filtering
    output_df["Combi"] = output_df[link_col].str.cat(output_df[index_cols])

    # Double check in STRING format!
    output_df["Combi"] = output_df["Combi"].astype(str)

    # Get a list of groups of combinations
    index_values = pd.DataFrame(output_df.Combi.value_counts())

    # Reset index to get values
    index_values = index_values.reset_index()

    # Filters out low count values
    print("Separating low count values: ")
    output_df = separate_low_counts(
        output_df, remove_small_vals, print_statement
    )

    # Let's split the catgeories into lists for easier training
    print("\n")
    print("Creating index of values")

    process_list = list(index_values["index"])

    # Return out values
    return (output_df, process_list)
