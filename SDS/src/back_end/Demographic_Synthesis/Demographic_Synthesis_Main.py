# coding: utf-8

# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy

from SDS.src.back_end.Demographic_Synthesis.Demographic_Synthesis_Diff_Piv import (
    prob_dataframe_gen_with_dp,
)

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This file contains the main method to create a synthetic demographic
based off of sampling methods in src/Demographic_Synthesis_Diff_Piv.py.

The functions in this file iterate over various columns in the real data so as
to build up the synthetic demographic dataframe to be passed on to the ML
synthesis methods.
"""


def col_tuple_pair_gen(var_list):

    """Creates pairs of tuples for conditional probability across 2 columns
            at a time

    Parameters
    ----------
    var_list: list
        A list of demographic based dataframe column names to be processed.


    Returns
    -------
    item_tuple: list, tuples
        A list of pairs (tuple) of columns.
    """

    # Output Tuple
    item_tuple = []

    # Loop to create pairs
    for col_1, col_2 in zip(var_list[::1], var_list[1::1]):
        pairs = (col_1, col_2)
        item_tuple.append(pairs)

    # Return values
    return item_tuple


def Demographic_Synthesis(
    real_dataframe,
    demo_vars,
    cat_demo_vars,
    original_df_size,
    synth_df_size,
    percent,
):

    """Synthesises using conditional probability across demographic cols.

    Parameters
    ----------
    real_dataframe: pd.DataFrame
        Dataframe to be worked on.

    demographic_variables: list
        These are the variables in the real dataset (specified by user) that
        relate to the background information (demographic) of a sample. These
        are synthesised first as they can only take a small amount of discrete
        values i.e. people can't have an age of 1000 years.

    cat_demo_vars: list
        Any variables that are categorical and demographic. This is created
        by a previous function output.

    original_df_size: integer
        The size of the current group of real data.

    synth_df_size: integer
        The size to create the current group of synthetic data.

    percent: float
        The percentage size of the current group in relation to the original
        data.


    Returns
    -------
    probs_df: pd.DataFrame
        Pandas dataframe that has no missing/NaN values.
    """

    # Subset out data needed from main dataframe
    real_data_sub_df = real_dataframe[demo_vars]

    # Create the tuples needed for probability calculation
    col_tuples = col_tuple_pair_gen(demo_vars)

    # Insertion of noise function
    probs_df = prob_dataframe_gen_with_dp(
        real_data_sub_df, original_df_size, synth_df_size, col_tuples, percent
    )

    # Refine length of data down
    probs_df = probs_df[demo_vars]

    # Resetting Categroical variables
    for col_var in cat_demo_vars:
        real_data_sub_df[col_var] = real_data_sub_df[col_var].astype(str)

    return probs_df
