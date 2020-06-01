# coding: utf-8

# Standard Libraries
import pdb

# External Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy

from SDS.src.back_end.Machine_Learning_Synthesis.ML_Synthesis_Utilities import (
    looping_var_list,
    categorical_var_list_creation,
)

from SDS.src.back_end.Tree_Methods.Tree_Functions_debug import tree_synth

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This is the main function that creates synthetic data by ingesting the data
made by traditional probability sampling and builds on it using tree methods.

IMPORTANT: Just now there is only the method for categorical data. There is a
    function to deal with continous data but it needs further testing before
    being added in here.
"""


def Machine_Learning_Synthesis(
    feed_real_data_sub_df,
    prob_synth_df,
    demographic_vars,
    ml_variables,
    categorical_variables,
    GPU_IDs,
    seed_training,
    mapping_dict,
    numeric_group_vars,
):

    """Iterates over real data and conditionally created demographic data to
            begin synthesising out the non-demographic parts of the real data.

    Parameters
    ----------
    feed_real_data_sub_df: pd.DataFrame
        The real data in dataframe format.

    prob_synth_df: pd.DataFrame
        Demographic variables that were synthesised by Demographic_Synthesis.

    demographic_vars: list
        A list of demographic variables.

    ml_variables: list
        A list of non-demographic variables that need to be synthesised by
        this method.

    categorical_variables: list
        A list of columns from the real data that are categorical. Numeric
        columns are assumed to be those that are not in this list.

    GPU_IDs: integer, list
        If training with more than 1 GPU put numbers in a list. Else, just
        assumes GPU ID is 0.

    seed_training: integer
        Integer value for train/test split of the data by tree_synth.

    mapping_dict: dict
        A dictionary of labels and their original values for EACH column.
        Used for checking the data types for error correction.


    numeric_group_vars: list, optional
        Used to check type of synthesis.


    Returns
    -------
    prob_synth_df: pd.DataFrame
        Pandas dataframe of fully synthesised values for a given subset of the
        overall real data.

    """

    # Ensuring seed for train/test split is set
    if seed_training is None:
        seed_training = 123

    # Testing correct input
    if not isinstance(seed_training, int):
        raise ValueError("Please only select Integers")

    # Deal with negative numbers
    seed_training = abs(seed_training)

    # Get number of categorical variables
    number_cat_vars = len(demographic_vars)

    # Iterating list for dataframe
    iterating_list = looping_var_list(demographic_vars, ml_variables)

    # Need to get synth_df in same order as real_df

    for list_number in range(0, len(iterating_list)):

        # Subset out real data needed at this point
        work_df = feed_real_data_sub_df[iterating_list[list_number]]

        # Generate a target variable
        target_var = iterating_list[list_number][-1]

        # Function to create needed categorical lists and index values
        cat_out, working_df_names = categorical_var_list_creation(
            work_df, categorical_variables, number_cat_vars
        )

        # Enforcing synthetic data compliance
        prob_synth_df = prob_synth_df[working_df_names]

        try:
            missing_map = mapping_dict[target_var][
                str(target_var + "_Missing")
            ]

            testing_value = all((work_df[target_var] == missing_map))

        except KeyError:
            testing_value = False

        print("\n")
        print("\n")
        message = str("The current Target Variable is: " + target_var)
        print(message)
        print("-" * len(message))

        if (
            len(work_df[target_var].value_counts()) <= 1
            and testing_value == True
        ):
            prob_synth_df[target_var] = str(missing_map)

        if (
            len(work_df[target_var].value_counts()) <= 1
            and testing_value == False
        ):

            """Take first column value"""
            prob_synth_df[target_var] = str(work_df[target_var].iloc[0])

        if len(work_df[target_var].value_counts()) > 1:
            prob_synth_df[target_var] = tree_synth(
                prob_synth_df,
                work_df,
                working_df_names,
                target_var,
                seed_training,
                cat_out,
                GPU_IDs,
            )

    return prob_synth_df
