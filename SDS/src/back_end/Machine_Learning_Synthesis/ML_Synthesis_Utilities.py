# coding: utf-8

# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy

"""
Please cite this system as: 

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This file contains functions to loop and create lists of column variables and
other needed information for the Machine_Learning_Synthesis_cat methods.
"""


def looping_var_list(demo_vars, ML_vars):

    """ Designed to create lists to subset the real data as part of synthesis
            in Machine_Learning_Synthesis_Cat().


    Parameters
    ----------
    demo_vars: list
        Demographic categorical variables derived from real data dataframe.

    ML_vars: list
        Non-demogrpahic variables that need to be synthesised in an
        iterative process.


    Returns
    -------
    master_iter_list: list
        A list of lists (tuple) that contains vairables that increment per
        list so builds up whole dataframe subset one column at a time.


    Related Functions
    -----------------
    Machine_Learning_Synthesis_cat()

    """

    # Create an empty list
    master_iter_list = []

    # Create sublist
    iterating_list = demo_vars

    # Loop to create a list
    for variable in ML_vars:

        # Adding on target variable
        output = iterating_list + [variable]
        iterating_list = output

        # Append the above list to full one
        master_iter_list.append(output)

    return master_iter_list


def categorical_var_list_creation(
    work_df, categorical_variables, number_cat_vars
):

    """ Designed to help get names and index positions of columns in
            dataframes for Machine_Learning_Synthesis_cat().


    Parameters
    ----------
    work_df: pd.dataframe
        The dataframe form which to derive column names and indexes.

    categorical_variables: list, str
        Categorical variables derived from real data dataframe

    number_cat_vars: integer
        The total number of the categorical variables.


    Returns:
        cat_out: list, int 
            Categorical variables numeric indexes for use in tree_synth 
            function.

        working_df_names: list
            List of dataframe vairables minus target variable.


    Note
    ----
    cat_out and working_df_names remove a column as part of calculation as
    this tree_synth() automatically assumes the target vairable is
    categorical.

    """

    # Getting column names needed
    working_df_names = list(work_df)

    # Categorical vairables
    cat_var_list = list(
        set(categorical_variables).intersection(working_df_names)
    )

    # Numeric Indexes for categorical cols for tree function
    numeric_cat_list = list(set(working_df_names) - set(cat_var_list))

    # Loop to create categorical numeric indexes
    for i in cat_var_list:
        work_df[i] = work_df[i].astype(str)

        numeric_cat_values = []
        for i in numeric_cat_list:
            values = work_df.columns.get_loc(i)
            numeric_cat_values.append(values)

        cat_features = list(range(0, len(working_df_names), 1))

        cat_out = list(set(cat_features) - set(numeric_cat_values))

    # Remove final label
    working_df_names.pop()

    # Remove final numeric index
    if i != number_cat_vars:
        cat_out.pop()

    if i == number_cat_vars:
        cat_out = cat_out

    # Return output
    return (cat_out, working_df_names)


def remove_synth_labels(dataframe):

    """A small function to remove any synth labels in output.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The dataframe to be worked on.


    Returns
    -------
    dataframe_out: pd.Dataframe
        The final processed output dataframe with no synthetic labels.

    """
    dataframe_columns = list(dataframe)
    removal_columns = []
    for column in dataframe_columns:
        if "Synth_Label_" in str(column):
            removal_columns.append(column)

    dataframe_out = dataframe.drop(removal_columns, axis=1)

    return (dataframe_out, removal_columns)
