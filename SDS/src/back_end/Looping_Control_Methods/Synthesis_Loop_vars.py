### Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy
import time
import secrets

### General Modules
from SDS.src.back_end.General_Utility.General_Utilities import (
    open_file,
    synth_label_create,
    string_cut,
    NaN_Handle_Cat,
)

from SDS.src.back_end.General_Utility.Label_Convertor import (
    cat_col_convertor,
    invertor_cat_col_convertor,
)

### Demographic Synthesis Modules
from SDS.src.back_end.Demographic_Synthesis.Demographic_Synthesis_Utilities import (
    create_filtered_index,
)

from SDS.src.back_end.Demographic_Synthesis.Demographic_Synthesis_Main import (
    Demographic_Synthesis,
)

### Machine Learning Synthesis
from SDS.src.back_end.Machine_Learning_Synthesis.ML_Synthesis_Main_debug import (
    Machine_Learning_Synthesis,
)

from SDS.src.back_end.Machine_Learning_Synthesis.ML_Synthesis_Utilities import (
    remove_synth_labels,
)

### Tree Based Synthesis Methods
from SDS.src.back_end.Tree_Methods.Tree_Functions_debug import tree_synth

### Gaussian Synthesis Methods
from SDS.src.back_end.GMM_Methods.GMM_Transform import (
    GMM_Transform,
    grouping_reversal,
)

### Date Transform Methods
from SDS.src.simple_date_interface.Date_Pre_Processing.Date_Transform_Functions import (
    date_only,
    date_strip,
    split_out_date,
)

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This file contains the functions related to how the SDS iterates over the
'Combi' column groups. It also contains the application of other functions
on each of the groups to synthesise the data.
"""

def prep_synth_loop(
    categorical_variables,
    combination_cols,
    demographic_variables,
    cutting_vars,
    length_cuts,
    remove_small_vals,
    numeric_group_vars,
    number_gaussian,
    GMM_cutoff,
    GPU_IDs,
    synth_label_cols,
    synth_label_cols_stucture,
    date_columns,
    file_path,
    machine_learning_variables,
):

    """Function to prep the data for demographic/ML synthesis.

    Parameters
    ----------
    categorical_variables: list
        A list of all categorical columns in data.

    combination_cols: list
        The columns listed help form an index and remove small count variables.
        They will be merged together to create a special temporary 'Combi'
        column. The columns in the list should be chosen based on how
        important they are to the user's overall dataset. A balance needs to
        be struck as too few and it won't remove small records but too many
        and it will reduce the dataset to almost nothing.

    demographic_variables: list
        A list of demographic variables.

    cutting_vars: :list, string
        The columns listed will be shortened down to the lengths specified by
        length_cuts.

    length_cuts: :list, integer
        The integers in this list are directly related to the columns listed
        in cutting_vars. This means that the user can have different length
        columns.

    remove_small_vals: integer
        The threshold number to remove records (including Combi) of count
        below it.

    numeric_group_vars: list, string
        If integer based columns are listed here then they will processed by a
        Gaussian Mixture Model (GMM_Transform function) to create groups.

    number_gaussian: integer
        Number of Gaussian Distributions allowed.

    GMM_cutoff: integer, optional
        The value is directly related to numeric_group_vars as the integer
        controls how many Gaussian Distributions are used to group values. The
        default number is 10 but note that while more distrbutions provides
        better grouping, it does in crease the computational need and can
        produce problems with interaction with the tree based synthesis.

    GPU_IDs: comma seperated integers
        This specifies the name in integer form of the user's GPUs. It should
        take the form of 0, 1 if the user has 2 GPUs. The default behaviour is
        to default to GPU_IDs = 0.

    synth_label_cols: list, optional
        A synthetic column eases the transition from traditional sampling used
        for the demographic variables and the modern machine learning based
        methods that synthesise the rest of the data. A synthetic label should
        be created from the first column after the demographic variables. A
        user can have more than 1 synthetic label if they think it will help.

    synth_label_cols_stucture: :list, int, optional
        This controls the cutoff for length of synthetic variables based on
        those specified in synth_label_cols.


    date_columns: list
        A list of columns specified by the user that are to be processed using
        the date handling methods


    Returns
    -------
    Note: all below is returned as a LIST.

    real_data_frame: pd.DataFrame
        The processed dataframe of real data, ready for synthesis.

    groups_list: list
        An index list of the Combi columns used to organise and loop through
        the data.

    information_dictionary:
        A dictionary of GMM distributions related to the data points with the
        related means/variances as the dictionary values.

    threshold_hit: list
        A list that details whether a column has less or more than a
        threshold, the cutoff argument. If the threshold is not hit then a 0
        will be recorded in the list. This means the tiny count data is
        discarded. If it is above the threshold then a 1 will be recorded in
        the threshold hit list and the data will be split later when reversing
        that column.

    m_values_list: list
        Part of a future update that will remove the _Missing markers.

    demographic_variables: list
        The processed demographic variables (inlcuding synthetic labels now)
        for further processing.

    """

    # Get the main file - tkinter interface
    main_file = open_file(file_path)

    ### Putting in print set up

    print("\n")
    print("Performing the following processes")
    print("----------------------------------")

    # Creation of synthetic variable
    if synth_label_cols is not None:
        main_file, synth_column_name_list = synth_label_create(
            main_file, synth_label_cols_stucture, synth_label_cols
        )

        # Add synthetic columns to demographic_vars
        demographic_variables = demographic_variables + synth_column_name_list
        categorical_variables = categorical_variables + synth_column_name_list

        # Synth col Update
        length_vars = str(len(synth_column_name_list))

        print("Creating: " + length_vars + " Synthetic Labels")

    else:
        main_file = main_file
        demographic_variables = demographic_variables
        categorical_variables = categorical_variables

    # Cut down variable length if needed
    if cutting_vars is not None:
        print("\n")
        print("Cutting variables to size needed")

        main_file = string_cut(main_file, length_cuts, cutting_vars)

    else:
        main_file = main_file

    """ Splice in Dates here"""
    # Creation of synthetic variable
    if date_columns is not None:

        main_file = date_only(main_file, date_columns)

        (
            main_file,
            processed_date_columns_reverse,
            working_date_columns,
        ) = split_out_date(main_file, date_columns)

        # Removing Redundant Columns
        demographic_variables = [
            x
            for x in demographic_variables
            if x not in list(processed_date_columns_reverse.keys())
        ]

        categorical_variables = [
            x
            for x in categorical_variables
            if x not in list(processed_date_columns_reverse.keys())
        ]

        machine_learning_variables = [
            x
            for x in machine_learning_variables
            if x not in list(processed_date_columns_reverse.keys())
        ]

        # Add synthetic columns to demographic_vars
        demographic_variables = demographic_variables + working_date_columns
        categorical_variables = categorical_variables + working_date_columns

    if date_columns is None:
        demographic_variables = demographic_variables
        categorical_variables = categorical_variables

    else:
        main_file = main_file
        demographic_variables = demographic_variables
        categorical_variables = categorical_variables
        processed_date_columns_reverse = None
        machine_learning_variables = machine_learning_variables

    # Deal with missing values
    print("\n")
    print("Dealing with missing values")

    m_values_list, real_data_frame = NaN_Handle_Cat(main_file)

    # Auto convert all columns in categorical to categorical
    for column in categorical_variables:
        real_data_frame[column] = real_data_frame[column].astype(str)

    print("\n")
    print("Applying Label Encoder")

    label_categorical_variables = [
        x for x in categorical_variables if x not in numeric_group_vars
    ]

    real_data_frame, mapping_dict = cat_col_convertor(
        real_data_frame, label_categorical_variables
    )

    # Auto convert all columns in categorical to categorical
    for column in categorical_variables:
        real_data_frame[column] = real_data_frame[column].astype(str)

    # Create the index for main loop and remove counts
    print("\n")
    print(
        "Creating loop index and removing values less than: "
        + str(remove_small_vals)
    )

    real_data_frame, groups_list = create_filtered_index(
        real_data_frame,
        combination_cols,
        remove_small_vals,
        print_statement=True,
    )

    # Automatic binning of variables
    if len(numeric_group_vars) != 0:
        real_data_frame, information_dictionary, threshold_hit = GMM_Transform(
            dataframe=real_data_frame,
            columns=numeric_group_vars,
            num_modes=number_gaussian,
            cutoff=GMM_cutoff,
        )

    if len(numeric_group_vars) == 0:
        real_data_frame = real_data_frame
        information_dictionary = []
        threshold_hit = []

    return (
        real_data_frame,
        groups_list,
        information_dictionary,
        threshold_hit,
        m_values_list,
        demographic_variables,
        categorical_variables,
        mapping_dict,
        m_values_list,
        processed_date_columns_reverse,
        machine_learning_variables,
    )


def synthesis_loop_system(
    real_data_frame,
    groups_list,
    group_size,
    demographic_variables,
    size_of_synth_rows,
    machine_learning_variables,
    categorical_variables,
    threshold_hit,
    GPU_IDs,
    information_dictionary,
    numeric_group_vars,
    mapping_dict,
    processed_date_columns_reverse,
):

    """ Ingests all information from the Synth_Control_Function() function.
            This is then parsed by the various subfunctions in this one to
            eventually build a list of dataframes of synthetic data.

    Parameters
    ----------
    real_data_frame: pd.DataFrame
        The dataframe of real data to be worked on.

    groups_list: list
        An index list of the Combi columns used to organise and loop through
        the data.

    group_size: int
        How large each group of the Combi variable will be.

    demographic_variables: list
        A list of demographic variables.

    size_of_synth_rows: integer
        How many rows in the final synthesis, as this increases then as does
        the richness of the synthesis.

    machine_learning_variables: list
        variables to be synthesised using tree method.

    categorical_variables: list
        A list of all categorical columns in data.

    threshold_hit: list
        A list that details whether a column has less or more than a
        threshold, the cutoff argument. If the threshold is not hit then a 0
        will be recorded in the list. This means the tiny count data is
        discarded. If it is above the threshold then a 1 will be recorded in
        the threshold hit list and print(df_col_1) print(df_col_2) the data
        will be split later when reversing that column.

    GPU_IDs: comma seperated integers
        This specifies the name in integer form of the user's GPUs. It should
        take the form of 0, 1 if the user has 2 GPUs. The default behaviour is
        to default to GPU_IDs = 0.

    information_dictionary: dict
        A dictionary of GMM distributions related to the data points with the
        related means/variances as the dictionary values.

    numeric_group_vars: list, string
        If integer based columns are listed here then they will processed by a
        Gaussian Mixture Model (GMM_Transform function) to create groups.

    mapping_dict: dict
        A dictionary of labels and their original values for EACH column.
        Used for checking the data types for error correction.

    processed_date_columns_reverse: list
        A list of columns to be handled by the reversing methods for date 
        columns. 

    Returns
    -------
    main_list: list, pd.DataFrames
        The synthetic data where each index value in main_list is a seperate
        group in the group index.

    removal_columns: list
        A list of synthetic columns to be removed.
    """

    # Get size of real data
    real_data_size = len(real_data_frame)

    # Create a variable to pipe into main_control_loop
    final_size = len(real_data_frame.Combi.value_counts())

    # This will be the main synthetic dataframe collection point
    main_list = list()

    # This is the real data list
    original_data_list = list()

    # Main loop for setting length of groups
    main_control_loop = [
        ((i), (i + group_size)) for i in range(0, final_size, group_size)
    ]

    for group_start, group_end in main_control_loop:

        # Update statement
        print("\n")
        print("**********************************************************")
        print(
            "Starting on groups: "
            + str(group_start)
            + " to "
            + str(group_end)
            + " of "
            + str(final_size)
        )
        print("**********************************************************")

        # Current working list
        current_working_list = groups_list[group_start:group_end]

        # Create a dataframe of current group data to be worked on
        working_real_data = real_data_frame[
            real_data_frame["Combi"].isin(current_working_list)
        ]

        # Create current percentage
        cur_percent = (len(working_real_data) / real_data_size) * 100

        """ Synthesise demographic variables"""
        # Create synthetic demo vars
        cat_demo_variables = list(
            set(demographic_variables).intersection(categorical_variables)
        )

        synthetic_demo = Demographic_Synthesis(
            real_dataframe=working_real_data,
            demo_vars=demographic_variables,
            cat_demo_vars=cat_demo_variables,
            original_df_size=real_data_size,
            synth_df_size=size_of_synth_rows,
            percent=cur_percent,
        )

        """ Synthesise ML variables"""
        ### Generate CSPRNG value for seed
        seed_num = secrets.SystemRandom()
        secure_seed_num = seed_num.randrange(0, 1000)

        machine_synthesis_df = Machine_Learning_Synthesis(
            feed_real_data_sub_df=working_real_data,
            prob_synth_df=synthetic_demo,
            demographic_vars=demographic_variables,
            GPU_IDs=GPU_IDs,
            ml_variables=machine_learning_variables,
            categorical_variables=categorical_variables,
            seed_training=secure_seed_num,
            mapping_dict=mapping_dict,
            numeric_group_vars=numeric_group_vars,
        )

        """Do inversion of GMM model here"""
        final_synth_df = grouping_reversal(
            dataframe=machine_synthesis_df,
            thres_hit_check=threshold_hit,
            information_dictionary=information_dictionary,
            grouped_cols=numeric_group_vars,
        )

        """ Removing Synthetic Labels """
        final_data_out, removal_columns = remove_synth_labels(final_synth_df)

        working_real_data, out = remove_synth_labels(working_real_data)

        del out

        """Append to mainlist of dataframes"""
        main_list.append(final_data_out)

        """ Adding in real data list here to invert later """
        original_data_list.append(working_real_data)

    return (
        main_list,
        removal_columns,
        original_data_list,
        processed_date_columns_reverse,
    )
