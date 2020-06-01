### Standard Libraries
import numpy as np
import pandas as pd
import sys
import os

import sklearn as sk
import scipy
import time

### General Modules
from SDS.src.back_end.General_Utility.General_Utilities import (
    open_file,
    synth_label_create,
    string_cut,
    NaN_Handle_Cat,
    reverse_NaN,
)

from SDS.src.back_end.General_Utility.Label_Convertor import (
    cat_col_convertor,
    invertor_cat_col_convertor,
)

### Demographic Synthesis Modules
from SDS.src.back_end.Demographic_Synthesis.Demographic_Synthesis_Utilities \
    import (
    create_filtered_index,
)

### Date Transform Methods
from SDS.src.simple_date_interface.\
    Date_Pre_Processing.Date_Transform_Functions \
    import (
    date_only,
    date_strip,
)

from SDS.src.simple_date_interface.\
    Reverse_Date_Processing.Reverse_Date_Transforms \
    import (
    return_date_nan,
)

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This file contains functions that return a filtered dataset of the real data,
if the user has applied a threshold value - else it is passed. 

The reason for this file is that if a threshold value is applied then rows are 
potentially discarded so a comparison of real and synthetic data could become 
difficult. This means that if a threshold value is specified then there 
will be 3 files created by the SDS:
    
    - The original untouched data file
    - A filtered and altered version of the original data (this file creates)
    - A synthetic dataset
"""

def original_data_process(
    file_path,
    name_of_output_original,
    categorical_variables,
    combination_cols,
    demographic_variables,
    cutting_vars,
    numeric_group_vars,
    length_cuts,
    remove_small_vals,
    date_columns,
):

    """Function to prep the data for demographic/ML synthesis.

    Parameters
    ----------
    file_path: string
        A string that points the system to the selected file. Obtained using
        tkinter file dialogue methods earlier in system.

    name_of_output_original: string
        A string that represents the name of the file the user selected or the
        name of the file that the user has specified it to be called.

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

    Returns
    -------
    original_data_out: pd.DataFrame
        The processed dataframe of real data, saved to .csv file.

    """

    # Disable
    def blockPrint():
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enablePrint():
        sys.stdout = sys.__stdout__

    blockPrint()

    ### Get main file

    main_file = open_file(file_path)

    # Cut down variable length if needed
    if cutting_vars is not None:

        main_file = string_cut(main_file, length_cuts, cutting_vars)
    else:
        main_file = main_file

    """ Splice in Dates here"""
    if date_columns is not []:

        main_file = date_only(main_file, date_columns)

    else:
        main_file = main_file

    # Deal with missing values
    m_values_list, real_data_frame = NaN_Handle_Cat(main_file)

    # Auto convert all columns in categorical to categorical
    for column in categorical_variables:
        real_data_frame[column] = real_data_frame[column].astype(str)

    # Ensure correct columns
    real_data_frame, mapping_dict = cat_col_convertor(
        real_data_frame, categorical_variables
    )

    # Auto convert all columns in categorical to categorical
    for column in categorical_variables:
        real_data_frame[column] = real_data_frame[column].astype(str)

    # Create the index for main loop and remove counts
    real_data_frame, groups_list = create_filtered_index(
        real_data_frame,
        combination_cols,
        remove_small_vals,
        print_statement=False,
    )
    del groups_list

    ### Section 1 - Remove Label Encoder
    final_out = invertor_cat_col_convertor(
        real_data_frame, mapping_dict, categorical_variables
    )

    ### Section 2 - Replace '_Missing' values with ''
    original_data_out = reverse_NaN(
        final_out, m_values_list, removal_columns=[]
    )

    ### Section 3 - Return NaN dates
    original_data_out = return_date_nan(original_data_out, date_columns)

    ### Section 4 - Save to .csv
    original_data_out.to_csv(name_of_output_original, index=False)

    enablePrint()
