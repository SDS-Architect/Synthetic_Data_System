### Standard Libraries
import pandas as pd
import pdb
import time

### Front End Modules
from SDS.src.front_interface.Test_Script import testing

import SDS.src.front_interface.Autopopulate_Columns_Functions as ap

from SDS.src.back_end.Control_Function import Synth_Control_Function

### General Information
"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

def synthesis_activation():

    file_path = testing()

    column_names = ap.read_col_names(file_path)

    control_variables = ap.create_control_file(column_headers=column_names)

    ### Main Column Inputs
    categorical_variables = column_names

    demographic_variables = control_variables["Column Parameters"][
        "Demographic Columns"
    ]

    ML_vars = control_variables["Column Parameters"][
        "Machine Learning Columns"
    ]

    combination_cols = control_variables["Column Parameters"][
        "Combination Columns"
    ]

    ### Output File Controls
    name_of_output = control_variables["Synthetic File Parameters"][
        "Output File Name"
    ]

    size_of_synth_rows = control_variables["Synthetic File Parameters"][
        "Number of Rows"
    ]

    ### Security Parameters
    remove_small_vals = control_variables["Security Parameters"][
        "Remove Small Values Below"
    ]

    remove_small_vals = int(remove_small_vals[0])

    cutting_vars = control_variables["Security Parameters"][
        "Columns to be Cut Down in Length"
    ]

    length_cuts = control_variables["Security Parameters"]["Length of Cuts"]

    ### Optional Synthesis Controls
    numeric_group_vars = control_variables["Optional Parameters"][
        "Numeric Grouping Columns"
    ]

    synth_label_cols = control_variables["Optional Parameters"][
        "Synthetic Label Columns"
    ]

    synth_label_cols_stucture = control_variables["Optional Parameters"][
        "Synthetic Label Structure"
    ]

    date_columns = control_variables["Optional Parameters"]["Date Columns"]

    ### Computer Control Parameters

    group_size = control_variables["Computer Parameters"]["Group Size"]

    GPU_IDs = control_variables["Computer Parameters"][
        "Graphics Card(s) ID Number(s)"
    ]

    tree_iterations = control_variables["Computer Parameters"][
        "Number of Training Cycles for Random Forests"
    ]

    tree_depth = control_variables["Computer Parameters"][
        "Depth of Random Forests Allowed"
    ]

    Synth_Control_Function(
        demographic_variables=demographic_variables,
        categorical_variables=categorical_variables,
        combination_cols=combination_cols,
        cutting_vars=cutting_vars,
        numeric_group_vars=numeric_group_vars,
        length_cuts=length_cuts,
        GPU_IDs=GPU_IDs,
        machine_learning_variables=ML_vars,
        size_of_synth_rows=size_of_synth_rows,
        name_of_output=name_of_output,
        group_size=group_size,
        synth_label_cols=synth_label_cols,
        synth_label_cols_stucture=synth_label_cols_stucture,
        remove_small_vals=remove_small_vals,
        date_columns=date_columns,
        file_path=file_path,
    )
