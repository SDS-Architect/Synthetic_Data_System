### Standard Libraries
import pandas as pd
import pdb
import time
import sys

### General Modules
from SDS.src.back_end.General_Utility.Label_Convertor import (
    invertor_cat_col_convertor,
)

from SDS.src.back_end.General_Utility.General_Utilities import reverse_NaN

from SDS.src.front_interface.terminalsize import get_terminal_size

### Control Method Modules
from SDS.src.back_end.Looping_Control_Methods.Synthesis_Loop_vars import (
    prep_synth_loop,
    synthesis_loop_system,
)

### Date Reversal Modules
from SDS.src.simple_date_interface.\
    Reverse_Date_Processing.Reverse_Date_Transforms import (
    reverse_split_out_date,
    nan_test,
    return_date_nan,
)

from SDS.src.back_end.Original_Data_Out.Original_Data_Out import (
    original_data_process,
)


### General Information
"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""


def Synth_Control_Function(
    demographic_variables,
    machine_learning_variables,
    categorical_variables,
    size_of_synth_rows,
    name_of_output,
    combination_cols,
    GPU_IDs,
    file_path,
    # Optional vars in here
    cutting_vars=None,
    length_cuts=None,
    numeric_group_vars=None,
    group_size=None,
    remove_small_vals=None,
    number_gaussian=None,
    GMM_cutoff=None,
    synth_label_cols=None,
    synth_label_cols_stucture=None,
    date_columns=None,
):

    """Controls all the synthesis activity from user input.

    One function to rule them all, one function to find them, one
        function to bring them all and in the darkness bind them.

    Parameters
    ----------
    demographic_variables: list
        These are the variables in the real dataset (specified by user) that
        relate to the background information (demographic) of a sample. These
        are synthesised first as they can only take a small amount of discrete
        values i.e. people can't have an age of 1000 years.

    machine_learning_variables: list
        These are any variables that are not demographic and will be modelled
        using machine leanring methods.

    categorical_variables: list
        These are ANY variables that are discrete (only take certain values).
        Whether they are demographic or machine learning, then list them here.
        Inpits are specified by the user.

    size_of_synth_rows: integer
        This is how many rows will be generated in the final synthetic output.

    name_of_output: string
        The name of the file output. It is not nessecary to add in .csv to the
        end but the function handles this and corrects behaviour as nessecary.

    combination_cols: list
        The columns listed help form an index and remove small count variables.
        They will be merged together to create a special temporary 'Combi'
        column. The columns in the list should be chosen based on how
        important they are to the user's overall dataset. A balance needs to
        be struck as too few and it won't remove small records but too many
        and it will reduce the dataset to almost nothing.

    GPU_IDs: comma seperated integersGMM_Methods
        This specifies the name in integer form of the user's GPUs. It should
        take the form of 0, 1 if the user has 2 GPUs. The default behaviour is
        to default to GPU_IDs = 0.


    Optional Parameters
    -------------------
    cutting_vars: list, str, optional
        The columns listed will be shortened down to the lengths specified by
        length_cuts.

    length_cuts: list, int, optional
        The integers in this list are directly related to the columns listed
        in cutting_vars. This means that the user can have different length
        columns.

    numeric_group_vars: list, optional
        If integer based columns are listed here then they will processed by a
        Gaussian Mixture Model (GMM_Transform function) to create groups.

    group_size: integer, optional
        Controls how large the system groups the 'Combi' column into groups.
        The larger it is, the larger the groups are and the faster the system
        runs but at a cost of accuracy. The inverse of this is also true so a
        balance needs to be struck!

    GMM_cutoff: integer, optional
        The value is directly related to numeric_group_vars as the integer
        controls how many Gaussian Distributions are used to group values. The
        default number is 10 but note that while more distrbutions provides
        better grouping, it does in crease the computational need and can
        produce problems with interaction with the tree based synthesis.


    synth_label_cols: list, optional
        A synthetic column eases the transition from traditional sampling used
        for the demographic variables and the modern machine learning based
        methods that synthesise the rest of the data. A synthetic label should
        be created from the first column after the demographic variables. A
        user can have more than 1 synthetic label if they think it will help.


    synth_label_cols_stucture: list, int, optional
        This controls the cutoff for length of synthetic variables based on
        those specified in synth_label_cols.


    date_columns: list
        A list of columns specified by the user that are to be processed using
        the date handling methods


    Returns
    -------
    final_out: dataframe, .csv file
        This is the final output that is a large synthetic dataframe stored in
        both memory and also as a .csv file, name specified by user.
    """

    ### Obtain width of terminal for printing
    size_x, size_y = get_terminal_size()

    del size_y

    cur_message = " SDS Commencing Synthesis Process "
    print("\n" + "=" * int(size_x))
    print(" " * int(size_x / 2) + cur_message)
    print("=" * int(size_x) + "\n")

    # Welcoming Statement
    print("\n")
    print(
        "Welcome and thank you for choosing the NHS NSS Experimental"
        + " Synthetic Data Research System"
        + ", you wonderful human you!"
    )

    # Citation
    print(
        "\n"
        + "Please cite this system as: "
        + "\n"
        + "Gardner, E. (2019). Synthetic Data Experimental Research"
        + " System. Edinburgh: NHS NSS and Public Health Scotland."
        + "\n"
    )

    ### Print what vairables people have selected

    print("You selected these Categorical variables:")
    print("-----------------------------------------")
    for value in categorical_variables:
        print(value)
    print("\n")

    print("You selected these Machine Learning variables:")
    print("---------------------------------------------")
    for value in machine_learning_variables:
        print(value)
    print("\n")

    if date_columns is not []:
        print("You selected these Date variables: ")
        print("-----------------------------------")
        for i in date_columns:
            print(i)
        print("\n")

    if numeric_group_vars is not None:
        print("You selected these Numeric variables: ")
        print("--------------------------------------")
        for i in numeric_group_vars:
            print(i)
        print("\n")

    if combination_cols is not None:
        print("You selected these Combination variables: ")
        print("------------------------------------------")
        for i in combination_cols:
            print(i)
        print("\n")

    ### Get user input to continue
    answer = None

    while answer not in ("Yes", "No"):

        answer = input("Are you happy with all variables?" + "\n")

        if answer == "Yes":
            print("\n" + "Woo, synthesis adventure begins!!!")

        elif answer == "No":
            print("No synthesis :(")
            sys.exit()

        else:
            print("Please enter Yes or No." + "\n")

    print("=" * int(size_x))

    print("\n")
    cur_message = " Data Pre-Processing "
    print("\n" + "=" * int(size_x))
    print(" " * int(size_x / 2) + cur_message)
    print("=" * int(size_x) + "\n")

    # Various Diagnostic information tracking
    start = time.time()

    if group_size is None:
        group_size = 10

    if remove_small_vals is None:
        remove_small_vals = 10

    if number_gaussian is None:
        number_gaussian = 10

    if GMM_cutoff is None:
        GMM_cutoff = 20

    # Saving real file special vars
    reverse_categorical_variables = categorical_variables

    rev_demographic_variables = demographic_variables

    """ Data load and process """
    (
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
    ) = prep_synth_loop(
        categorical_variables=categorical_variables,
        combination_cols=combination_cols,
        demographic_variables=demographic_variables,
        cutting_vars=cutting_vars,
        length_cuts=length_cuts,
        GPU_IDs=GPU_IDs,
        remove_small_vals=remove_small_vals,
        numeric_group_vars=numeric_group_vars,
        number_gaussian=number_gaussian,
        GMM_cutoff=GMM_cutoff,
        synth_label_cols=synth_label_cols,
        synth_label_cols_stucture=synth_label_cols_stucture,
        date_columns=date_columns,
        file_path=file_path,
        machine_learning_variables=machine_learning_variables,
    )

    """ Reversal for real data out """

    if remove_small_vals != 0:
        # Create name of filtered data
        name_of_output = str(name_of_output).replace("[", "")
        name_of_output = str(name_of_output).replace("]", "")
        name_of_output = str(name_of_output).replace("'", "")

        name_of_output_original = "Real_Filt_" + str(name_of_output) + ".csv"

        print("\n")
        message = "Saving original filtered data file as: "
        print(message)
        print("-" * len(message))
        print(name_of_output_original)
        print("\n")

        original_data_process(
            file_path=file_path,
            name_of_output_original=name_of_output_original,
            categorical_variables=reverse_categorical_variables,
            combination_cols=combination_cols,
            demographic_variables=rev_demographic_variables,
            cutting_vars=cutting_vars,
            numeric_group_vars=numeric_group_vars,
            length_cuts=length_cuts,
            remove_small_vals=remove_small_vals,
            date_columns=date_columns,
        )

    ### Timing Data pre-processing
    stage_1_end = time.time()
    stage_1_time = str(round(stage_1_end - start, 3))

    print("\n")
    message = "Data pre-processing took: "
    print(message)
    print("-" * len(message))
    print(stage_1_time + " Seconds")
    print("\n")

    print("=" * int(size_x) + "\n")
    print("\n")

    cur_message = " SDS Main Synthesis "
    print("\n" + "=" * int(size_x))
    print(" " * int(size_x / 2) + cur_message)
    print("=" * int(size_x))

    """ Main Processing Script """
    (
        main_list,
        removal_columns,
        original_data_list,
        processed_date_columns_reverse,
    ) = synthesis_loop_system(
        real_data_frame=real_data_frame,
        groups_list=groups_list,
        group_size=group_size,
        GPU_IDs=GPU_IDs,
        demographic_variables=demographic_variables,
        size_of_synth_rows=size_of_synth_rows,
        machine_learning_variables=machine_learning_variables,
        categorical_variables=categorical_variables,
        threshold_hit=threshold_hit,
        information_dictionary=information_dictionary,
        numeric_group_vars=numeric_group_vars,
        mapping_dict=mapping_dict,
        processed_date_columns_reverse=processed_date_columns_reverse,
    )

    print("=" * int(size_x) + "\n")

    """ Final Output as CSV """
    print("\n")
    cur_message = " Synthesis Complete: Post Processing "
    print("\n" + "=" * int(size_x))
    print(" " * int(size_x / 2) + cur_message)
    print("=" * int(size_x) + "\n")

    """ Synthetic Data - Full Dataframe"""
    print("\n")
    message = "Now performing the following actions:"
    print(message)
    print("-" * len(message))
    print("\n")

    # Information
    print("Converting synthetic data into one dataframe")

    # Get Column names
    final_out = pd.DataFrame(columns=list(main_list[0]))

    for i in range(0, len(main_list), 1):
        final_out = final_out.append(main_list[i])

    """ Original Real Data - Full Dataframe"""
    # Get Column names
    original_data_out = pd.DataFrame(columns=list(original_data_list[0]))

    for i in range(0, len(original_data_list), 1):
        original_data_out = original_data_out.append(original_data_list[i])

    """ Label Conversion """
    # More information

    print("\n")
    print("Decoding Label Encoder Variables")
    print("\n")

    ### Get rid of synthetic labels
    categorical_variables = [
        x for x in categorical_variables if x not in removal_columns
    ]

    categorical_variables = [
        x for x in categorical_variables if x not in numeric_group_vars
    ]

    """ Synthetic Data"""
    final_out = invertor_cat_col_convertor(
        final_out, mapping_dict, categorical_variables
    )

    """ Original Real Data"""
    original_data_out = invertor_cat_col_convertor(
        original_data_out, mapping_dict, categorical_variables
    )

    """ Saving out as CSV files """
    ### Dealing with m_values_list
    final_out = reverse_NaN(final_out, m_values_list, removal_columns)

    original_data_out = reverse_NaN(
        original_data_out, m_values_list, removal_columns
    )

    """ Date Reversal """

    if date_columns is []:

        final_out = reverse_split_out_date(
            final_out, processed_date_columns_reverse
        )

        final_out = return_date_nan(final_out, date_columns)

    # Information
    name_of_output = str(name_of_output).replace("[", "")

    name_of_output = str(name_of_output).replace("]", "")

    name_of_output = str(name_of_output).replace("'", "")

    name_of_output_synth = str(name_of_output) + ".csv"

    print("\n")
    message = "Saving synthetic data file as: " + str(name_of_output_synth)
    print(message)
    print("-" * len(message))
    print("\n")

    # Saving synthetic file
    final_out.to_csv(name_of_output_synth, index=False)

    # Information
    end = time.time()
    time_final = round(((end - start) / 60), 3)
    total_time = str(time_final)

    print("\n")
    message = str("The total synthesis time was: " + total_time + " minutes")
    print(message)
    print("-" * len(message))
    print("\n")
    print("=" * int(size_x) + "\n")

    ### Final out message
    print("\n")
    cur_message = " END "
    print("\n" + "=" * int(size_x))
    print(" " * int(size_x / 2) + cur_message)
    print("=" * int(size_x) + "\n")

    print(
        "\n"
        + "Thank you for using the system, you wonderful human you!"
        + "\n"
    )

    # Citation
    print(
        "\n"
        + "Please cite this system as: "
        + "\n"
        + "Gardner, E. (2019). Synthetic Data Experimental Research"
        + " System. Edinburgh: NHS NSS and Public Health Scotland."
        + "\n"
    )

    print("=" * int(size_x) + "\n")
