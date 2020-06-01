# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy
import secrets
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Remove non-needed warnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


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


def samping_col(range_num, secure_num):
    """ Builds a column using CSPRNG sampling.

    Parameters
    ----------
    range_num: int
        A number the size of the dataset

    secure_num: int
        CSPRNG generated random number for a set of values below

    Returns
    -------
    col_list_vals: pd.Series
        A pd.Series of values for use in the GMM column
    """

    col_list_vals = []
    for i in range(0, range_num):
        value = secrets.randbelow(secure_num)
        col_list_vals.append(value)

    col_list_vals = pd.Series(col_list_vals)
    return col_list_vals


### Function to process MIMIC
def process_mimc(mimic_dataset):
    """ One-click opening and loading of file.

    Parameters
    ----------
    mimic_dataset: pd.Dataframe


    Returns
    -------
    main_file: pd.DataFrame
        A pandas dataframe of processed mimic_dataset ready to
        be synthesised in examples.
    """

    # Column cutting
    mimic_dataset = mimic_dataset[
        [
            "ADMISSION_LOCATION",
            "DISCHARGE_LOCATION",
            "INSURANCE",
            "LANGUAGE",
            "RELIGION",
            "MARITAL_STATUS",
            "ETHNICITY",
            "ADMISSION_TYPE",
            "ADMITTIME",
            "DISCHTIME",
            "DEATHTIME",
            "HOSPITAL_EXPIRE_FLAG",
            "EDREGTIME",
            "EDOUTTIME",
            "HAS_CHARTEVENTS_DATA",
            "DIAGNOSIS",
        ]
    ]

    # GMM example sampling

    size_data = len(mimic_dataset)

    seq = secrets.token_hex(20)

    random_try = secrets.choice(seq)

    if random_try is int():
        secure_num = num.randrange(100, 150)

        # Making a new column
        mimic_dataset["DEMO_GMM"] = samping_col(size_data, secure_num)

    if random_try is str():
        secure_num = num.randrange(95, 151)

        # Making a new column
        mimic_dataset["DEMO_GMM"] = samping_col(size_data, secure_num)

    # Save the file out
    mimic_dataset.to_csv(
        "Ready_to_synth_MIMIC.csv", index=None, index_label=None
    )


mimic_dataset = open_file()

process_mimc(mimic_dataset)
