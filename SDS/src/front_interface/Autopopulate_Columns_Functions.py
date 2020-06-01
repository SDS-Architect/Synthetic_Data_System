# Standard library imports
from pathlib import Path
import os.path
from os.path import abspath, dirname, join, exists
import sys
import subprocess
import tempfile
import time
import textwrap
import csv

# External library imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
import yaml

### General Information
"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

def read_col_names(file_path):
    """ Function designed to simply read in only the columns of a csv
        file without loading the full file"""

    """
    Parameters
    ----------
    file_path: str
        Simple string for path to the needed file.


    Returns
    -------
    headers: list
        A list of all column headers in file.
    """

    with open(file_path, "r") as f:
        d_reader = csv.DictReader(f)

        # get fieldnames from DictReader object and store in list
        headers = d_reader.fieldnames

    return headers


"""
Please note that all code below is taken from:

https://github.com/gherka/summarize_two

All credit and many thanks to the author: German Priks
"""


def convert_dtypes(dtype):
    """
    Rename pandas default dtypes for readability
    """
    if is_numeric_dtype(dtype):
        return "Continuous"
    elif is_datetime64_dtype(dtype):
        return "Timeseries"
    return "Categorical"


def create_control_file(column_headers=None):
    """
    Open a temporary .yml file in OS-appropriate text editor
    and return a dictionary when user finishes editing it
    Accepted types are "dtypes", "xtab" and "ridge"
    Allowed kwargs for each type:
    type == dtypes:
        "dtypes" : converted dtypes
    type == xtab:
        "common_cols" = columns shared between two DFs
    """

    if column_headers is not None:

        comments = textwrap.dedent(
            """\
            #------------------------------------------------------------------
            # Please choose the fields for the x and y axis of the crosstab.
            # Valid column names are:

            #   %s

            #------------------------------------------------------------------
        """
            % (", ".join(map(str, column_headers)))
        )

        column_inputs = yaml.safe_dump(
            {
                "Column Parameters": {
                    "Demographic Columns": [],
                    "Machine Learning Columns": [],
                    "Combination Columns": [],
                }
            }
        )

        output_option_inputs = yaml.safe_dump(
            {
                "Synthetic File Parameters": {
                    "Output File Name": [],
                    "Number of Rows": [],
                }
            }
        )

        security_option_inputs = yaml.safe_dump(
            {
                "Security Parameters": {
                    "Remove Small Values Below": [],
                    "Columns to be Cut Down in Length": [],
                    "Length of Cuts": [],
                }
            }
        )

        optional_inputs = yaml.safe_dump(
            {
                "Optional Parameters": {
                    "Numeric Grouping Columns": [],
                    "Synthetic Label Columns": [],
                    "Synthetic Label Structure": [],
                    "Date Columns": [],
                }
            }
        )

        computer_option_inputs = yaml.safe_dump(
            {
                "Computer Parameters": {
                    "Group Size": [],
                    "Graphics Card(s) ID Number(s)": [],
                    "Number of Training Cycles for Random Forests": [],
                    "Depth of Random Forests Allowed": [],
                }
            }
        )

        temp_name = "xtab.yml"

    with tempfile.TemporaryDirectory() as td:
        f_name = join(td, temp_name)
        with open(f_name, "w") as f:
            f.write(comments)
            f.write("\n")
            f.write(column_inputs)
            f.write("\n")
            f.write(output_option_inputs)
            f.write("\n")
            f.write(security_option_inputs)
            f.write("\n")
            f.write(optional_inputs)
            f.write("\n")
            f.write(computer_option_inputs)

        if sys.platform == "win32":
            proc = subprocess.Popen(["notepad.exe", f_name])
        else:
            proc = subprocess.Popen(["gedit", "-s", f_name])

        modified = time.ctime(os.path.getmtime(f_name))
        created = time.ctime(os.path.getctime(f_name))

        while modified == created:
            time.sleep(0.5)
            modified = time.ctime(os.path.getmtime(f_name))

        proc.kill()

        with open(f_name, "r") as f:

            output = yaml.safe_load(f)
            return output
