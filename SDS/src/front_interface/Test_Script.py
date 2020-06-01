import psutil
import platform
from datetime import datetime

import os
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

from SDS.src.front_interface.terminalsize import get_terminal_size

from SDS.src.front_interface.test_gpu import ConvertSMVer2Cores, main
import numpy

### General Information
"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

def open_file():

    """ One-click opening and loading of file.

    Parameters
    ----------
    None.


    Returns
    -------
    main_file: pd.DataFrame
        A pandas dataframe of the selected file.
    """
    # Load a Tk Inter instance
    root = Tk()
    root.withdraw()

    # Select the file
    file_path = askopenfilename()

    # Destroy the root/open window
    root.destroy()

    statinfo = os.stat(file_path)

    file_size = statinfo.st_size

    return (file_size, str(file_path))


def get_size(bytes, suffix="B"):

    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Adapted from:
    https://stackoverflow.com/questions/5194057/better-way-
    to-convert-file-sizes-in-python

    Parameters
    ----------
    bytes: integer
        The amount of bytes from a file. 

    
    suffix: string
        Controls the input/output format. 


    Returns
    -------
    factor: string 
        A string detailing information about the file.

    """

    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def testing():

    """
    A function that assists users by analysing computer specs and returning
    information by printing to the terminal.

    Parameters
    ----------
    None.


    Returns
    -------
    file_path: String
        The path to the file the user has picked to be synthesised.   

    """

    ### Obtain width of terminal for printing
    size_x, size_y = get_terminal_size()

    del size_y

    cur_message = " Synthetic Data System: Current Machine Test "

    print("\n" + "=" * int(size_x))
    print(" " * int(size_x / 2) + cur_message)
    print("=" * int(size_x) + "\n")

    """ System Information """
    cur_message = " System Information "
    print(
        "*" * int(size_x / 2)
        + cur_message
        + "*" * int(size_x - (int(size_x / 2) + len(cur_message)))
    )
    print("\n")

    uname = platform.uname()
    print(f"System: {uname.system}")
    print(f"Node Name: {uname.node}")
    print(f"Release: {uname.release}")
    print(f"Version: {uname.version}")
    print(f"Machine: {uname.machine}")
    print(f"Processor: {uname.processor}")
    print("\n")

    print(
        "\n"
        + "Please cite this system as: "
        + "\n"
        + "Gardner, E. (2019). Synthetic Data Experimental Research"
        + " System. Edinburgh: NHS NSS and Public Health Scotland."
        + "\n"
    )

    print("*" * size_x)
    print("\n")
    print("\n")

    # File Diagnostics
    cur_message = " File Diagnostics "
    print(
        "-" * int(size_x / 2)
        + cur_message
        + "-" * int(size_x - (int(size_x / 2) + len(cur_message)))
    )

    raw_byte, file_path = open_file()
    size = get_size(raw_byte)
    print("File Size:", size)

    print("\n")
    print("-" * size_x)
    print("\n")
    print("\n")

    # CPU information
    cur_message = " CPU Info "
    print(
        "-" * int(size_x / 2)
        + cur_message
        + "-" * int(size_x - (int(size_x / 2) + len(cur_message)))
    )

    # number of cores
    print("Physical cores:", psutil.cpu_count(logical=False))
    print("Total cores:", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    print(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    print("\n")
    print("-" * size_x)
    print("\n")
    print("\n")

    # Memory Information

    cur_message = " Memory Information "
    print(
        "-" * int(size_x / 2)
        + cur_message
        + "-" * int(size_x - (int(size_x / 2) + len(cur_message)))
    )

    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total: {get_size(svmem.total)}")
    print(f"Available: {get_size(svmem.available)}")
    print(f"Used: {get_size(svmem.used)}")
    print(f"Percentage: {svmem.percent}%")

    print("\n")
    print("-" * size_x)
    print("\n")
    print("\n")

    # GPU Information
    cur_message = " GPU Information "
    print(
        "-" * int(size_x / 2)
        + cur_message
        + "-" * int(size_x - (int(size_x / 2) + len(cur_message)))
    )

    gpu_total = main()

    print("-" * size_x)
    print("\n")
    print("\n")

    ############ Recommended System Settings
    cur_message = " Recommended Settings "
    print(
        "-" * int(size_x / 2)
        + cur_message
        + "-" * int(size_x - (int(size_x / 2) + len(cur_message)))
    )

    ### RAM Check
    print("RAM Check")
    print("---------")
    svmem = psutil.virtual_memory()
    RAM_Total = svmem.total
    RAM_Avaliable = int(svmem.available)

    ### Total RAM Check
    if int(raw_byte) / RAM_Total * 100 < 50:
        print("Total RAM Check: PASS")
        print("\n")

    if (
        int(raw_byte) / RAM_Total * 100 > 50
        and int(raw_byte) / RAM_Total * 100 < 70
    ):
        print("Total RAM Check: PASS")
        print("NOTE: Advise lowering parameters due to memory")
        print("\n")

    if int(raw_byte) / RAM_Total * 100 >= 70:
        print("Total RAM Check: FAIL")
        print(
            "The amount RAM in your computer is likely insufficient to "
            + "run this synthesis"
        )
        print("\n")

    ### Avaliable RAM Checks - closing programmes
    if int(raw_byte) / RAM_Avaliable * 100 <= 50:
        print("Avaliable RAM Check: PASS")
        print("\n")

    if (
        75 >= int(raw_byte) / RAM_Avaliable * 100
        and int(raw_byte) / RAM_Avaliable * 100 > 50
    ):
        print("Avaliable RAM Check: PASS")
        print(
            "NOTE: Advise lowering parameters or closing open"
            + "programmes to compensate"
        )
        print("\n")

    if int(raw_byte) / RAM_Avaliable * 100 > 75:
        print("Avaliable RAM Check: FAIL")
        print(
            "Your avaliable RAM is insufficient, close some programmes or"
            + " you may not have enough memory overall"
        )
        print("\n")

    ### GPU Check
    print("GPU Card(s) Check")
    print("-----------------")

    ### Calculate Avaliable Memory

    gpu_total = sum(gpu_total)

    if int(raw_byte) / gpu_total * 100 <= 50:
        print("GPU Memory Check: PASS")
        print("\n")

    if (
        int(raw_byte) / gpu_total * 100 > 50
        and int(raw_byte) / gpu_total * 100 < 70
    ):
        print("GPU Memory Check: PASS")
        print("NOTE: Advise lowering parameters due to memory")
        print("\n")

    if int(raw_byte) / gpu_total * 100 > 75:
        print("GPU RAM Check: FAIL")
        print(
            "NOTE: Try CPU only configuration (GPU_IDs = No_Gpus) in "
            + " Config file"
        )
        print("\n")

    print("=" * size_x)
    print("\n")
    return file_path
