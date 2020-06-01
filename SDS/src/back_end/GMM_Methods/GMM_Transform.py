# coding: utf-8
import numpy as np
from sklearn.mixture import GaussianMixture

"""
Please cite this system as: 

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
The purpose of this file is to allow columns with large amounts of numeric
information to be made into groups. The Gaussian Mixture Model (GMM) allows
this as it will cluster values into num_modes number of distributions. These
groups can be reduced later.
"""


def GMM_Model(dataframe, column, num_modes):
    """Function to fit GMM and returns array of grouped values

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe to fit model on column.

    column: string 
        Name of column of dataframe to fit model to.

    num_modes: integer
        Number of Gaussian Distributions allowed.


    Returns
    -------
    groups: np.array
        Integers that represent which distribution mean and variance that a 
        data point belongs to.

    means: np.array 
        The mean for each of the Gaussian Distributions allowed, number is 
        equal to num_modes (0: num_modes-1).

    variances: np.array
        Same as means above but with variance values instead.
    """

    # Subset out column
    modelling_data = np.array(dataframe[column])

    # Transform into linear array
    modelling_data_array = modelling_data.reshape(-1, 1)

    # Define the model
    model = GaussianMixture(num_modes)

    # Fit the model
    model.fit(modelling_data_array)

    # Get the array group values
    groups = model.predict(modelling_data_array)

    # Get means
    means = model.means_

    # Get vairances
    variances = model.covariances_

    return (groups, means, variances)


def GMM_Transform(dataframe, columns=None, num_modes=None, cutoff=None):

    """ Fits a GMM model, groups items in a column and returns a transformed 
            dataframe plus information needed to reverse the transform later.

    Parameters
    ----------
    dataframe: pd.Dataframe
        The main dataframe to work on.


    Optional Parameters
    -------------------
    columns: list
        The numeric columns that need grouped (default is None).

    num_modes: integer
        Number of Gaussian Distributions that can be used in model fitting 
        (default = 10).

    cutoff: integer 
        The threshold at which if the number of _Missing strings is above then 
        it will split the data to keep these. Otherwise if the number of 
        _Missing is below the threshold then it will just drop the data 
        (default = 20).


    Returns
    -------
    data_removal_thres_hit: integer 
        If column contains missing values (_Missing) les in count than the 
        threshold, then these will be dropped and a 0 will be recorded. If 
        _Missing data is higher than the threshold then a 1 will be recorded 
        to keep the data and split it to preserve the _Missing strings.

    numeric_col_transform_dict: dict
        Dictionary where the key is the number of the GMM distribution that a 
        data point belongs to. The value is a tuple of the means and variances
        that can be used to reverse the grouping of the data point later.

    threshold_hit: list
        A list that details whether a column has less or more than a 
        threshold, the cutoff argument. If the threshold is not hit then a 0 
        will be recorded in the list. This means the tiny count data is 
        discarded. If it is above the threshold then a 1 will be recorded in 
        the threshold hit list and the data will be split later when reversing
        that column. 
    """

    # If no groups are needed return 3 empty values
    if columns is None:
        print("\n")
        print("No grouping needed")
        return (dataframe, "No grouping needed")

    # Run the function if group cols needed
    if columns is not None:

        # Set default values
        if cutoff is None:
            cutoff = 20

        # Set default values
        if num_modes is None:
            num_modes = 10

        # Dictionary of mean/var to reverse later
        numeric_col_transform_dict = dict()

        # Threhold values check
        threshold_hit = []

        for column in columns:

            # Create a target _missing variable
            target = str(column) + "_Missing"

            # Split out any missing
            non_numeric_real_data = dataframe.loc[
                dataframe[column].isin([target]),
            ]

            # Drop out any missing values
            real_data = dataframe[
                ~dataframe.isin(non_numeric_real_data)
            ].dropna()

            # Ensure correct format
            real_data[column] = real_data[column].astype(float)

            # Removal of low count values
            if non_numeric_real_data.shape[0] <= cutoff:

                # Fit and return model
                groups, means, variances = GMM_Model(
                    real_data, column, num_modes=num_modes
                )

                # Transform the old column
                real_data[column] = groups

                # Esnure return as string
                real_data[column] = real_data[column].astype(str)

                # Add to dictionary to reverse later
                numeric_col_transform_dict[column] = [means, variances]

                # Testing to see if threshold hit
                threshold_hit.append(0)

            # Deals with high count of missing values
            if non_numeric_real_data.shape[0] > cutoff:

                # Fit the model
                groups, means, variances = GMM_Model(
                    real_data, column, num_modes=num_modes
                )

                # Get the data
                real_data[column] = groups

                # Combine the two df's
                real_data = real_data.append(non_numeric_real_data)

                # Esnure return as string
                real_data[column] = real_data[column].astype(str)

                numeric_col_transform_dict[column] = [means, variances]

                # Testing to see if threshold hit
                threshold_hit.append(1)

            # Returning outputs
            return (real_data, numeric_col_transform_dict, threshold_hit)


def grouping_reversal(
    dataframe, information_dictionary, thres_hit_check, grouped_cols
):

    """This function reverses the transformations made by GMM_Transform and
            returns transformed dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        The main datset being worked on.

    information_dictionary: dictionary 
        Made by GMM_Transform which  contains  key(Column):
        values([means for column group, variances for columns group]).

    thres_hit_check: list
        Binary representation to check if further work and data splitting 
        needed or to just use standard method. This triggers whether or not 
        function is needed to run.

    grouped_cols: list
        A list of columns that have been grouped.


    Returns
    -------
    dataframe: pd.DataFrame
        A dataframe where GMM encoded columns have been resampled to values in
        original dataset. 
    """

    # If no groups are needed return 2 empty values
    if thres_hit_check == "No grouping needed":
        print("\n")
        print("No grouping reversal needed")
        return dataframe

    else:

        if 1 in thres_hit_check:
            for column in grouped_cols:
                # Create a target _missing variable
                target = str(column) + "_Missing"

                # Split out any missing
                non_numeric_real_data = dataframe.loc[
                    dataframe[column].isin([target]),
                ]

                # Drop out any missing values
                numeric_real_data = dataframe[
                    ~dataframe.isin(non_numeric_real_data)
                ].dropna()

                for column in grouped_cols:

                    # Initialise reconstructed column
                    recon_column = []

                    # Get the column as array
                    column_array = numeric_real_data[column]

                    # Get mean and var for a column
                    mean = information_dictionary[column][0]
                    variance = information_dictionary[column][1]

                    for value in column_array:

                        # Set as int
                        value = int(value)

                        # Get local mean
                        mean_value = mean[value]

                        # Get local variance
                        var_value = variance[value]

                        data = int(
                            abs(np.random.normal(mean_value, var_value))
                        )

                        recon_column.append(data)

                    numeric_real_data[column] = recon_column

                split_out_combined = numeric_real_data.append(
                    non_numeric_real_data
                )

                dataframe[column] = split_out_combined

        if 1 not in thres_hit_check:

            for column in grouped_cols:

                # Initialise reconstructed column
                recon_column = []

                # Get the column as array
                column_array = dataframe[column]

                # Get mean and var for a column
                mean = information_dictionary[column][0]
                variance = information_dictionary[column][1]

                for value in column_array:

                    # Set as int
                    value = int(value)

                    # Get local mean
                    mean_value = mean[value]

                    # Get local variance
                    var_value = variance[value]

                    data = int(abs(np.random.normal(mean_value, var_value)))

                    recon_column.append(data)

                dataframe[column] = recon_column

    return dataframe
