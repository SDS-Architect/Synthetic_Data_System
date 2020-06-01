# coding: utf-8

# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import secrets
import scipy

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a)
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

def coin_flip_noise_value(Lap_scale, Lap_loc, prob_vector):

    """ Generates the Laplacian noise to add to probability sample.

    Parameters
    ----------
    Lap_scale: float
        Location value for Laplace distribution. Similar (but not same) in
        idea to mean in Gaussian.

    Lap_loc:  float
        Scale value for Laplace distribution. Similar (but not same) in idea
        to Std. Dev. in Gaussian.

    prob_vector: np.array, float
        How big the synthetic data file is. This is used to proprtionally
        synthesise the synthetic file so if category_A is 10% in the real
        file size (original_real_size) then it will be 10% of the final
        synthetic file (final_samp_size).


    Returns
    -------
    final_noise_vector: np.array, float
        A dataframe that is proportional to the final synthetic dataset size.
    """

    # Get length of prob vector
    length_vector = len(prob_vector)

    # Add here
    if length_vector % 2 > 0:
        real_length = length_vector - 1
    else:
        real_length = length_vector

    # Ensure correct vector length
    real_length = int(real_length / 2)

    # Creating the noise
    noise_value = np.random.laplace(Lap_scale, Lap_loc, real_length)

    # Type correction and rounding
    noise_value = noise_value.astype("float32")
    noise_value = np.round(noise_value, 9)

    # Create the inverse vector for 0 sum
    noise_value_negative = np.negative(noise_value)

    # Adding noise and shuffle
    noise_vector = np.concatenate((noise_value, noise_value_negative))
    np.random.shuffle(noise_vector)

    # Add here in random 0 if needed
    if length_vector % 2 > 0:
        zero_balance = np.array([0])
        noise_vector = np.append(noise_vector, zero_balance)

    # Add the noise to original probability vector
    final_noise_vector = np.add(prob_vector, noise_vector)

    # Returns actual noised probs
    return final_noise_vector


def get_probability(s):

    """ Quick function to grab probabilities of a column.

    Parameters
    ----------
    s: pd.Series
        A column of a dataframe


    Returns
    -------
    temp: pd.Series
        A series with probabilities
    """

    temp = s.value_counts()
    return temp / temp.sum()


def probability_eval_one(initial_probs):

    """ Ensures that probabilities sum to 1. If the probabilities don't sum to
    1 then the difference will subtracted from the largest probability in the
    list - as this should have a small impact.

    Parameters
    ----------
    initial_probs: list
        A list of probabilities with noise added.


    Returns
    -------
    initial_probs: np.array
        A numpy array with probabilities that sum to 1
    """

    initial_diff_zero = 1 - sum(initial_probs)

    if initial_diff_zero != 0:

        max_val = max(initial_probs)

        initial_probs = initial_probs.tolist()

        max_val_pos = initial_probs.index(max_val)

        initial_probs[max_val_pos] = max_val + initial_diff_zero

        initial_probs = np.array(initial_probs)

        return initial_probs

    return initial_probs


def secure_coin_toss(percent):

    """ Securely generates a coinflip probability using CSPRNG.

    Parameters
    ----------
    percent: float
        Represents the size of the current dataset in relation to the original
        real dataset. It controls what set of low/med/high paranters are used.


    Returns
    -------
    noise_choice: float
        Represents the probability (chosen at random) of noise being added
        (the probability of heads or 'H').

    """

    ### Set up a range of possible values to sample from
    low_coin_flip = [[0.55, 0.6], [0.5, 0.60], [0.46, 0.5], [0.35, 0.55]]

    med_coin_flip = [[0.55, 0.6], [0.6, 0.65], [0.55, 0.67], [0.55, 0.7]]

    high_coin_flip = [[0.75, 0.8], [0.8, 0.87], [0.8, 0.9], [0.85, 0.9]]

    ### CSPRNG - to select a random range
    num = secrets.SystemRandom()
    secure_num = num.randrange(0, 4)

    ### Obtain a range
    if percent < 50:
        security_range = low_coin_flip[secure_num]

    if percent >= 50 and percent <= 75:
        security_range = med_coin_flip[secure_num]

    if percent > 75:
        security_range = high_coin_flip[secure_num]

    ### Sample from that range
    secure_num = round(num.uniform(security_range[0], security_range[1]), 5)

    # Coin flip in here with CSPRNG range
    noise_choice = np.random.choice(
        a=["H", "T"], p=[secure_num, (1 - secure_num)], size=1
    )

    return noise_choice


def diff_priv_alg(percent, initial_probs):

    """ Securely generates a coinflip probability using CSPRNG.

    Parameters
    ----------
    percent: float
        Represents the probability (chosen at random) of noise being added
        (the probability of heads or 'H').

    initial_probs: np.array
        A numpy array with probabilities that represent the conditional probs
        of the demographic variables and may or may not have noise added to
        them.


    Returns
    -------
    noised_probs: np.array
        A numpy array that may or may not have noise added to it.

    """

    ### Flip coin for noise
    add_noise_choice = secure_coin_toss(percent)

    if len(initial_probs) == 1 or add_noise_choice == "T":
        return initial_probs

    ### Adding Laplacian Noise
    if add_noise_choice == "H":

        ### Low values
        lap_low_values = [
            [0.0000005, 0.00000005],
            [0.0000003, 0.0000007],
            [0.000000002, 0.000000005],
            [0.0000002, 0.00000003],
        ]

        ### High values
        lap_high_values = [
            [0.000006, 0.0000066],
            [0.0000041, 0.000005],
            [0.000003, 0.000007],
            [0.000001, 0.000003],
        ]

        ### Addition of noise
        num = secrets.SystemRandom()
        secure_num = num.randrange(0, 4)

        ### Obtain a range
        if percent <= 50:
            security_range = lap_low_values[secure_num]

        if percent > 50:
            security_range = lap_high_values[secure_num]

        ### Obtain a range
        lap_loc_value = num.uniform(security_range[0], security_range[1])

        ### Adding the noise
        noised_probs = coin_flip_noise_value(0, lap_loc_value, initial_probs)

        ### Ensuring value returns as summed to 1
        noised_probs = probability_eval_one(noised_probs)

        ### Return the values
        return noised_probs


def prob_dataframe_gen_with_dp(
    real_data, original_real_size, final_samp_size, col_tuples, percent
):

    """ Creates the synthetic variables for the demographic columns.

    Parameters
    ----------
    real_data: pd.dataframe
         A subset of the original real data file to be worked on.

    original_real_size: integer
        The FULL size of your original dataset. You can do something like:
        len(your_data_set_here).

    final_samp_size: integer
        How big the synthetic data file is. This is used to proprtionally
        synthesise the synthetic file so if category_A is 10% in the real file
        size (original_real_size) then it will be 10% of the final synthetic
        file (final_samp_size).

    col_tuples: list, tuple
        Demograohic column pairs created by col_tuple_pair_gen function.

    percent: float
        Derived from main function, detects how far along index loop the
        function is.


    Returns
    -------
    synth_df: pd.DataFrame
        A dataframe that is proportional to the final synthetic dataset size.
    """

    # Stop Pandas annoying me about my bad coding.
    pd.options.mode.chained_assignment = None

    """ Initialise main vars"""
    # Get the proportional size
    current_sample = len(real_data) / original_real_size

    # Initialise synthetic dataframe
    synth_df = pd.DataFrame()

    # Get initial Synth DF column name
    initial_column_name = list(real_data)[0]

    ### Patching in Issue
    initial_probs = get_probability(real_data[initial_column_name])

    ### ADD IN HERE CSPRNG
    initial_probs = diff_priv_alg(percent, initial_probs)

    # Create initial Synth DF column
    initial_vals = np.random.choice(
        a=real_data[initial_column_name].unique(),
        size=int(final_samp_size * current_sample),
        p=initial_probs,
    )

    synth_df[initial_column_name] = initial_vals

    """Main Loop - iterates over and creates conditional prob"""
    for df_col_1, df_col_2 in col_tuples:

        real_data_probs = real_data.groupby(df_col_1)[df_col_2].apply(
            get_probability
        )

        synth_df[df_col_2] = None

        for i, group in synth_df.groupby(df_col_1):

            # Calculate out raw probabilities
            choices = list(pd.DataFrame(real_data_probs[i]).T)

            probs = pd.DataFrame(real_data_probs[i]).T.values[0]

            # Adding in effects of noise here
            probs = diff_priv_alg(percent, probs)

            # Modify the
            synth_series = np.random.choice(
                a=choices, p=probs, size=len(group)
            )

            group[df_col_2] = synth_series

            synth_df.iloc[group.index] = group

    print("\n")
    print("Demographic Synthesis for this group")
    print("------------------------------------")
    print("Completed")

    return synth_df
