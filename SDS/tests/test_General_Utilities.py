""" Test files for General_Utilities functions """

### Load in test module
import General_Utilities as tm

### Load in needed libraries
import unittest
import pandas as pd
import numpy as np
import sys

### Load in useful toy dataset
from sklearn.datasets import load_iris


class Test_General_Utilities(unittest.TestCase):
    """ 
    ---------------------------------------------------------------------------
    TESTING FOR open_file() 
    ---------------------------------------------------------------------------
    Testing the open file protocol for different file types
    """

    def test_open_csv(self):
        """ Tests if file successfully opens csv and returns a pandas
            dataframe. 
        """

        ### Call the function
        result = tm.open_file("test_datasets/test_1_full.csv")

        ### Test it
        self.assertIsInstance(result, pd.DataFrame)

    def test_open_non_csv(self):
        """ Tests if file unsuccessfully opens npy and returns a 
        pandas dataframe. 
        """

        ### Test and call in one
        with self.assertRaises(TypeError):
            tm.open_file("sdgsdgdgdsg")

    """ 
    ---------------------------------------------------------------------------
    TESTING FOR string_cut() 
    ---------------------------------------------------------------------------
    Testing if string cutting function cuts to correct lengths and can
    deal with non-string information being tried and raising the correct
    error messages.
    """

    def test_string_cut_return_correct(self):
        """ 
        Tests if file successfully opens csv and returns a pandas
        dataframe. 
        """

        ### Call the function
        test_df = tm.open_file("test_datasets/test_1_full.csv")

        test_string_cut_value = 3

        test_col_list = ["Column_1", "Column_3"]

        result = tm.string_cut(test_df, test_string_cut_value, test_col_list)

        ### Test it
        self.assertIsInstance(result, pd.DataFrame)

    def test_string_cut_empty_arguments(self):
        """
        Tests to ensure throws exception when no argument passed.
        """

        with self.assertRaises(TypeError):
            tm.string_cut()

    def test_string_cut_wong_data_types(self):
        """
        Tests to ensure throws exception when wrong argument types are used.
        """

        test_df = tm.open_file("test_datasets/test_1_full.csv")

        test_string_cut_value = "f"

        test_col_list = ["Column_1", "Column_3"]

        with self.assertRaises(TypeError):
            tm.string_cut(test_df, test_string_cut_value, test_col_list)

    def test_string_cut_wong_column_names(self):
        """
        Tests to ensure throws exception when wrong argument types are used.
        """

        test_df = tm.open_file("test_datasets/test_1_full.csv")

        test_string_cut_value = 3

        test_col_list = ["Column_", "Column_"]

        with self.assertRaises(TypeError):
            tm.string_cut(test_df, test_string_cut_value, test_col_list)

    def test_string_cut_return_correct_length_single(self):
        """ 
        Tests if file successfully opens csv and returns a pandas
        dataframe. 
        """

        ### Seeting test parameters
        test_df = tm.open_file("test_datasets/test_1_full.csv")

        test_string_cut_value = 3

        test_col_list = ["Column_1", "Column_3"]

        for col in test_col_list:
            all(test_df[col].str.len() < test_string_cut_value)

        test_df = test_df[test_col_list]

        processed_df = tm.string_cut_multi(
            test_df, test_string_cut_value, test_col_list
        )

        measurer = np.vectorize(len)

        result = measurer(processed_df.values.astype(str)).max(axis=0)

        result = all(elem == test_string_cut_value for elem in result)

        self.assertTrue(result)

    def test_string_cut_return_correct_length_multiple(self):
        """ 
        Tests if file successfully opens csv and returns a pandas
        dataframe. 
        """

        ### Seeting test parameters
        test_df = tm.open_file("test_datasets/test_1_full.csv")

        test_string_cut_value = [1, 3]

        test_col_list = ["Column_1", "Column_3"]

        test_df = test_df[test_col_list]

        processed_df = tm.string_cut_multi(
            test_df, test_string_cut_value, test_col_list
        )

        testing_list = []

        for column in test_col_list:
            value_test = processed_df[column].map(len).max()
            testing_list.append(value_test)

        self.assertEqual(testing_list, test_string_cut_value)

    def test_string_cut_return_wrong_string_values(self):
        """ 
        Tests if file successfully opens csv and returns a pandas
        dataframe. 
        """

        ### Seeting test parameters
        test_df = tm.open_file("test_datasets/test_1_full.csv")

        test_string_cut_value = [1, "3"]

        test_col_list = ["Column_1", "Column_3"]

        test_df = test_df[test_col_list]

        with self.assertRaises(TypeError):
            tm.string_cut_multi(test_df, test_string_cut_value, test_col_list)

    # '''
    # ---------------------------------------------------------------------------
    # TESTING FOR synth_label_create()
    # ---------------------------------------------------------------------------
    # Testing to see if the function can:
    #  - Create correct synthetic labels
    #  - Handle missing values in a column
    #  - Deal with problems with input
    # '''

    # def test_synth_label_create_return_correct_single(self):

    #     tm.synth_label_create

    # def test_synth_label_create_return_correct_multi(self):

    #     pass


if __name__ == "__main__":
    unittest.main()
