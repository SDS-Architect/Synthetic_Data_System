from sklearn import preprocessing
import numpy as np

"""
Please cite this system as: 

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
The purpose of this file is to ensure that all the categories are label
encoded. This is so that strings, numbers etc. can all be used without issue.
"""


def cat_col_convertor(dataframe, categorical_variables):

    """Designed to auto-convert all needed columns using sklearn label 
       encoder method. 

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe to be worked on.

    categorical_variables: list
        These are the variables in the real dataset (specified by user) that
        relate are categorical - regardless of whether they are demographic or
        not. 


    Returns
    -------    
    main_file: pd.Dataframe
        The processed dataframe that is label encoded. 
    
    mapping_dict: dict
        Contains the label classes and information for EACH column for 
        reversal later. 

    """

    ### Make a copy
    main_file = dataframe.copy()

    ### Layout variables
    category_col = categorical_variables
    labelEncoder = preprocessing.LabelEncoder()

    ### Creating a map of all the numerical values of each categorical labels.
    mapping_dict = {}
    for col in category_col:
        main_file[col] = labelEncoder.fit_transform(main_file[col])

        le_name_mapping = dict(
            zip(
                labelEncoder.classes_,
                labelEncoder.transform(labelEncoder.classes_),
            )
        )

        mapping_dict[col] = le_name_mapping

    return (main_file, mapping_dict)


def invertor_cat_col_convertor(dataframe, mapping_dict, categorical_variables):

    """Designed to auto-convert all needed columns using sklearn label 
       encoder method. 

    Parameters
    ----------
    dataframe: pd.DataFrame
        Dataframe to be worked on.

    mapping_dict: dict
        Contains the label classes and information for EACH column. This means
        that to reverse, the correct dict must be selected.  

    categorical_variables: list
        These are the variables in the real dataset (specified by user) that
        relate are categorical - regardless of whether they are demographic or
        not. 


    Returns
    -------    
    main_file: pd.Dataframe
        The processed dataframe that is no longer encoded. 
    


    """
    ### Make a copy
    main_file = dataframe.copy()

    for column_name in categorical_variables:

        ### Get array
        arr_values = np.array(main_file[column_name].values)

        ### Dict lookup needs integers
        arr_values = arr_values.astype(int)

        ### Get dictionary mapping
        my_map = mapping_dict[column_name]

        ### Invert mapping
        inv_map = {v: k for k, v in my_map.items()}

        new = np.array([inv_map[x] for x in arr_values])

        ### Turn coolumn into string
        main_file[column_name] = main_file[column_name].astype(str)

        main_file[column_name] = new

    return main_file
