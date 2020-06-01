#!/usr/bin/env python
# coding: utf-8

# Standard Libraries
import numpy as np
import pandas as pd
import sklearn as sk
import scipy
import secrets

# ML Libraries
from catboost import Pool, CatBoostClassifier, CatBoostRegressor

"""
Please cite this system as:

Gardner, E. (2019). Synthetic Data Experimental Research System (Version 0.1a) 
    [software]. Avaliable from: https://github.com/SDS-Architect/SDS_Public
"""

"""
This file contains the Machine Learning core of the system.
"""


def tree_synth(
    Synthetc_Prob_Df,
    Real_Data,
    Real_Data_Cols,
    Real_Label_Col,
    Rand_Seed,
    Cat_Features,
    GPU_IDs,
):

    """ Main tree synthesis algorithm.

    Parameters
    ----------
    Synthetc_Prob_Df: pd.Dataframe
        The previously generated demographics from the probability function.

    Real_Data: pd.dataframe
        A subset of the original real data file to be worked on.

    Real_Data_Cols: list
        The data to be trained on.

    Real_Label_Col: string
        Target label.

    Rand_Seed: integer
        For replicability, sets the randomness of the train/test split.

    Cat_Features: list
        The columns in your data NOT continous numeric.


    Returns
    -------
        preds_class: np.array
            An array of predicted values that can be appended to a dataframe
            by later functions.
    """

    # Set random seed
    seed = Rand_Seed

    # Find data
    Data = Real_Data[Real_Data_Cols]
    Label = Real_Data[Real_Label_Col]

    # train/test split for evaluation
    from sklearn import preprocessing
    from sklearn.model_selection import train_test_split

    test_size = 0.2
    X_train, X_test, Y_train, Y_test = train_test_split(
        Data, Label, test_size=test_size, random_state=seed
    )

    # Setting up model
    train_data = X_train

    eval_data = X_test

    cat_features = Cat_Features

    train_label = Y_train

    eval_label = Y_test

    try:
        train_dataset = Pool(
            data=train_data, label=train_label, cat_features=cat_features
        )

        eval_dataset = Pool(
            data=eval_data, label=eval_label, cat_features=cat_features
        )

        # Initialize CatBoostClassifier
        model = CatBoostClassifier(
            iterations=400,
            task_type="GPU",
            devices=GPU_IDs,
            learning_rate=0.11,
            depth=11,
            verbose=10,
            loss_function="MultiClass",
        )
        # Fit model
        print("\n" + "Fitting Multi-Classification Tree Model")
        model.fit(train_dataset)

        # Get predicted RawFormulaVal
        preds_raw = model.predict(
            eval_dataset, prediction_type="RawFormulaVal"
        )

        preds_class = model.predict(Synthetc_Prob_Df)

    except:

        print(
            "\n"
            + "\n"
            + "Potential Error with Data: "
            + "\n"
            + "Pooling All Data"
            + "\n"
            + "\n"
        )

        train_dataset = Pool(data=Data, label=Label, cat_features=cat_features)

        # Initialize CatBoostClassifier
        model = CatBoostClassifier(
            iterations=400,
            task_type="GPU",
            devices=GPU_IDs,
            learning_rate=0.11,
            depth=11,
            verbose=100,
            loss_function="MultiClass",
        )
        # Fit model
        print(
            "\n"
            + "Fitting Multi-Classification Tree Model"
            + "- Pooled Data (NO METRICS)"
        )

        model.fit(train_dataset)

        # Get predicted RawFormulaVal
        preds_raw = model.predict(
            train_dataset, prediction_type="RawFormulaVal"
        )

        preds_class = model.predict(Synthetc_Prob_Df)

    return preds_class
