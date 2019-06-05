"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test I/O module.
"""

import pandas as pd

from .utils import get_data_path

DATA_PATH = get_data_path()


def load_training():
    labels_training_path = DATA_PATH.joinpath("ids_1000g.R")
    features_training_path = DATA_PATH.joinpath("plink.26.Q")

    X_train = pd.read_csv(features_training_path, header=None, sep=" ")
    y_train = pd.read_csv(
        labels_training_path,
        header=None,
        sep=" ",
        index_col=0,
        names=["nationality", "continent"]
    )

    y_train_bin = y_train["nationality"].values.ravel() == "Spanish"

    return X_train, y_train, y_train_bin
