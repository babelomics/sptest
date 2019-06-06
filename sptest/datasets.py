"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test I/O module.
"""

from pathlib import Path
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

def load_test_file(fpath):

    fpath = Path(fpath)
    # load .Q file as pandas DataFrame
    df = pd.read_csv(fpath, header=None, sep=" ")

    if df.shape[0] == 0:
        raise Exception("No samples found in {}".format(fpath))

    # Only the first sample must be evaluated
    return df.iloc[0, :].copy()

def load_test_folder(inputpath):

    inputpath = Path(inputpath)

    groups = []
    sample_names = []
    frames = []
    for fpath in inputpath.glob("**/*.Q"):
    #     print(fpath)
        sample_name = fpath.parent.name
        group = fpath.parent.parent.name

        try:
            df = load_test_file(fpath)
            groups.append(group)
            sample_names.append(sample_name)
            frames.append(df.iloc[0, :].copy())

    test_v2 = pd.concat(frames, axis=1, ignore_index=True).T
    test_v2.index = sample_names
    test_v2.head()
