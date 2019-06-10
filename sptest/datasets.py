"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test I/O module.
"""

from pathlib import Path
import traceback

import pandas as pd

from .utils import get_data_path

DATA_PATH = get_data_path()


def load_training(data_path=None):

    if data_path is None:
        data_path = DATA_PATH

    labels_training_path = data_path.joinpath("ids_1000g.R")
    features_training_path = data_path.joinpath("plink.26.Q")

    features = pd.read_csv(features_training_path, header=None, sep=" ")
    labels = pd.read_csv(
        labels_training_path,
        header=None,
        sep=" ",
        index_col=0,
        names=["nationality", "continent"]
    )

    labels_bin = labels["nationality"].values.ravel() == "Spanish"

    return features, labels_bin


def load_test_file(fpath):

    fpath = Path(fpath)
    # load .Q file as pandas DataFrame
    features = pd.read_csv(fpath, header=None, sep=" ")

    if features.shape[0] == 0:
        raise Exception("No samples found in {}".format(fpath))

    features = features.iloc[0, :].copy()

    # Only the first sample must be evaluated
    return features


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
            features_single = load_test_file(fpath)
            groups.append(group)
            sample_names.append(sample_name)
            frames.append(features_single.iloc[0, :].copy())
        except IOError as io_except:
            print("No available data for {} which".format(fpath))
            print(traceback.format_exc(io_except))

    features = pd.concat(frames, axis=1, ignore_index=True).T
    features.index = sample_names

    return features
