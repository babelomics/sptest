"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test utilities module.
"""

from pathlib import Path


DATA_PATH = Path(__file__).parent.joinpath("data")


def get_data_path():
    """Get path for package data folder.

    Returns
    -------
    p: Path

    """

    return DATA_PATH

def get_default_model_path():
    """Get path for a pre-trained model (binary form).

    Returns
    -------
    p: Path

    """

    model_fname = "default_model.skl"

    return DATA_PATH.joinpath(model_fname)
