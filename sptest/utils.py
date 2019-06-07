"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test utilities module.
"""

import os
from pathlib import Path


import click
from dotenv import find_dotenv, load_dotenv
    # click.echo(fname)
    # return


DOTENV_FILE_PATH = Path(find_dotenv())
load_dotenv(DOTENV_FILE_PATH)

def get_project_path():
    return DOTENV_FILE_PATH.parent

def get_data_path():
    return Path(os.environ.get("DATA_PATH"))

def get_default_model_path():
    return Path(__file__).parent.joinpath("data", "default_model.skl")
    # fname = resource_string(__file__,  "default_model.skl")
    # click.echo(fname)
    # return Path(fname)
