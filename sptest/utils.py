"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test utilities module.
"""

import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv


DOTENV_FILE_PATH = Path(find_dotenv())
load_dotenv(DOTENV_FILE_PATH)

def get_project_path():
    return DOTENV_FILE_PATH.parent

def get_data_path():
    return Path(os.environ.get("DATA_PATH"))

def get_default_model_path():
    return Path(os.environ.get("DEAFULT_MODEL_PATH"))

class NotRequiredIf(click.Option):
    """[summary]
    See https://stackoverflow.com/questions/44247099/click-command-line-interfaces-make-options-required-if-other-optional-option-is

    Arguments:
        click {[type]} -- [description]

    Raises:
        click.UsageError: [description]

    Returns:
        [type] -- [description]
    """
    def __init__(self, *args, **kwargs):
        self.not_required_if = kwargs.pop('not_required_if')
        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs['help'] = (kwargs.get('help', '') +
            ' NOTE: This argument is mutually exclusive with %s' %
            self.not_required_if
        ).strip()
        super(NotRequiredIf, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        we_are_present = self.name in opts
        other_present = self.not_required_if in opts

        if other_present:
            if we_are_present:
                raise click.UsageError(
                    "Illegal usage: `%s` is mutually exclusive with `%s`" % (
                        self.name, self.not_required_if))
            else:
                self.prompt = None

        return super(NotRequiredIf, self).handle_parse_result(
            ctx, opts, args)
