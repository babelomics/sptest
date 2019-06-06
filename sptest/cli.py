"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test program logic.
"""

import click

from .model import SpanishPredictor
from .utils import get_default_model_path

@click.group()
@click.option("--ncpu", type=int, help="Number of cores to use.")
@click.pass_context
def main(ctx, ncpu):
    ctx["ncpu"] = ncpu


@main.command()
@click.option('--tune', is_flag=True, help="Will print verbose messages.")
@click.option("--inputpath", default="", help="Training path")
@click.argument("--outputpath", help="Where to store the trained model")
@click.pass_context
def train(ctx, tune, inputpath, outputpath):
    model = SpanishPredictor(
        tune=tune,
        n_jobs=ctx["ncpu"]
        )


@main.command()
@click.argument("-inputpath", help="Test file (.Q)")
@clcik.option("--model_path", help="Model for prediction")
@click.pass_context
def eval(ctx, inputpath, model_path=None):

    if model_path is None:
        model_path = get_default_model_path()

    model = SpanishPredictor.load(model_path)
    proba = model.predict_from_file(inputpath)
    click.echo(proba)


def start():
    main(obj={})


if __name__ == "__main__":
    start()
