"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test program logic.
"""

import click

from .model import SpanishPredictor
from .utils import get_default_model_path


@click.group()
@click.option("--ncpu", default=-1, type=int, help="Number of cores to use.")
@click.pass_context
def main(ctx, ncpu):
    ctx.obj = {
        "ncpu": ncpu
    }


@main.command()
@click.option('--tune', is_flag=True, help="Will print verbose messages.")
@click.option("--inputpath", default="", help="Training path")
@click.argument("--outputpath")
@click.pass_context
def train(ctx, tune, inputpath, outputpath):
    """ Where to store the trained model """
    model = SpanishPredictor(
        tune=tune,
        n_jobs=ctx.obj["ncpu"]
        )


@main.command()
@click.argument("inputpath", required=1)
@click.option(
    "--model_path",
    default=get_default_model_path(),
    help="Model for prediction")
@click.pass_context
def eval(ctx, inputpath, model_path):
    """"Test file (.Q)"
    """

    model = SpanishPredictor.build_estimator(
        n_jobs=ctx.obj["ncpu"],
        filename=model_path
    )
    proba = model.predict_proba_from_file(inputpath)
    if proba.shape[0] == 1:
        click.echo(proba[0])
    else:
        click.echo(proba)


if __name__ == "__main__":
    main()
