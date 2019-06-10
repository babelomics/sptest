"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test program logic.
"""

import click

from .model import SpanishPredictor
from .utils import get_default_model_path
from .datasets import load_training


@click.group()
@click.option("--ncpu", default=-1, type=int, help="Number of cores to use.")
@click.pass_context
def main(ctx, ncpu):
    """CLI entry point."""
    ctx.obj = {
        "ncpu": ncpu
    }


@main.command()
@click.option('--tune', is_flag=True, help="If tune use BO with XGBoost.")
@click.option("--inputpath", default=None, help="Training path")
@click.argument("outputpath", required=1)
@click.pass_context
def train(ctx, tune, inputpath, outputpath):
    """1k-Genome training and store model in outputpath."""

    features, labels = load_training(inputpath)

    model = SpanishPredictor(
        tune=tune,
        n_jobs=ctx.obj["ncpu"]
        )
    model.fit(features, labels)
    model.save(outputpath)


@main.command()
@click.argument("inputpath", required=1)
@click.option(
    "--model_path",
    default=get_default_model_path(),
    help="Model for prediction")
@click.pass_context
def predict(ctx, inputpath, model_path):
    """"Predict a given .Q file. The ouput is the probability to be Spanish. Use
    a pretrained model if no binary file is provided.
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
    # pylint: disable=no-value-for-parameter
    main()
