"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test program logic.
"""

import click

from .model import SpanishPredictor

@click.group()
@click.option("--mode", click.Choice(["builtin", "new"]))
@click.pass_context
def main(ctx, mode : str):
    ctx["mode"] = mode


@main.command()
@click.option("--folderpath", default="", help="Training path")
@click.option("--outputpath", help="Wher to store the trained model")
@click.pass_context
def train(ctx, folderpath : str, outputpath : str):
    if ctx["mode"] == "builtin":

    elif ctx["mode"] == "new":
        raise NotImplementedError()

@main.command()
@click.option("--filename", help="Test file (.Q)")
@clcik.option("--model_path", default="", help="")
@click.pass_context
def eval(ctx, filename : str, model_path : str):
    if model_path:
        model = SpanishPredictor.load(model_path)




def start():
    main(obj={})


if __name__ == "__main__":
    start()
