# Spanish Test

Python package with a pre-trained classifier for detecting if a given sample is
from Spanish origin.

The training features are population probabilities.

The training labels must contain a column labeled as "nationality". Those
samples of Spanish origin must be labelled as ``Spanish``.

The pre-trained model has been trained with samples from the 1k Genome project.

The current master points to the internal development version and includes a pre-trained model using a pipeline obtained with ``TPOT``.

In the following months we are going to include the ``TPOT`` library as a dependency. For now on, we only include the necessary files along their respective copyright notifications.

The published paper used the best model found by means of the ``TPE`` algorithm.
