"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test learning module.
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.externals import joblib
from .stacking_estimator import StackingEstimator
from .zero_count import ZeroCount

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hpsklearn import HyperoptEstimator, random_forest_regression
from hyperopt import tpe
from lightgbm.sklearn import LightGBMError
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_is_fitted
from skopt import gp_minimize
from skopt import load
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from xgboost.core import XGBoostError



def build_default_model(seed=42):
    estimator = make_pipeline(
        StandardScaler(),
        StackingEstimator(estimator=KNeighborsClassifier(n_neighbors=40,
                                                         p=2,
                                                         weights="uniform")),
        StackingEstimator(estimator=LogisticRegression(C=5.0,
                                                       dual=False,
                                                       penalty="l1",
                                                       random_state=seed)),
        RobustScaler(),
        ZeroCount(),
        PCA(iterated_power=4, svd_solver="randomized", random_state=seed),
        LogisticRegression(C=20.0, dual=False, penalty="l1", random_state=seed)
    )

    return estimator


class SpanishPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, mode=None, copy_X_train=True, random_state=42):
        self.mode = mode
        self.copy_X_train = copy_X_train
        self.random_state=42

    def fit(self, X, y=None):
        """Fit estimator.

        A suitable set of hyperparameters is found via either Tree-structured
        Parzen Estimator (TPE) or Bayesian Optimization (BO).

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
        y : array-like, [n_samples, n_outputs]
            The target (continuous) values for regression.

        Returns
        -------
        self : object
        """

        # validate X, y
        X, y = check_X_y(X, y, multi_output=False, y_numeric=False)

    @classmethod
    def load(cls, filename : str = None):
        if filename is None:
            clf = build_default_model()
        else :
            with open(filename, "rb") as f:
                clf = joblib.load(f)

        return clf

    def save(self, filename : str):
        joblib.dump(self, filename)
