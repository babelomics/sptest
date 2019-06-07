"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test learning module.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y
from skopt import BayesSearchCV, gp_minimize, load
from skopt.space import Categorical, Integer, Real
from skopt.utils import use_named_args
from xgboost.core import XGBoostError

from .datasets import load_test_file, load_test_folder
from .stacking_estimator import StackingEstimator
from .zero_count import ZeroCount

POS_CLASS_INDEX = 1


class SpanishPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, tune=False, copy_X_train=True, seed=42, n_jobs=-1,
                 n_iter=10**3):

        self.tune = tune
        self.copy_X_train = copy_X_train
        self.seed = seed
        if tune:
            self.estimator = None
        else:
            self.estimator = SpanishPredictor.build_default_model(n_jobs, seed)
        self.n_jobs = n_jobs
        self.n_iter = n_iter

    def fit(self, X, y=None):
        """Fit estimator, it expects a binary response.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
        y : array-like, [n_samples, n_outputs]
            The target (bianry) values for classification.

        Returns
        -------
        self : object
        """

        # validate X, y
        X, y = check_X_y(X, y, multi_output=False, y_numeric=False)
        self.fit_(X, y)

    def fit_(self, X, y=None):

        if self.tune:
            tuner = self.bayes_tuner(self.n_jobs, self.seed, self.n_iter)
            result = tuner.fit(X, y)
            self.estimator = result.best_estimator_
        else:
            self.estimator.fit(X, y)

    def predict_proba(self, X):

        return self.estimator.predict_proba(X)

    def predict_proba_from_file(self, inputpath):

        X_test = self.load_features(inputpath)
        if X_test.ndim != 2:
            # Only one sample (i.e. single .Q file)
            X_test = X_test.values.reshape(1, -1)

        return self.predict_proba(X_test)[:, POS_CLASS_INDEX]

    @staticmethod
    def load_features(inputpath):

        inputpath = Path(inputpath)

        if inputpath.is_dir():
            features = load_test_folder(inputpath)
        elif inputpath.is_file():
            features = load_test_file(inputpath)
        else:
            raise IOError("Invalid file or folder {}".format(inputpath.name))

        return features

    @classmethod
    def build_estimator(cls, n_jobs=-1, filename=None):

        if filename is None:
            estimator = SpanishPredictor.build_default_model(n_jobs)
        else:
            with open(filename, "rb") as f:
                estimator = joblib.load(f)

        model = SpanishPredictor()
        model.estimator = estimator
        if hasattr(estimator, "n_jobs"):
            model.n_jobs = estimator.n_jobs
        else:
            model.n_jobs = n_jobs

        return model

    def save(self, filename: str):

        joblib.dump(self.estimator, filename)

    @staticmethod
    def build_default_model(n_jobs=-1, seed=42):

        estimator = make_pipeline(
            StandardScaler(),
            StackingEstimator(estimator=KNeighborsClassifier(
                n_neighbors=40,
                p=2,
                weights="uniform")),
            StackingEstimator(estimator=LogisticRegression(
                C=5.0,
                dual=False,
                penalty="l1",
                random_state=seed,
                n_jobs=n_jobs,
                max_iter=10**4)),
            RobustScaler(),
            ZeroCount(),
            PCA(iterated_power=4, svd_solver="randomized", random_state=seed),
            LogisticRegression(
                C=20.0, dual=False, penalty="l1", random_state=seed,
                n_jobs=n_jobs,
                max_iter=10**4
            )
        )

        return estimator

    @staticmethod
    def bayes_tuner(n_jobs, seed, n_iter):

        tuner = BayesSearchCV(
            estimator=xgb.XGBClassifier(
                n_jobs=n_jobs,
                objective='binary:logistic',
                eval_metric='aucpr',
                silent=1,
                tree_method='approx'
            ),
            search_spaces={
                'learning_rate': (0.01, 1.0, 'log-uniform'),
                'min_child_weight': (0, 10),
                'max_depth': (0, 50),
                'max_delta_step': (0, 20),
                'subsample': (0.01, 1.0, 'uniform'),
                'colsample_bytree': (0.01, 1.0, 'uniform'),
                'colsample_bylevel': (0.01, 1.0, 'uniform'),
                'reg_lambda': (1e-9, 1000, 'log-uniform'),
                'reg_alpha': (1e-9, 1.0, 'log-uniform'),
                'gamma': (1e-9, 0.5, 'log-uniform'),
                'n_estimators': (50, 100),
                'scale_pos_weight': (1e-6, 500, 'log-uniform')
            },
            scoring='average_precision',
            cv=StratifiedKFold(
                n_splits=10,
                shuffle=True,
                random_state=42
            ),
            n_jobs=n_jobs,
            n_iter=n_iter,
            verbose=0,
            refit=True,
            random_state=seed
        )

        return tuner
