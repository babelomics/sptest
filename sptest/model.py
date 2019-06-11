#coding: utf-8
# pylint: disable=invalid-name, too-many-arguments

"""
author: Carlos Loucera
email: carlos.loucera@juntadeandalucia.es

Spanish Test learning module. Spanish here refers to the ML class.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_X_y

from .datasets import load_test_file, load_test_folder
from .stacking_estimator import StackingEstimator
from .zero_count import ZeroCount

# Index of the positive class in an probability array,
POS_CLASS_INDEX = 1


class SpanishPredictor(BaseEstimator, ClassifierMixin):
    """This class implments a Machine Learning (ML) method to classify a given
    sample as Spanish or  no-Spanish.

    Parameters
    ----------
    tune : bool, optional (default: False)
        Tune an ensamble of trees with Bayessian Optimization, by default False
        uses a predefined model (found via TPOT and AUCPR).
    copy_X_train : bool, optional (default: True)
        If True, a persistent copy of the training data is stored in the
        object. Otherwise, just a reference to the training data is stored,
        which might cause predictions to change if the data is modified
        externally.
    seed : int, optional (default: 42)
        Seed to initalize the random number generator, by default 42.
    n_jobs : int, optional (default: -1)
        Number of jobs to run in parallel, by default -1 uses all available
        cores.
    n_iter : int, optional _(default: 1000)
        Number of BO iterations during BO tunning, by default 10**3.

    """

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
        -------
        X : array-like featuresse matrix, shape=(n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.
        y : array-like, [n_samples, n_outputs]
            The target (bianry) values for classification.

        Returns
        -------
        self : returns an instance of self.
        """

        # validate X, y
        X, y = check_X_y(X, y, multi_output=False, y_numeric=False)
        self.fit_(X, y)

    def fit_(self, X, y=None):
        """If self.tune fits an BO-tunned ensemble of trees, otherwise fits a
        predefined stacked estimator.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values

        Returns
        -------
        self : returns an instance of self.

        """

        if self.tune:
            tuner = ParzenCV(
                n_jobs=self.n_jobs,
                random_state=self.seed,
                n_iter=self.n_iter)

            if hasattr(X, "values"):
                X = X.values
            if hasattr(y, "values"):
                y = y.values.ravel()

            tuner.fit(X, y)

            self.estimator = tuner.best_estimator_
        else:
            self.estimator.fit(X, y)

    def predict_proba(self, X):
        """Predict the probability estimate of a given sample or set of samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Population probabilities as features.

        Returns
        -------
        T : array-like, shape = [n_samples, ]
            Returns the probability of being Spanish of the samples.

        """

        check_is_fitted(self.estimator, "fit")

        return self.estimator.predict_proba(X)

    def predict_proba_from_file(self, inputpath):
        """Predict the probability estimate of a given sample or set of samples.

        Parameters
        ----------
        inputpath : string
            A .Q file or a folder where several .Q files are stored.

        Returns
        -------
        T : array-like, shape = [n_samples, ]
            Returns the probability of being Spanish of the samples.

        """

        X_test = self.load_features(inputpath)
        if X_test.ndim != 2:
            # Only one sample (i.e. single .Q file)
            X_test = X_test.values.reshape(1, -1)
        else:
            X_test = X_test.values

        X_test = pd.DataFrame(X_test)

        return self.predict_proba(X_test)[:, POS_CLASS_INDEX]

    @staticmethod
    def load_features(inputpath):
        """Load a single .Q file or a set of grouped .Q files.

        Parameters
        ----------
        inputpath : string
            Either a .Q  path of a given sample or the folder where several
            groups of .Q files are stored.

        Returns
        -------
        DataFrame, shape=[n_samples, n_features]
            Poulation probability estimates.

        Raises
        ------
        IOError
            I/O error.
        """

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
        """Another form to build and estimator.

        Parameters
        ----------
        n_jobs : int, optional (default: -1)
        Number of jobs to run in parallel, by default -1 uses all available
        cores.

        filename : string, optional (default: None)
            Loads a model stored in ``filename``, by default None loads a
            predefined model.

        Returns
        -------
        An instance of ``SpanishPredictor``.
        """

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
        """Saves a fitted model into a binary a file.

        Parameters
        ----------
        filename : str
            File path where to store the model in binary form.
        """

        joblib.dump(self.estimator, filename)

    @staticmethod
    def build_default_model(n_jobs=-1, seed=42):
        """Builds an stacked estimator (a binary classifier):

        Parameters
        ----------
        n_jobs : int, optional (default: -1)
            Number of jobs to run in parallel, by default -1 uses all available
            cores.

        seed : int, optional (default: 42)
            Seed to initalize the random number generator, by default 42.

        Returns
        -------
        p : Pipeline

        """

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
                max_iter=10**4,
                solver='saga')),
            RobustScaler(),
            ZeroCount(),
            PCA(iterated_power=4, svd_solver="randomized", random_state=seed),
            LogisticRegression(
                C=20.0,
                dual=False,
                penalty="l1",
                random_state=seed,
                n_jobs=n_jobs,
                max_iter=10**4,
                solver='saga'
            )
        )

        return estimator


class ParzenCV(object):
    """Find an optimized XGBoost via Tree of Parzen estimators."""

    def __init__(self, search_spaces=None,
                 scoring="average_precision", cv=3, n_jobs=-1,
                 n_iter=10**3, verbose=0, refit=True,
                 random_state=42):

        if search_spaces is None:
            self.search_spaces = ParzenCV.get_default_xgb_space()
        else:
            self.search_spaces = search_spaces
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.verbose = verbose
        self.refit = refit
        self.random_state = random_state
        self.best_estimator_ = None
        self.history = None

    # Estimate XGB params
    def fit(self, X, y):
        """Fit/optiize via Tree-structured Parzen Estimator."""

        space = self.get_default_xgb_space()
        self.history = self.optimize(space, X, y)

        best_params = {k: self.history[k] for k in space.keys()}
        problem_params = self.get_problem_params()

        params = {**best_params, **problem_params}

        estimator_ = xgb.XGBClassifier(**params)

        if self.refit:
            estimator_.fit(X, y)

        self.best_estimator_ = estimator_

    @staticmethod
    def get_default_xgb_space():
        """Get XGBoost default hyperparamter space."""

        _space = dict(
            max_depth=scope.int(hp.uniform(
                "max_depth", 1, 11)),
            learning_rate=hp.loguniform(
                "learning_rate", np.log(0.0001), np.log(0.5)) - 0.0001,
            n_estimators=scope.int(hp.quniform(
                "n_estimators", 100, 6000, 200)),
            gamma=hp.loguniform(
                "gamma", np.log(0.0001), np.log(5)) - 0.0001,
            min_child_weight=scope.int(hp.loguniform(
                "min_child_weight", np.log(1), np.log(100))),
            subsample=hp.uniform(
                "subsample", 0.5, 1),
            colsample_bytree=hp.uniform(
                "colsample_bytree", 0.5, 1),
            colsample_bylevel=hp.uniform(
                "colsample_bylevel", 0.5, 1),
            reg_alpha=hp.loguniform(
                "reg_alpha", np.log(0.0001), np.log(1)) - 0.0001,
            reg_lambda=hp.loguniform(
                "reg_lambda", np.log(1), np.log(4)),
            scale_pos_weight=1
        )

        return _space

    def get_problem_params(self):
        """Get xgb paramters for unbalanced classifcation."""
        params = dict(
            max_delta_step=0,
            verbose=1,
            verbose_eval=1,
            silent=1,
            nthread=self.n_jobs,
            tree_method='approx',
            eval_metric='aucpr',
            objective='binary:logistic'
        )

        return params

    def optimize(self, params_space, X, y):
        """Optimization loop."""

        def objective(params):
            """Objective function to minimize."""

            problem_params = self.get_problem_params()

            params = {**params, **problem_params}

            _estimator = xgb.XGBClassifier(**params)

            cv_scores = cross_val_score(
                _estimator,
                X,
                y,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs
            )

            score = 1 - np.mean(cv_scores)

            return {'loss': 1.0 - score, 'status': STATUS_OK}

        trials = Trials()
        history = fmin(
            fn=objective,
            space=params_space,
            algo=tpe.suggest,
            max_evals=self.n_iter,
            trials=trials,
            verbose=0
        )

        return history
