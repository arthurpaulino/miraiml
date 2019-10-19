"""
:mod:`miraiml.core` contains internal classes responsible for the optimization
process.
"""

import random as rnd
import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from miraiml.util import load, dump, sample_random_len


class BaseModel:
    """
    Represents an element from the search space, defined by an instance of
    :class:`miraiml.SearchSpace` and a set of features.

    Read more in the :ref:`User Guide <base_model>`.

    :type model_class: type
    :param model_class: A statistical model class that must implement the methods
        ``fit`` and ``predict`` for regression or ``predict_proba`` classification
        problems.

    :type parameters: dict
    :param parameters: The parameters that will be used to instantiate objects of
        ``model_class``.

    :type features: list
    :param features: The list of features that will be used to train the statistical
        model.
    """
    def __init__(self, model_class, parameters, features):
        self.model_class = model_class
        self.parameters = parameters
        self.features = features

    def predict(self, X_train, y_train, X_test, config):
        """
        Performs the predictions for the training and testing datasets and also
        computes the score of the model.

        :type X_train: pandas.DataFrame
        :param X_train: The dataframe that contains the training inputs for the
            model.

        :type y_train: pandas.Series or numpy.ndarray
        :param y_train: The training targets for the model.

        :type X_test: pandas.DataFrame
        :param X_test: The dataframe that contains the testing inputs for the model.

        :type config: miraiml.Config
        :param config: The configuration of the engine.

        :rtype: tuple
        :returns: ``(train_predictions, test_predictions, score)``

            * ``train_predictions``: The predictions for the training dataset
            * ``test_predictions``: The predictions for the testing dataset
            * ``score``: The score of the model on the training dataset

        :raises: ``RuntimeError`` when fitting or predicting doesn't work.
        """
        X_train = X_train[self.features]
        train_predictions = np.zeros(X_train.shape[0])

        test_predictions = None
        if X_test is not None:
            X_test = X_test[self.features]
            test_predictions = np.zeros(X_test.shape[0])

        if config.problem_type == 'classification' and config.stratified:
            fold = StratifiedKFold(n_splits=config.n_folds, shuffle=False)
        elif config.problem_type == 'regression' or not config.stratified:
            fold = KFold(n_splits=config.n_folds, shuffle=False)
        for big_part, small_part in fold.split(X_train, y_train):

            X_train_big, y_train_big = X_train.iloc[big_part], y_train.iloc[big_part]
            X_train_small = X_train.iloc[small_part]

            model = self.model_class(**self.parameters)
            class_name = self.model_class.__name__

            try:
                model.fit(X_train_big, y_train_big)
            except Exception:
                raise RuntimeError('Error when fitting with model class {}'.format(class_name))
            try:
                if config.problem_type == 'classification':
                    train_predictions[small_part] = model.predict_proba(X_train_small)[:, 1]
                    if X_test is not None:
                        test_predictions += model.predict_proba(X_test)[:, 1]
                elif config.problem_type == 'regression':
                    train_predictions[small_part] = model.predict(X_train_small)
                    if X_test is not None:
                        test_predictions += model.predict(X_test)
            except Exception:
                raise RuntimeError('Error when predicting with model class {}'.format(
                    class_name
                ))

        if X_test is not None:
            test_predictions /= config.n_folds
        return (train_predictions, test_predictions,
                config.score_function(y_train, train_predictions))


def dump_base_model(base_model, path):
    """
    Saves the characteristics of a base model as a checkpoint.

    :type base_model: miraiml.core.BaseModel
    :param base_model: The base model to be saved

    :type path: str
    :param path: The path to save the base model

    :rtype: tuple
    :returns: ``(train_predictions, test_predictions, score)``
    """
    attributes = dict(parameters=base_model.parameters, features=base_model.features)
    dump(attributes, path)


def load_base_model(model_class, path):
    """
    Loads the characteristics of a base model from disk and returns its respective
    instance of :class:`miraiml.core.BaseModel`.

    :type model_class: type
    :param model_class: The model class related to the base model

    :type path: str
    :param path: The path to load the base model from

    :rtype: miraiml.core.BaseModel
    :returns: The base model loaded from disk
    """
    attributes = load(path)
    return BaseModel(model_class=model_class,
                     parameters=attributes['parameters'],
                     features=attributes['features'])


class MiraiSeeker:
    """
    This class implements a smarter way of searching good parameters and sets of
    features.

    Read more in the :ref:`User Guide <mirai_seeker>`.

    :param base_models_ids: The list of base models' ids to keep track of.
    :type base_models_ids: list

    :param all_features: A list containing all available features.
    :type all_features: list

    :param config: The configuration of the engine.
    :type config: miraiml.Config
    """
    def __init__(self, search_spaces, all_features, config):
        self.all_features = all_features
        self.config = config

        histories_path = config.local_dir + 'histories/'

        if not os.path.exists(histories_path):
            os.makedirs(histories_path)

        self.search_spaces_dict = {}
        self.histories = {}
        self.histories_paths = {}
        for search_space in search_spaces:
            id = search_space.id
            self.search_spaces_dict[id] = search_space

            self.histories_paths[id] = histories_path + id
            if os.path.exists(self.histories_paths[id]):
                self.histories[id] = load(self.histories_paths[id])
            else:
                self.histories[id] = pd.DataFrame()
                dump(self.histories[id], self.histories_paths[id])

    def reset(self):
        """
        Deletes all base models registries.
        """
        for id in self.search_spaces_dict:
            self.histories[id] = pd.DataFrame()
            dump(self.histories[id], self.histories_paths[id])

    def parameters_features_to_dataframe(self, parameters, features, score):
        """
        Creates an entry for a history.

        :type parameters: dict
        :param parameters: The set of parameters to transform.

        :type parameters: list
        :param parameters: The set of features to transform.

        :type score: float
        :param score: The score to transform.
        """
        entry = {'score': score}
        for parameter in parameters:
            entry[parameter+'__(hyperparameter)'] = parameters[parameter]
        for feature in self.all_features:
            entry[feature+'__(feature)'] = 1 if feature in features else 0
        return pd.DataFrame([entry])

    def register_base_model(self, id, base_model, score):
        """
        Registers the performance of a base model and its characteristics.

        :type id: str
        :param id: The id associated with the base model.

        :type base_model: miraiml.core.BaseModel
        :param base_model: The base model being registered.

        :type score: float
        :param score: The score of ``base_model``.
        """
        new_entry = self.parameters_features_to_dataframe(
            base_model.parameters,
            base_model.features, score)

        self.histories[id] = pd.concat([self.histories[id], new_entry], sort=True)
        self.histories[id].drop_duplicates(inplace=True)
        dump(self.histories[id], self.histories_paths[id])

    def is_ready(self, id):
        """
        Tells whether the history of an id is large enough for more advanced
        strategies or not.

        :type id: str
        :param id: The id to be inspected.

        :rtype: bool
        :returns: Whether ``id`` can be used to generate parameters and features
            lists or not.
        """
        return self.histories[id].shape[0] > 1

    def seek(self, id):
        """
        Manages the search strategy for better solutions.

        With a probability of 0.5, the random strategy will be chosen. If it's
        not, the other strategies will be chosen with equal probabilities.

        :type id: str
        :param id: The id for which a new base model is required.

        :rtype: miraiml.core.BaseModel
        :returns: The next base model for exploration.

        :raises: ``KeyError`` if ``parameters_rules`` tries to access an invalid
            key.
        """
        if rnd.choice([0, 1]) == 1 or not self.is_ready(id):
            parameters, features = self.random_search(id)
        else:
            available_method_names = [method_name for method_name in dir(self)
                                      if method_name.endswith('_search')
                                      and method_name != 'random_search']

            method_name = rnd.choice(available_method_names)
            parameters, features = getattr(self, method_name)(id)

        search_space = self.search_spaces_dict[id]
        if len(parameters) > 0:
            try:
                search_space.parameters_rules(parameters)
            except Exception:
                raise KeyError('Error on parameters rules for the id {}'.format(id))

        model_class = search_space.model_class

        return BaseModel(model_class, parameters, features)

    def random_search(self, id):
        """
        Generates completely random sets of parameters and features.

        :type all_features: list
        :param all_features: The list of available features.

        :rtype: tuple
        :returns: ``(parameters, features)``
            Respectively, the dictionary of parameters and the list of features
            that can be used to generate a new base model.
        """
        search_space = self.search_spaces_dict[id]
        parameters = {}
        for parameter in search_space.parameters_values:
            parameters[parameter] = rnd.choice(
                search_space.parameters_values[parameter])
        if self.config.use_all_features:
            features = self.all_features
        else:
            features = sample_random_len(self.all_features)
        return (parameters, features)

    def naive_search(self, id):
        """
        Characteristics that achieved higher scores have independently higher
        chances of being chosen again.

        :type id: str
        :param id: The id for which we want a new set of parameters and features.

        :rtype: tuple
        :returns: ``(parameters, features)``
            Respectively, the dictionary of parameters and the list of features
            that can be used to generate a new base model.
        """
        history = self.histories[id]
        parameters = {}
        features = []
        for column in history.columns:
            if column == 'score':
                continue
            dist = history[[column, 'score']].groupby(column).mean().reset_index()
            chosen_value = rnd.choices(
                dist[column].values,
                cum_weights=dist['score'].cumsum().values)[0]
            del dist
            if column.endswith('__(hyperparameter)'):
                parameter = column.split('__(')[0]
                parameters[parameter] = chosen_value
            elif column.endswith('__(feature)'):
                feature = column.split('__(')[0]
                if self.config.use_all_features:
                    features.append(feature)
                else:
                    if chosen_value:
                        features.append(feature)
        if len(features) == 0:
            features = sample_random_len(self.all_features)
        return (parameters, features)

    @staticmethod
    def __dataframe_to_parameters_features__(dataframe):
        """
        Transforms a history entry in a pair of parameters and features.

        :type dataframe: pandas.DataFrame
        :param dataframe: The history entry to be transformed,

        :rtype: tuple
        :returns: ``(parameters, features)``. The transformed sets of parameters
            and features, respectively.
        """
        parameters = {}
        features = []
        for column in dataframe.columns:
            if column == 'score':
                continue
            column_filtered = column.split('__(')[0]
            value = dataframe[column].values[0]
            if column.endswith('__(hyperparameter)'):
                parameters[column_filtered] = value
            elif column.endswith('__(feature)'):
                if value:
                    features.append(column_filtered)
        return (parameters, features)

    def linear_regression_search(self, id):
        """
        Uses the history to model the score with a linear regression. Guesses the
        scores of `n`/2 random sets of parameters and features, where `n` is the
        size of the history. The one with the highest score is chosen.

        :type id: str
        :param id: The id for which we want a new set of parameters and features.

        :rtype: tuple
        :returns: ``(parameters, features)``
            Respectively, the dictionary of parameters and the list of features
            that can be used to generate a new base model.
        """
        history = self.histories[id]
        n_guesses = history.shape[0]//2

        # Creating guesses:
        guesses_df = pd.DataFrame()
        for _ in range(n_guesses):
            guess_parameters, guess_features = self.random_search(id)
            guess_df = self.parameters_features_to_dataframe(
                guess_parameters, guess_features, np.nan)
            guesses_df = pd.concat([guesses_df, guess_df], sort=True)

        # Concatenating data to perform one-hot-encoding:
        data = pd.concat([history, guesses_df], sort=True)
        object_columns = [col for col in data.columns if data[col].dtype == object]
        data_ohe = pd.get_dummies(data, columns=object_columns, drop_first=True)

        # Separating train and test:
        train_mask = data_ohe['score'].notna()
        data_ohe_train = data_ohe[train_mask]
        data_ohe_test = data_ohe[~train_mask].drop(columns='score')
        y = data_ohe_train.pop('score')

        # Fitting and predicting scores:
        model = LinearRegression(normalize=True)
        model.fit(data_ohe_train, y)
        guesses_df['score'] = model.predict(data_ohe_test)

        # Choosing the best guess:
        best_guess = guesses_df.sort_values('score', ascending=False).head(1).copy()

        del guesses_df, data, data_ohe, data_ohe_train, data_ohe_test, y, model

        return self.__dataframe_to_parameters_features__(best_guess)


class Ensembler:
    """
    Performs the ensemble of the base models and optimizes its weights.

    Read more in the :ref:`User Guide <ensemble>`.

    :type y_train: pandas.Series or numpy.ndarray
    :param y_train: The target column.

    :type base_models_ids: list
    :param base_models_ids: The list of base models' ids to keep track of.

    :type train_predictions_df: pandas.DataFrame
    :param train_predictions_df: The dataframe of predictions for the training
        dataset.

    :type test_predictions_df: pandas.DataFrame
    :param test_predictions_df: The dataframe of predictions for the testing
        dataset.

    :type scores: dict
    :param scores: The dictionary of scores.

    :type config: miraiml.Config
    :param config: The configuration of the engine.
    """
    def __init__(self, base_models_ids, y_train, train_predictions_df,
                 test_predictions_df, scores, config):
        self.y_train = y_train
        self.base_models_ids = sorted(base_models_ids)
        self.train_predictions_df = train_predictions_df
        self.test_predictions_df = test_predictions_df
        self.scores = scores
        self.config = config
        self.id = config.ensemble_id
        self.weights_path = config.local_dir + 'models/' + self.id
        self.must_interrupt = False

        if os.path.exists(self.weights_path):
            self.weights = load(self.weights_path)
        else:
            self.weights = self.gen_weights()
            dump(self.weights, self.weights_path)

    def interrupt(self):
        """
        Sets an internal flag to interrupt the optimization process on the first
        opportunity.
        """
        self.must_interrupt = True

    def update(self):
        """
        Updates the ensemble with the newest predictions from the base models.
        """
        train_predictions, test_predictions, score = self.ensemble(self.weights)
        self.train_predictions_df[self.id] = train_predictions
        self.test_predictions_df[self.id] = test_predictions
        self.scores[self.id] = score

    def gen_weights(self):
        """
        Generates the ensemble weights according to the score of each base model.
        Higher scores have higher chances of generating higher weights.

        :rtype: dict
        :returns: A dictionary containing the weights for each base model id.
        """
        weights = {}
        if len(self.scores) > 0:
            min_score, max_score = np.inf, -np.inf
            for id in self.base_models_ids:
                score = self.scores[id]
                min_score = min(min_score, score)
                max_score = max(max_score, score)
            diff_score = max_score - min_score
            for id in self.base_models_ids:
                if self.scores[id] == max_score:
                    weights[id] = rnd.triangular(0, 1, 1)
                else:
                    normalized_score = (self.scores[id]-min_score)/diff_score
                    range_ = rnd.triangular(0, 1, normalized_score)
                    weights[id] = rnd.triangular(0, range_, 0)
        else:
            for id in self.base_models_ids:
                weights[id] = 1
        return weights

    def ensemble(self, weights):
        """
        Performs the ensemble of the current predictions of each base model.

        :type weights: dict
        :param weights: A dictionary containing the weights related to the id of
            each base model.

        :rtype: tuple
        :returns: ``(train_predictions, test_predictions, score)``

            * ``train_predictions``: The ensemble predictions for the training dataset
            * ``test_predictions``: The ensemble predictions for the testing dataset
            * ``score``: The score of the ensemble on the training dataset
        """
        weights_list = [weights[id] for id in self.base_models_ids]
        train_predictions = np.average(
            self.train_predictions_df[self.base_models_ids],
            axis=1, weights=weights_list)
        test_predictions = None
        if self.test_predictions_df.shape[0] > 0:
            test_predictions = np.average(
                self.test_predictions_df[self.base_models_ids],
                axis=1, weights=weights_list)
        return (train_predictions, test_predictions,
                self.config.score_function(self.y_train, train_predictions))

    def optimize(self, max_duration):
        """
        Performs ensembling cycles for ``max_duration`` seconds.

        :type max_duration: float
        :param max_duration: The maximum duration allowed for the optimization
            process.

        :rtype: bool
        :returns: ``True`` if a better set of weights was found and ``False``
            otherwise.
        """
        optimized = False
        start = time.time()
        while time.time() - start < max_duration and not self.must_interrupt:
            weights = self.gen_weights()
            train_predictions, test_predictions, score = self.ensemble(weights)
            if self.id not in self.scores or score > self.scores[self.id]:
                self.scores[self.id] = score
                self.weights = weights
                self.train_predictions_df[self.id] = train_predictions
                self.test_predictions_df[self.id] = test_predictions
                dump(self.weights, self.weights_path)
                optimized = True
            else:
                del weights, train_predictions, test_predictions
        return optimized


class BasePipelineClass:
    """
    This is the base class for your custom pipeline classes.

    .. warning::
        Instantiating this class directly **does not work**.
    """
    def __init__(self, **params):
        self.pipeline = Pipeline(
            # self.steps has been set from outside at this point!
            [(alias, class_type()) for (alias, class_type) in self.steps]
        )
        self.set_params(**params)

    def get_params(self):
        """
        Gets the list of parameters that can be set.

        :type X: iterable
        :param X: Data to predict on.

        :rtype: list
        :returns: The list of allowed parameters
        """
        params = [param for param in self.pipeline.get_params() if
                  'copy' not in param]
        prefixes = [alias + '__' for alias, _ in self.steps]
        return [param for param in params if
                any([param.startswith(prefix) for prefix in prefixes])]

    def set_params(self, **params):
        """
        Sets the parameters for the pipeline. You can check the parameters that
        are allowed to be set by calling :func:`get_params`.

        :rtype: miraiml.core.BasePipelineClass
        :returns: self
        """
        allowed_params = self.get_params()
        for param in params:
            if param not in allowed_params:
                raise ValueError(
                    'Parameter ' + param + ' is incompatible. The allowed ' +
                    'parameters are:\n' + ', '.join(allowed_params)
                )
        self.pipeline.set_params(**params)
        return self

    def fit(self, X, y):
        """
        Fits the pipeline to ``X`` using ``y`` as the target.

        :type X: iterable
        :param X: The training data.

        :type y: iterable
        :param y: The target.

        :rtype: miraiml.core.BasePipelineClass
        :returns: self
        """
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        """
        Predicts the class for each element of ``X`` in case of classification
        problems or the estimated target value in case of regression problems.

        :type X: iterable
        :param X: Data to predict on.

        :rtype: numpy.ndarray
        :returns: The set of predictions
        """
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        """
        Returns the probabilities for each class. Available only if your end
        estimator implements it.

        :type X: iterable
        :param X: Data to predict on.

        :rtype: numpy.ndarray
        :returns: The probabilities for each class
        """
        return self.pipeline.predict_proba(X)
