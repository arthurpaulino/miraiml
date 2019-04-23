"""
:mod:`miraiml.core` contains internal classes responsible for the optimization
process.

- :class:`miraiml.core.BaseModel` represents a solution
- :class:`miraiml.core.MiraiSeeker` implements the strategies to search for good
  solutions
- :class:`miraiml.core.Ensembler` searches for smart ways of combining the current
  solutions o generate a better one
"""

import random as rnd
import pandas as pd
import numpy as np
import time
import os

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression

from .util import load, dump, sample_random_len

class BaseModel:
    """
    Represents an element from the search space, defined by an instance of
    :class:`miraiml.HyperSearchSpace` and a set of features.

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

        :raises: ``RuntimeError``
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
            except:
                raise RuntimeError('Error when fitting with model class {}'.\
                    format(class_name))
            try:
                if config.problem_type == 'classification':
                    train_predictions[small_part] = model.predict_proba(X_train_small)[:,1]
                    if X_test is not None:
                        test_predictions += model.predict_proba(X_test)[:,1]
                elif config.problem_type == 'regression':
                    train_predictions[small_part] = model.predict(X_train_small)
                    if X_test is not None:
                        test_predictions += model.predict(X_test)
            except:
                raise RuntimeError('Error when predicting with model class {}'.\
                    format(class_name))

        if X_test is not None:
            test_predictions /= config.n_folds
        return (train_predictions, test_predictions,
                config.score_function(y_train, train_predictions))

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
    def __init__(self, hyper_search_spaces, all_features, config):
        self.all_features = all_features
        self.config = config

        histories_path = config.local_dir + 'histories/'

        if not os.path.exists(histories_path):
            os.makedirs(histories_path)

        self.hyper_search_spaces_dict = {}
        self.histories = {}
        self.histories_paths = {}
        for hyper_search_space in hyper_search_spaces:
            id = hyper_search_space.id
            self.hyper_search_spaces_dict[id] = hyper_search_space

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
        for id in self.hyper_search_spaces_dict:
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
        entry = {'score':score}
        for parameter in parameters:
            entry[parameter+'(parameter)'] = parameters[parameter]
        for feature in self.all_features:
            entry[feature+'(feature)'] = 1 if feature in features else 0
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

        self.histories[id] = pd.concat([self.histories[id], new_entry])
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

        :raises: ``KeyError``
        """
        if rnd.uniform(0, 1) > 0.5 or not self.is_ready(id):
            parameters, features = self.random_search(id)
        else:
            if rnd.uniform(0, 1) > 0.5:
                parameters, features = self.naive_search(id)
            else:
                parameters, features = self.linear_regression_search(id)

        hyper_search_space = self.hyper_search_spaces_dict[id]
        if len(parameters) > 0:
            try:
                hyper_search_space.parameters_rules(parameters)
            except:
                raise KeyError('Error on parameters rules for the id {}'.format(id))

        model_class = hyper_search_space.model_class

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
        hyper_search_space = self.hyper_search_spaces_dict[id]
        parameters = {}
        for parameter in hyper_search_space.parameters_values:
            parameters[parameter] = rnd.choice(
                hyper_search_space.parameters_values[parameter])
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
            if column.endswith('(parameter)'):
                parameter = column.split('(')[0]
                parameters[parameter] = chosen_value
            elif column.endswith('(feature)'):
                feature = column.split('(')[0]
                if chosen_value:
                    features.append(feature)
        if len(features) == 0:
            features = sample_random_len(self.all_features)
        return (parameters, features)

    @classmethod
    def dataframe_to_parameters_features(cls, dataframe):
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
            column_filtered = column.split('(')[0]
            value = dataframe[column].values[0]
            if column.endswith('(parameter)'):
                parameters[column_filtered] = value
            elif column.endswith('(feature)'):
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
            guesses_df = pd.concat([guesses_df, guess_df])

        # Concatenating data to perform one-hot-encoding:
        data = pd.concat([history, guesses_df])
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
        best_guess = guesses_df.sort_values('score', ascending=False).head(1)

        return self.dataframe_to_parameters_features(best_guess)

class Ensembler:
    """
    Performs the ensemble of the base models.

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
        return optimized
