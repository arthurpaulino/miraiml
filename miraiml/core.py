"""
:mod:`miraiml.core` contains the basic bricks for the optimization process.

- :class:`miraiml.core.BaseModel` represents a solution
- :class:`miraiml.core.MiraiSeeker` implements the strategies to search for good
  solutions
- :class:`miraiml.core.Ensembler` searches for smart ways of combining the current
  solutions o generate a better one
"""

from sklearn.model_selection import StratifiedKFold, KFold
import random as rnd
import pandas as pd
import numpy as np
import os

from .util import load, dump, sample_random_len

class BaseModel:
    """
    Represents an element from the search hyperspace defined by a
    :class:`miraiml.SearchSpace`, linking a set of parameters to a set of features.
    As an analogy, it represents a particular choice of clothes that someone can
    make.

    :param model_class: A statistical model class that must implement the methods
        ``fit`` and ``predict`` for regression or ``predict_proba`` classification
        problems.
    :type model_class: type

    :param parameters: The parameters that will be used to instantiate objects of
        ``model_class``.
    :type parameters: dict

    :param features: The list of features that will be used to train the statistical
        model.
    :type features: list
    """
    def __init__(self, model_class, parameters, features):
        self.model_class = model_class
        self.parameters = parameters
        self.features = features

    def predict(self, X_train, y_train, X_test, config):
        """
        Performs the predictions for the training and testing datasets and also
        provides the score of the model.

        For each fold of the training dataset, the model trains on the bigger
        part and then make predictions for the smaller part and for the testing
        dataset. After iterating over all folds, the predictions for the training
        dataset will be complete and there will be ``config.n_folds`` sets of
        predictions for the testing dataset. The final set of predictions for the
        testing dataset is the mean of the ``config.n_folds`` predictions.

        This mechanic may produce more stable predictions for the testing dataset
        than for the training dataset, resulting in slightly better accuracies
        than expected.

        :param X_train: The dataframe that contains the training inputs for the
            model.
        :type X_train: pandas.DataFrame

        :param y_train: The training targets for the model.
        :type y_train: pandas.Series or numpy.ndarray

        :param X_test: The dataframe that contains the testing inputs for the model.
        :type X_test: pandas.DataFrame

        :param config: The configuration of the engine.
        :type config: miraiml.Config

        :rtype: tuple
        :returns: ``(train_predictions, test_predictions, score)``

            * ``train_predictions``: The predictions for the training dataset
            * ``test_predictions``: The predictions for the testing dataset
            * ``score``: The score of the model on the training dataset

        :raises: ``RuntimeError``
        """
        X_train, X_test = X_train[self.features], X_test[self.features]
        train_predictions = np.zeros(X_train.shape[0])
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
                raise RuntimeError('Error when fitting with model class \'{}\'.'.\
                    format(class_name))
            try:
                if config.problem_type == 'classification':
                    train_predictions[small_part] = model.predict_proba(X_train_small)[:,1]
                    test_predictions += model.predict_proba(X_test)[:,1]
                elif config.problem_type == 'regression':
                    train_predictions[small_part] = model.predict(X_train_small)
                    test_predictions += model.predict(X_test)
            except:
                raise RuntimeError('Error when predicting with model class \'{}\'.'.\
                    format(class_name))

        test_predictions /= config.n_folds
        return (train_predictions, test_predictions, config.score_function(y_train,
            train_predictions))

class MiraiSeeker:
    """
    This class implements a smarter way of searching good parameters and sets of
    features.

    :param base_models_ids: The list of base models' ids to keep track of.
    :type base_models_ids: list

    :param all_features: A list containing all available features.
    :type all_features: list

    :param config: The configuration of the engine.
    :type config: miraiml.Config
    """
    def __init__(self, search_spaces, all_features, config):
        self.search_spaces_dict = {}
        for search_space in search_spaces:
            self.search_spaces_dict[search_space.id] = search_space
        self.all_features = all_features
        self.config = config
        self.history_path = config.local_dir + 'history'

        if os.path.exists(self.history_path):
            self.history = load(self.history_path)
        else:
            self.reset()

    def reset(self):
        """
        Deletes all base models registries.
        """
        self.history = {}
        for id in self.search_spaces_dict:
            self.history[id] = pd.DataFrame()
        dump(self.history, self.history_path)

    def register_base_model(self, id, base_model, score):
        """
        Registers the performance of a base model and its characteristics.

        :param id: The id associated with the base model.
        :type id: str

        :param base_model: The base model being registered.
        :type base_model: miraiml.core.BaseModel

        :param score: The score of ``base_model``.
        :type score: float
        """
        event = {'score':score}
        for parameter in base_model.parameters:
            event[parameter+'(parameter)'] = base_model.parameters[parameter]
        for feature in self.all_features:
            event[feature+'(feature)'] = 1 if feature in base_model.features else 0

        self.history[id] = pd.concat([self.history[id],
            pd.DataFrame([event])]).drop_duplicates()
        dump(self.history, self.history_path)

    def is_ready(self, id):
        """
        Tells whether it's ready to work for an id or not.

        :param id: The id to be inspected.
        :type id: str

        :rtype: bool
        :returns: Whether ``id`` can be used to generate parameters and features
            lists or not.
        """
        return self.history[id].shape[0] > 1

    def seek(self, id):
        """
        Manages the search strategy throughout the optimization hyperspace.

        :param id: The id for which a new base model is required.
        :type id: str

        :rtype: miraiml.core.BaseModel
        :returns: The next base model for exploration.

        :raises: ``KeyError``
        """
        if self.is_ready(id) and rnd.uniform(0, 1) > self.config.random_exploration_ratio:
            parameters, features = self.naive_search(id)
        else:
            parameters, features = self.random_search(id)

        search_space = self.search_spaces_dict[id]
        if len(parameters) > 0:
            try:
                search_space.parameters_rules(parameters)
            except:
                raise KeyError('Error on parameters rules for the id \'{}\'.'.format(id))
        model_class = search_space.model_class

        return BaseModel(model_class, parameters, features)

    def random_search(self, id):
        """
        Generates a completely random instance of :class:`miraiml.core.BaseModel`.

        :param all_features: The list of available features.
        :type all_features: list

        :rtype: tuple
        :returns: ``(parameters, features)``
            Respectively, the dictionary of parameters and the list of features
            that can be used to generate a new base model.
        """
        search_space = self.search_spaces_dict[id]
        model_class = search_space.model_class
        parameters = {}
        for parameter in search_space.parameters_values:
            parameters[parameter] = rnd.choice(search_space.parameters_values[parameter])
        features = sample_random_len(self.all_features)
        return (parameters, features)

    def naive_search(self, id):
        """
        For each hyperparameter and feature, its value (True or False for
        features) is chosen stochastically depending on the mean score of the
        registered entries in which the value was chosen before. Better
        parameters and features have higher chances of being chosen.

        :param id: The id for which we want a new set of parameters and features.
        :type id: str

        :rtype: tuple
        :returns: ``(parameters, features)``
            Respectively, the dictionary of parameters and the list of features
            that can be used to generate a new base model.
        """
        history = self.history[id]
        parameters = {}
        features = []
        for column in history.columns:
            if column == 'score':
                continue
            dist = history[[column, 'score']].groupby(column).mean().reset_index()
            chosen_value = rnd.choices(dist[column].values,
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

class Ensembler:
    """
    Performs the ensemble of the base models.

    :param y_train: The target column.
    :type y_train: pandas.Series or numpy.ndarray

    :param base_models_ids: The list of base models' ids to keep track of.
    :type base_models_ids: list

    :param train_predictions_dict: The dictionary of predictions for the training
        dataset.
    :type train_predictions_dict: dict

    :param test_predictions_dict: The dictionary of predictions for the testing
        dataset.
    :type test_predictions_dict: dict

    :param scores: The dictionary of scores.
    :type scores: dict

    :param config: The configuration of the engine.
    :type config: miraiml.Config
    """
    def __init__(self, base_models_ids, y_train, train_predictions_dict,
            test_predictions_dict, scores, config):
        self.y_train = y_train
        self.base_models_ids = base_models_ids
        self.train_predictions_dict = train_predictions_dict
        self.test_predictions_dict = test_predictions_dict
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

        :rtype: tuple
        :returns: ``(train_predictions, test_predictions, score)``: Updated
            predictions and score.

            * ``train_predictions``: The ensemble predictions for the training dataset
            * ``test_predictions``: The ensemble predictions for the testing dataset
            * ``score``: The score of the ensemble on the training dataset
        """
        train_predictions, test_predictions, score = self.ensemble(self.weights)
        self.train_predictions_dict[self.id] = train_predictions
        self.test_predictions_dict[self.id] = test_predictions
        self.scores[self.id] = score
        return (train_predictions, test_predictions, score)

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
                    range = rnd.triangular(0, 1, normalized_score)
                    weights[id] = rnd.triangular(0, range, 0)
        else:
            for id in self.base_models_ids:
                weights[id] = 1
        return weights

    def ensemble(self, weights):
        """
        Performs the ensemble of the current predictions of each base model.

        :param weights: A dictionary containing the weights related to the id of
            each base model.
        :type weights: dict

        :rtype: tuple
        :returns: ``(train_predictions, test_predictions, score)``

            * ``train_predictions``: The ensemble predictions for the training dataset
            * ``test_predictions``: The ensemble predictions for the testing dataset
            * ``score``: The score of the ensemble on the training dataset
        """
        id = self.base_models_ids[0]
        train_predictions = weights[id]*self.train_predictions_dict[id]
        test_predictions = weights[id]*self.test_predictions_dict[id]
        weights_sum = weights[id]
        for id in self.base_models_ids[1:]:
            if id in self.train_predictions_dict:
                train_predictions += weights[id]*self.train_predictions_dict[id]
                test_predictions += weights[id]*self.test_predictions_dict[id]
                weights_sum += weights[id]
        train_predictions /= weights_sum
        test_predictions /= weights_sum
        return (train_predictions, test_predictions,
            self.config.score_function(self.y_train, train_predictions))

    def optimize(self):
        """
        Performs ``config.n_ensemble_cycles`` attempts to improve ensemble weights.

        :rtype: bool
        :returns: ``True`` if a better set of weights was found and ``False``
            otherwise.
        """
        optimized = False
        for _ in range(self.config.n_ensemble_cycles):
            if self.must_interrupt:
                break
            weights = self.gen_weights()
            train_predictions, test_predictions, score = self.ensemble(weights)
            if self.id not in self.scores or score > self.scores[self.id]:
                self.scores[self.id] = score
                self.weights = weights
                self.train_predictions_dict[self.id] = train_predictions
                self.test_predictions_dict[self.id] = test_predictions
                dump(self.weights, self.weights_path)
                optimized = True
        return optimized
