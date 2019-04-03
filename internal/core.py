from sklearn.model_selection import StratifiedKFold, KFold
from random import choices
import pandas as pd
import numpy as np
import os

from .util import load, par_dump, sample_random_len

class BaseModel:
    """
    Represents an element from the search hyperspace of an object of BaseLayout,
    linking a set of parameters to a set of features. As an analogy, it
    represents a particular choice of clothes that someone can make.

    This is the basic brick for the optimizations.

    Parameters
    ----------
    model_class : A statistical model class that must implement the methods `fit`
        and `predict` for regression or `predict_proba` classification problems.

    parameters : dict
        The parameters that will be used to instantiate objects of `model_class`.

    features : list
        The list of features that will be used to train the statistical model.
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
        dataset will be complete and there will be `n_folds` sets of predictions
        for the testing dataset. The final set of predictions for the testing
        dataset is the mean of the `n_folds` predictions.

        This mechanic may produce more stable predictions for the testing dataset
        than for the training dataset, resulting in slightly better accuracies
        than expected.

        Parameters
        ----------
        X_train : pandas.DataFrame
            The dataframe that contains the training inputs for the model.

        y_train : pandas.Series
            The series of training targets for the model.

        X_test : pandas.DataFrame
            The dataframe that contains the testing inputs for the model.

        config : miraiml.Config
            The configuration of the engine.

        Returns
        -------
        train_predictions : numpy.array
            The predictions for the training dataset.

        test_predictions : numpy.array
            The predictions for the testing dataset.

        score : float
            The score of the model on the training dataset.
        """
        X_train, X_test = X_train[self.features], X_test[self.features]
        train_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros(X_test.shape[0])
        if config.problem_type == 'classification' and config.stratified:
            fold = StratifiedKFold(n_splits=config.n_folds, shuffle=False)
        elif config.problem_type == 'regression' or not config.stratified:
            fold = KFold(n_splits=config.n_folds, shuffle=False)
        for big_part, small_part in fold.split(X_train, y_train):
            X_train_big, X_train_small = X_train.values[big_part], X_train.values[small_part]
            y_train_big = y_train.values[big_part]

            model = self.model_class(**self.parameters)

            model.fit(X_train_big, y_train_big)
            if config.problem_type == 'classification':
                train_predictions[small_part] = model.predict_proba(X_train_small)[:,1]
                test_predictions += model.predict_proba(X_test)[:,1]
            elif config.problem_type == 'regression':
                train_predictions[small_part] = model.predict(X_train_small)
                test_predictions += model.predict(X_test)

        test_predictions /= config.n_folds
        return (train_predictions, test_predictions, config.score_function(y_train,
            train_predictions))

class MiraiSeeker:
    """
    This class implements a smarter way of searching good parameters and sets of
    features.

    Parameters
    ----------
    ids : list
        A list of miraiml.BaseModels ids to keep track of.

    all_features : list
        A list containing all available features.

    config : miraiml.Config
        The configuration of the engine.
    """
    def __init__(self, ids, all_features, config):
        self.ids = ids
        self.all_features = all_features
        self.history_path = config.local_dir + 'history'

        if os.path.exists(self.history_path):
            self.history = load(self.history_path)
        else:
            self.reset()

    def reset(self):
        """
        Cleans all base models registries.
        """
        self.history = {}
        for id in self.ids:
            self.history[id] = pd.DataFrame()
        par_dump(self.history, self.history_path)

    def register_base_model(self, id, base_model, score):
        """
        Registers the performance of a base model and its characteristics.

        Parameters
        ----------
        id : int or string
            The id of the model layout that represents the base model.

        base_model : miraiml.internal.core.BaseModel
            The base model being registered.

        score : float
            The score of `base_model`.
        """
        event = {'score':score}
        for parameter in base_model.parameters:
            event[parameter+'(parameter)'] = base_model.parameters[parameter]
        for feature in self.all_features:
            event[feature+'(feature)'] = feature in base_model.features

        self.history[id] = pd.concat([self.history[id],
            pd.DataFrame([event])]).drop_duplicates()
        par_dump(self.history, self.history_path)

    def is_ready(self, id):
        """
        Tells whether it's ready to work for an id or not.

        Parameters
        ----------
        id : int or string
            The id to be inspected.

        Returns
        -------
        ready : bool
            A boolean that tells whether an id can be used to generate parameters
            and features lists or not.
        """
        return self.history[id].shape[0] > 1

    def gen_parameters_features(self, id):
        """
        For each hyperparameter and feature, its value (True or False for
        features) is chosen stochastically depending on the mean score of the
        registered entries in which the value was chosen before. Better
        parameters and features have higher chances of being chosen.

        Parameters
        ----------
        id : int or string
            The id for which we want a new set of parameters and features.

        Returns
        -------
        (parameters, features) : (dict, list)
            Respectively, the set of parameters and the list of features that can
            be used to generate a new base model.
        """
        history = self.history[id]
        parameters = {}
        features = []
        for column in history.columns:
            if column == 'score':
                continue
            dist = history[[column, 'score']].groupby(column).mean().reset_index()
            chosen_value = choices(dist[column].values,
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
