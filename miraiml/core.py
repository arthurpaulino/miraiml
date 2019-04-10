__all__ = ['BaseModel', 'MiraiSeeker', 'Ensembler']

from sklearn.model_selection import StratifiedKFold, KFold
from random import triangular, choice, choices, uniform
from threading import Thread
from time import sleep, time
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

    :param base_models_ids: The list of base models' ids to keep track of.
    :type base_models_ids: list

    :param all_features: A list containing all available features.
    :type all_features: list

    :param config: The configuration of the engine.
    :type config: miraiml.Config
    """
    def __init__(self, base_models_ids, all_features, config):
        self.base_models_ids = base_models_ids
        self.all_features = all_features
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
        for id in self.base_models_ids:
            self.history[id] = pd.DataFrame()
        par_dump(self.history, self.history_path)

    def register_base_model(self, id, base_model, score):
        """
        Registers the performance of a base model and its characteristics.

        :param id: The id of the model layout that represents the base model.
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
            event[feature+'(feature)'] = feature in base_model.features

        self.history[id] = pd.concat([self.history[id],
            pd.DataFrame([event])]).drop_duplicates()
        par_dump(self.history, self.history_path)

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

    def gen_parameters_features(self, id):
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
        self.must_interrupt = False

        self.id = config.ensemble_id
        self.weights_path = config.local_dir + 'models/' + self.id

        if os.path.exists(self.weights_path):
            self.weights = load(self.weights_path)
        else:
            self.weights = self.gen_weights()
            par_dump(self.weights, self.weights_path)

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
        """
        weights = {}
        min_score, max_score = np.inf, -np.inf
        for id in self.base_models_ids:
            score = self.scores[id]
            min_score = min(min_score, score)
            max_score = max(max_score, score)
        diff_score = max_score - min_score
        for id in self.base_models_ids:
            if self.scores[id] == max_score:
                weights[id] = triangular(0, 1, 1)
            else:
                normalized_score = (self.scores[id]-min_score)/diff_score
                range = triangular(0, 1, normalized_score)
                weights[id] = triangular(0, range, 0)
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
                par_dump(self.weights, self.weights_path)
                optimized = True
        return optimized

class BaseLayout:
    """
    This class represents the search hyperspace for a base statistical model. As
    an analogy, it represents all possible sets of clothes that someone can use.

    :param model_class: Any class that represents a statistical model. It must
        implement the methods ``fit`` as well as ``predict`` for regression or
        ``predict_proba`` for classification problems.
    :type model_class: type

    :param id: An id to be associated with this layout.
    :type id: str

    :type parameters_values: dict
    :param parameters_values: optional, ``default={}``. A dictionary containing
        a list of values to be tested as parameters to instantiate objects of
        ``model_class``.

    :type parameters_rules: function
    :param parameters_rules: optional, ``default=lambda x: None``. A function that
        constrains certain parameters because of the values assumed by others.

    :Example:

    ::

        from sklearn.linear_model import LogisticRegression
        from miraiml import BaseLayout

        def logistic_regression_parameters_rules(parameters):
        if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
            parameters['penalty'] = 'l2'

        base_layout = BaseLayout(LogisticRegression, 'Logistic Regression', {
                'penalty': ['l1', 'l2'], # may assume values 'l1' or 'l2'
                'C': np.arange(0.1, 2, 0.1), # may assume values .1, .2, ... or 1.9
                'max_iter': np.arange(50, 300),
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'random_state': [0] # IMPORTANT
            },
            parameters_rules=logistic_regression_parameters_rules
        )

    .. warning::
        **Do not** allow ``random_state`` assume multiple values. If ``model_class``
        has a ``random_state`` parameter, force ``BaseLayout`` to always choose
        the same value by providing a list with a single element.

        Allowing ``random_state`` to assume multiple values will confuse the engine
        because the scores will be unstable even with the same choice of
        hyperparameters and features.
    """
    def __init__(self, model_class, id, parameters_values={},
            parameters_rules=lambda x: None):
        self.model_class = model_class
        self.id = id
        self.parameters_values = parameters_values
        self.parameters_rules = parameters_rules

    def gen_parameters_features(self, all_features):
        """
        Generates completely random parameters and features set.

        :param all_features: The list of available features.
        :type all_features: list

        :rtype: tuple
        :returns: ``(parameters, features)``

            Respectively, the dictionary of parameters and the list of features
            that can be used to generate a new base model.
        """
        parameters = {}
        for parameter in self.parameters_values:
            parameters[parameter] = choice(self.parameters_values[parameter])
        features = sample_random_len(all_features)
        return (parameters, features)

class Config:
    """
    This class defines the general behavior of the engine.

    :param local_dir: The path for the engine to save its internal files. If the
        directory doesn't exist, it will be created automatically.
    :type local_dir: str

    :type problem_type: str
    :param problem_type: ``'classification'`` or ``'regression'``. The problem
        type. Multi-class classification problems are not supported.

    :param base_layouts: The list of :class:`miraiml.BaseLayout` objects to optimize.
    :type base_layouts: list

    :param n_folds: The number of folds for cross-validations.
    :type n_folds: int

    :param stratified: Whether to stratify folds on target or not. Only used if
        ``problem_type == 'classification'``.
    :type stratified: bool

    :param score_function: A function that receives the "truth" and the predictions
        (in this order) and returns the score. Bigger scores must mean better models.
    :type score_function: function

    :param mirai_exploration_ratio: The proportion of attempts in which the engine
        will explore the search space by using an instance of
        :class:`miraiml.core.MiraiSeeker`.
    :type mirai_exploration_ratio: float

    :param ensemble_id: The id for the ensemble.
    :type ensemble_id: str

    :param n_ensemble_cycles: The number of times that the engine will attempt to
        improve the ensemble weights in each loop after optimizing all base models.
    :type n_ensemble_cycles: int

    :param report: Whether the engine should output the models' scores after
        improvements or not.
    :type report: bool

    :Example:

    ::

        from sklearn.metrics import roc_auc_score
        from miraiml import Config
        config = Config(
            local_dir = 'miraiml_local',
            problem_type = 'classification',
            base_layouts = base_layouts,
            n_folds = 5,
            stratified = True,
            score_function = roc_auc_score,
            mirai_exploration_ratio = 0.5,
            ensemble_id = 'Ensemble',
            n_ensemble_cycles = 1000,
            report = False
        )
    """
    def __init__(self, local_dir, problem_type, base_layouts, n_folds, stratified,
            score_function, mirai_exploration_ratio, ensemble_id, n_ensemble_cycles,
            report):
        self.local_dir = local_dir
        if self.local_dir[-1] != '/':
            self.local_dir += '/'
        self.problem_type = problem_type
        self.base_layouts = base_layouts
        self.n_folds = n_folds
        self.stratified = stratified
        self.score_function = score_function
        self.mirai_exploration_ratio = mirai_exploration_ratio
        self.ensemble_id = ensemble_id
        self.n_ensemble_cycles = n_ensemble_cycles
        self.report = report

class Engine:
    """
    This class offers the controls for the engine.

    :param config: The configurations for the behavior of the engine.
    :type config: miraiml.Config

    :Example:

    ::

        from miraiml import Engine
        engine = Engine(config)
    """
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.must_interrupt = False
        self.mirai_seeker = None
        self.models_dir = config.local_dir + 'models/'
        self.X_train = None
        self.ensembler = None

    def is_running(self):
        """
        Tells whether the engine is running or not.

        :rtype: bool
        :returns: ``True`` if the engine is running and ``False`` otherwise.
        """
        return self.is_running

    def interrupt(self):
        """
        Sets an internal flag to make the engine stop on the first opportunity.
        """
        self.must_interrupt = True
        if not self.ensembler is None:
            self.ensembler.interrupt()
        while self.is_running:
            sleep(.1)
        self.must_interrupt = False

    def update_data(self, train_data, test_data, target, restart=False):
        """
        Interrupts the engine and loads a new pair of train/test datasets.

        :param train_data: The training data.
        :type train_data: pandas.DataFrame

        :param test_data: The testing data.
        :type test_data: pandas.DataFrame

        :param target: The label of the target column.
        :type target: str

        :param restart: Optional, ``default=False``. Whether to restart the engine
            after updating data or not.
        :type restart: bool
        """
        self.interrupt()
        self.X_train = train_data.drop(columns=target)
        self.all_features = list(self.X_train.columns)
        self.y_train = train_data[target]
        self.X_test = test_data
        if not self.mirai_seeker is None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def shuffle_train_data(self, restart=False):
        """
        Interrupts the engine and shuffles the training data.

        :param restart: Optional, ``default=False``. Whether to restart the engine
            after shuffling data or not.
        :type restart: bool

        .. note::
            It's a good practice to shuffle the training data periodically to avoid
            overfitting on a certain folding pattern.
        """
        self.interrupt()
        if not self.X_train is None:
            seed = int(time())
            self.X_train = self.X_train.sample(frac=1, random_state=seed)
            self.y_train = self.y_train.sample(frac=1, random_state=seed)
        if restart:
            self.restart()

    def reconfigure(self, config, restart=False):
        """
        Interrupts the engine and loads a new configuration.

        :param config: The configurations for the behavior of the engine.
        :type config: miraiml.Config

        :param restart: Optional, ``default=False``. Whether to restart the engine
            after reconfiguring it or not.
        :type restart: bool
        """
        self.interrupt()
        self.config = config
        if not self.mirai_seeker is None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def restart(self):
        """
        Interrupts the engine and starts again from last checkpoint (if any).
        """
        self.interrupt()
        Thread(target=lambda: self.__main_loop__()).start()

    def __main_loop__(self):
        """
        Main optimization loop.
        """
        self.is_running = True
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.base_models = {}
        self.base_models_ids = []
        self.train_predictions_dict = {}
        self.test_predictions_dict = {}
        self.scores = {}
        self.best_score = None
        self.best_id = None

        ensemble_id = self.config.ensemble_id

        for base_layout in self.config.base_layouts:
            if self.must_interrupt:
                break
            id = base_layout.id
            self.base_models_ids.append(id)
            base_model_path = self.models_dir + id
            if os.path.exists(base_model_path):
                base_model = load(base_model_path)
            else:
                parameters, features = base_layout.gen_parameters_features(self.all_features)
                base_layout.parameters_rules(parameters)
                base_model = BaseModel(base_layout.model_class, parameters, features)
                par_dump(base_model, base_model_path)
            self.base_models[id] = base_model

            self.train_predictions_dict[id], self.test_predictions_dict[id],\
                self.scores[id] = self.base_models[id].predict(self.X_train,
                    self.y_train, self.X_test, self.config)

            if self.best_score is None or self.scores[id] > self.best_score:
                self.best_score = self.scores[id]
                self.best_id = id

        self.mirai_seeker = MiraiSeeker(self.base_models_ids, self.all_features,
            self.config)

        self.ensembler = Ensembler(self.base_models_ids, self.y_train,
            self.train_predictions_dict, self.test_predictions_dict, self.scores,
            self.config)

        if self.ensembler.optimize():
            score = self.scores[ensemble_id]
            if score > self.best_score:
                self.best_score = score
                self.best_id = ensemble_id

        if self.config.report:
            self.report()
        while not self.must_interrupt:
            for base_layout in self.config.base_layouts:
                if self.must_interrupt:
                    break
                id = base_layout.id

                if self.mirai_seeker.is_ready(id) and\
                    uniform(0, 1) < self.config.mirai_exploration_ratio:
                    parameters, features = self.mirai_seeker.gen_parameters_features(id)
                else:
                    parameters, features = base_layout.gen_parameters_features(self.all_features)
                base_layout.parameters_rules(parameters)
                base_model = BaseModel(base_layout.model_class, parameters, features)

                train_predictions, test_predictions, score = base_model.\
                    predict(self.X_train, self.y_train, self.X_test,
                        self.config)

                self.mirai_seeker.register_base_model(id, base_model, score)

                if score > self.scores[id]:
                    self.scores[id] = score
                    self.train_predictions_dict[id] = train_predictions
                    self.test_predictions_dict[id] = test_predictions
                    if score > self.best_score:
                        self.best_score = score
                        self.best_id = id

                    self.train_predictions_dict[ensemble_id],\
                        self.test_predictions_dict[ensemble_id],\
                        self.scores[ensemble_id] = self.ensembler.update()
                    if self.scores[ensemble_id] > self.best_score:
                        self.best_score = self.scores[ensemble_id]
                        self.best_id = ensemble_id
                    if self.config.report:
                        self.report()
                    par_dump(base_model, self.models_dir + id)

            if self.ensembler.optimize():
                score = self.scores[ensemble_id]
                if score > self.best_score:
                    self.best_score = score
                    self.best_id = ensemble_id
                if self.config.report:
                    self.report()

        self.is_running = False

    def request_score(self):
        """
        Queries the score of the best id on the training data.

        :rtype: float
        :returns: The score of the best model.
        """
        if len(self.scores) > 0:
            return self.scores[self.best_id]
        return None

    def request_predictions(self):
        """
        Queries the predictions of the best id for the testing
        data.

        :rtype: numpy.ndarray
        :returns: The predictions of the best model (or ensemble) for the testing
            data.
        """
        if len(self.test_predictions_dict) > 0:
            return self.test_predictions_dict[self.best_id]
        return None

    def report(self):
        """
        Prints the score for each id.
        """
        status = []
        for id in self.scores:
            status.append({
                'id': id,
                'score': self.scores[id]
            })
        print()
        print(pd.DataFrame(status).sort_values('score',
            ascending=False).reset_index(drop=True))
