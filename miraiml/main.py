from threading import Thread
import pandas as pd
import warnings
import time
import os
import gc

from miraiml.util import is_valid_filename
from miraiml.core import MiraiSeeker, Ensembler
from miraiml.core import load_base_model, dump_base_model


class SearchSpace:
    """
    This class represents the search space of hyperparameters for a base model.

    :type id: str
    :param id: The id that will be associated with the models generated within
        this search space.

    :type model_class: type
    :param model_class: Any class that represents a statistical model. It must
        implement the methods ``fit`` as well as ``predict`` for regression or
        ``predict_proba`` for classification problems.

    :type parameters_values: dict, optional, default=None
    :param parameters_values: A dictionary containing lists of values to be
        tested as parameters when instantiating objects of ``model_class`` for
        ``id``.

    :type parameters_rules: function, optional, default=lambda x: None
    :param parameters_rules: A function that constrains certain parameters because
        of the values assumed by others. It must receive a dictionary as input and
        doesn't need to return anything. Not used if ``parameters_values`` has no
        keys.

        .. warning::
             Make sure that the parameters accessed in ``parameters_rules`` exist
             in the set of parameters defined on ``parameters_values``, otherwise
             the engine will attempt to access an invalid key.

    :raises: ``NotImplementedError`` if a model class does not implement ``fit``
        or none of ``predict`` or ``predict_proba``.

    :raises: ``TypeError`` if some parameter is of a prohibited type.

    :raises: ``ValueError`` if a provided ``id`` is not allowed.

    :Example:

    ::

        >>> import numpy as np
        >>> from sklearn.linear_model import LogisticRegression
        >>> from miraiml import SearchSpace

        >>> def logistic_regression_parameters_rules(parameters):
        ...     if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
        ...         parameters['penalty'] = 'l2'

        >>> search_space = SearchSpace(
        ...     id = 'Logistic Regression',
        ...     model_class = LogisticRegression,
        ...     parameters_values = {
        ...         'penalty': ['l1', 'l2'],
        ...         'C': np.arange(0.1, 2, 0.1),
        ...         'max_iter': np.arange(50, 300),
        ...         'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        ...         'random_state': [0]
        ...     },
        ...     parameters_rules = logistic_regression_parameters_rules
        ... )

    .. warning::
        **Do not** allow ``random_state`` assume multiple values. If ``model_class``
        has a ``random_state`` parameter, force the engine to always choose the
        same value by providing a list with a single element.

        Allowing ``random_state`` to assume multiple values will confuse the engine
        because the scores will be unstable even with the same choice of
        hyperparameters and features.
    """
    def __init__(self, id, model_class, parameters_values=None,
                 parameters_rules=lambda x: None):
        self.__validate__(id, model_class, parameters_values, parameters_rules)
        self.model_class = model_class
        self.id = id
        if parameters_values is None:
            parameters_values = {}
        self.parameters_values = parameters_values
        self.parameters_rules = parameters_rules

    @staticmethod
    def __validate__(id, model_class, parameters_values, parameters_rules):
        """
        Validates the constructor parameters.
        """
        if not isinstance(id, str):
            raise TypeError('id must be a string')
        if not is_valid_filename(id):
            raise ValueError('Invalid id: {}'.format(id))
        dir_model_class = dir(model_class)
        if 'fit' not in dir_model_class:
            raise NotImplementedError('model_class must implement fit')
        if 'predict' not in dir_model_class and 'predict_proba' not in dir_model_class:
            raise NotImplementedError('model_class must implement predict or predict_proba')
        if parameters_values is not None and not isinstance(parameters_values, dict):
            raise TypeError('parameters_values must be None or a dictionary')
        if not callable(parameters_rules):
            raise TypeError('parameters_rules must be a function')


class Config:
    """
    This class defines the general behavior of the engine.

    :type local_dir: str
    :param local_dir: The name of the folder in which the engine will save its
        internal files. If the directory doesn't exist, it will be created
        automatically. ``..`` and ``/`` are not allowed to compose ``local_dir``.

    :type problem_type: str
    :param problem_type: ``'classification'`` or ``'regression'``. The problem
        type. Multi-class classification problems are not supported.

    :type search_spaces: list
    :param search_spaces: The list of :class:`miraiml.SearchSpace`
        objects to optimize. If ``search_spaces`` has length 1, the engine
        will not run ensemble cycles.

    :type score_function: function
    :param score_function: A function that receives the "truth" and the predictions
        (in this order) and returns the score. Bigger scores must mean better models.

    :type use_all_features: bool, optional, default=False
    :param use_all_features: Whether to force MiraiML to always use all features
        or not.

    :type n_folds: int, optional, default=5
    :param n_folds: The number of folds for the fitting/predicting process. The
        minimum value allowed is 2.

    :type stratified: bool, optional, default=True
    :param stratified: Whether to stratify folds on target or not. Only used if
        ``problem_type == 'classification'``.

    :type ensemble_id: str, optional, default=None
    :param ensemble_id: The id for the ensemble. If none is given, the engine will
        not ensemble base models.

    :type stagnation: int or float, optional, default=60
    :param stagnation: The amount of time (in minutes) for the engine to
        automatically interrupt itself if no improvement happens. Negative numbers
        are interpreted as "infinite".

        .. warning::
            Stagnation checks only happen after the engine finishes at least one
            optimization cycle. In other words, every base model and the ensemble
            (if set) must be scored at least once.

    :raises: ``NotImplementedError`` if a model class does not implement the proper
        method for prediction.

    :raises: ``TypeError`` if some parameter is not of its allowed type.

    :raises: ``ValueError`` if some parameter has an invalid value.

    :Example:

    ::

        >>> from sklearn.metrics import roc_auc_score
        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from miraiml import SearchSpace, Config

        >>> search_spaces = [
        ...     SearchSpace('Naive Bayes', GaussianNB),
        ...     SearchSpace('Decicion Tree', DecisionTreeClassifier)
        ... ]

        >>> config = Config(
        ...     local_dir = 'miraiml_local',
        ...     problem_type = 'classification',
        ...     score_function = roc_auc_score,
        ...     search_spaces = search_spaces,
        ...     use_all_features = False,
        ...     n_folds = 5,
        ...     stratified = True,
        ...     ensemble_id = 'Ensemble',
        ...     stagnation = -1
        ... )
    """
    def __init__(self, local_dir, problem_type, score_function, search_spaces,
                 use_all_features=False, n_folds=5, stratified=True,
                 ensemble_id=None, stagnation=60):
        self.__validate__(local_dir, problem_type, score_function, search_spaces,
                          use_all_features, n_folds, stratified, ensemble_id,
                          stagnation)
        self.local_dir = local_dir
        if self.local_dir[-1] != '/':
            self.local_dir += '/'
        self.problem_type = problem_type
        self.search_spaces = search_spaces
        self.score_function = score_function
        self.use_all_features = use_all_features
        self.n_folds = n_folds
        self.stratified = stratified
        self.ensemble_id = ensemble_id
        self.stagnation = stagnation

    @staticmethod
    def __validate__(local_dir, problem_type, score_function, search_spaces,
                     use_all_features, n_folds, stratified, ensemble_id,
                     stagnation):
        """
        Validates the constructor parameters.
        """
        if not isinstance(local_dir, str):
            raise TypeError('local_dir must be a string')

        if not is_valid_filename(local_dir):
            raise ValueError('Invalid directory name: {}'.format(local_dir))

        if not isinstance(problem_type, str):
            raise TypeError('problem_type must be a string')
        if problem_type not in ('classification', 'regression'):
            raise ValueError('Invalid problem type')

        if not callable(score_function):
            raise TypeError('score_function must be a function')

        if not isinstance(search_spaces, list):
            raise TypeError('search_spaces must be a list')
        if len(search_spaces) == 0:
            raise ValueError('No search spaces')

        ids = []
        for search_space in search_spaces:
            if not isinstance(search_space, SearchSpace):
                raise TypeError('All search spaces must be objects of ' +
                                'miraiml.SearchSpace')
            id = search_space.id
            if id in ids:
                raise ValueError('Duplicated search space id: {}'.format(id))
            ids.append(id)
            dir_model_class = dir(search_space.model_class)
            if problem_type == 'classification' and 'predict_proba' not in dir_model_class:
                raise NotImplementedError('Model class of id {} '.format(id) +
                                          'must implement predict_proba for ' +
                                          'classification problems')
            if problem_type == 'regression' and 'predict' not in dir_model_class:
                raise NotImplementedError('Model class of id {} '.format(id) +
                                          'must implement predict for regression problems')

        if not isinstance(use_all_features, bool):
            raise TypeError('use_all_features must be a boolean')

        if not isinstance(n_folds, int):
            raise TypeError('n_folds must be an integer')
        if n_folds < 2:
            raise ValueError('n_folds must be greater than 1')

        if not isinstance(stratified, bool):
            raise TypeError('stratified must be a boolean')

        if ensemble_id is not None and not isinstance(ensemble_id, str):
            raise TypeError('ensemble_id must be None or a string')
        if isinstance(ensemble_id, str) and not is_valid_filename(ensemble_id):
            raise ValueError('invalid ensemble_id')
        if ensemble_id in ids:
            raise ValueError('ensemble_id cannot have the same id of a ' +
                             'search space')
        if not isinstance(stagnation, int) and not isinstance(stagnation, float):
            raise TypeError('stagnation must be an integer or a float')


class Engine:
    """
    This class offers the controls for the engine.

    :type config: miraiml.Config
    :param config: The configurations for the behavior of the engine.

    :type on_improvement: function, optional, default=None
    :param on_improvement: A function that will be executed everytime the engine
        finds an improvement for some id. It must receive a ``status`` parameter,
        which is the return of the method :func:`request_status` (an instance of
        :class:`miraiml.Status`).

    :raises: ``TypeError`` if ``config`` is not an instance of :class:`miraiml.Config`
        or ``on_improvement`` (if provided) is not callable.

    :Example:

    ::

        >>> from sklearn.metrics import roc_auc_score
        >>> from sklearn.naive_bayes import GaussianNB
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> from miraiml import SearchSpace, Config, Engine

        >>> search_spaces = [
        ...     SearchSpace('Naive Bayes', GaussianNB),
        ...     SearchSpace('Decision Tree', DecisionTreeClassifier)
        ... ]

        >>> config = Config(
        ...     local_dir = 'miraiml_local',
        ...     problem_type = 'classification',
        ...     score_function = roc_auc_score,
        ...     search_spaces = search_spaces,
        ...     ensemble_id = 'Ensemble'
        ... )

        >>> def on_improvement(status):
        ...     print('Scores:', status.scores)

        >>> engine = Engine(config, on_improvement=on_improvement)
    """
    def __init__(self, config, on_improvement=None):
        self.__validate__(config, on_improvement)
        self.config = config
        self.on_improvement = on_improvement
        self.train_predictions_df = None
        self.test_predictions_df = None
        self.__is_running__ = False
        self.must_interrupt = False
        self.mirai_seeker = None
        self.models_dir = config.local_dir + 'models/'
        self.train_data = None
        self.ensembler = None
        self.n_cycles = 0
        self.last_improvement_timestamp = None

    @staticmethod
    def __validate__(config, on_improvement):
        """
        Validates the constructor parameters.
        """
        if not isinstance(config, Config):
            raise TypeError('miraiml.Engine\'s constructor requires an object ' +
                            'of miraiml.Config')
        if on_improvement is not None and not callable(on_improvement):
            raise TypeError('on_improvement must be None or a function')

    def is_running(self):
        """
        Tells whether the engine is running or not.

        :rtype: bool
        :returns: ``True`` if the engine is running and ``False`` otherwise.
        """
        return self.__is_running__

    def interrupt(self):
        """
        Makes the engine stop on the first opportunity.

        .. note::
            This method is **not** asynchronous. It will wait until the engine
            stops.
        """
        self.must_interrupt = True
        if self.ensembler is not None:
            self.ensembler.interrupt()
        while self.__is_running__:
            time.sleep(.1)
        self.must_interrupt = False

    def load_train_data(self, train_data, target_column, restart=False):
        """
        Interrupts the engine and loads the train dataset. All of its columns must
        be either instances of ``str`` or ``int``.

        .. warning::
            Loading new training data will **always** trigger the loss of history
            for optimization.

        :type train_data: pandas.DataFrame
        :param train_data: The training data.

        :type target_column: str or int
        :param target_column: The target column identifier.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after updating data or not.

        :raises: ``TypeError`` if ``train_data`` is not an instance of
            ``pandas.DataFrame``.

        :raises: ``ValueError`` if ``target_column`` is not a column of
            ``train_data`` or if some column name is of a prohibited type.
        """
        self.__validate_train_data__(train_data, target_column)
        self.columns_renaming_map = {}
        self.columns_renaming_unmap = {}

        for column in train_data.columns:
            column_renamed = str(column)
            self.columns_renaming_map[column] = column_renamed
            self.columns_renaming_unmap[column_renamed] = column

        self.target_column = target_column
        train_data = train_data.rename(columns=self.columns_renaming_map)

        self.interrupt()
        self.train_data = train_data.drop(columns=target_column)
        self.train_target = train_data[target_column]
        self.all_features = list(self.train_data.columns)

        if self.mirai_seeker is not None:
            self.mirai_seeker.reset()

        if restart:
            self.restart()

    @staticmethod
    def __validate_train_data__(train_data, target_column):
        """
        Validates the train data.
        """
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError('Training data must be an object of pandas.DataFrame')

        train_columns = train_data.columns

        if target_column not in train_columns:
            raise ValueError('target_column must be a column of train_data')

        for column in train_columns:
            if not isinstance(column, str) and not isinstance(column, int):
                raise ValueError('All columns names must be either str or int')

    def load_test_data(self, test_data, restart=False):
        """
        Interrupts the engine and loads the test dataset. All of its columns must
        be columns in the train data.

        The test dataset is the one for which we don't have the values for the
        target column. This method should be used to load data in production.

        .. warning::
            This method can only be called after
            :func:`miraiml.Engine.load_train_data`

        :type test_data: pandas.DataFrame, optional, default=None
        :param test_data: The testing data. Use the default value if you don't
            need to make predictions for data with unknown labels.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after loading data or not.

        :raises: ``RuntimeError`` if this method is called before loading the
            train data.

        :raises: ``ValueError`` if the column names are not consistent.
        """
        if self.train_data is None:
            raise RuntimeError('This method cannot be called before load_train_data')

        self.__validate_test_data__(test_data)
        self.test_data = test_data.rename(columns=self.columns_renaming_map)
        if restart:
            self.restart()

    def __validate_test_data__(self, test_data):
        """
        Validates the test data.
        """
        for column in self.columns_renaming_map:
            if column != self.target_column and column not in test_data.columns:
                raise ValueError(
                    'Column {} is not a column in the train data'.format(column)
                )

    def clean_test_data(self, restart=False):
        """
        Cleans the test data from the buffer.

        .. note::
            Keep in mind that if you don't intend to make predictions for
            unlabeled data, the engine will run faster with a clean test data
            buffer.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after cleaning test data or
            not.
        """
        self.interrupt()
        self.test_data = None
        if restart:
            self.restart()

    def shuffle_train_data(self, restart=False):
        """
        Interrupts the engine and shuffles the training data.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after shuffling data or not.

        :raises: ``RuntimeError`` if the engine has no data loaded.

        .. note::
            It's a good practice to shuffle the training data periodically to avoid
            overfitting on a particular folding pattern.
        """
        if self.train_data is None:
            raise RuntimeError('No data to shuffle')

        self.interrupt()

        seed = int(time.time())
        self.train_data = self.train_data.sample(frac=1, random_state=seed)
        self.train_target = self.train_target.sample(frac=1, random_state=seed)

        if restart:
            self.restart()

    def reconfigure(self, config, restart=False):
        """
        Interrupts the engine and loads a new configuration.

        .. warning::
            Reconfiguring the engine will **always** trigger the loss of history
            for optimization.

        :type config: miraiml.Config
        :param config: The configurations for the behavior of the engine.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after reconfiguring it or
            not.
        """
        self.interrupt()
        self.config = config
        if self.mirai_seeker is not None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def restart(self):
        """
        Interrupts the engine and starts again from last checkpoint (if any). It
        is also used to start the engine for the first time.

        :raises: ``RuntimeError`` if no data is loaded.
        """
        if self.train_data is None:
            raise RuntimeError('No data to train')
        self.interrupt()

        def starter():
            try:
                self.__main_loop__()
            except Exception:
                self.__is_running__ = False
                raise

        Thread(target=starter).start()

    def __improvement_trigger__(self):
        """
        Called when an improvement happens.
        """
        self.last_improvement_timestamp = time.time()
        if self.on_improvement is not None:
            self.on_improvement(self.request_status())

    def __update_best__(self, score, id):
        """
        Updates the best id of the engine.
        """
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_id = id

    def __check_stagnation__(self):
        """
        Checks whether the engine has reached stagnation or not. If so, the
        engine is interrupted.
        """
        if self.config.stagnation >= 0:
            diff_in_seconds = time.time() - self.last_improvement_timestamp
            if diff_in_seconds/60 > self.config.stagnation:
                self.interrupt()

    def __main_loop__(self):
        """
        Main optimization loop.
        """
        self.__is_running__ = True
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.base_models = {}
        self.train_predictions_df = pd.DataFrame()
        self.test_predictions_df = pd.DataFrame()
        self.scores = {}
        self.best_score = None
        self.best_id = None
        self.ensembler = None

        self.mirai_seeker = MiraiSeeker(
            self.config.search_spaces,
            self.all_features,
            self.config
        )

        self.n_cycles = 0
        self.last_improvement_timestamp = time.time()

        start = time.time()
        for search_space in self.config.search_spaces:
            if self.must_interrupt:
                break
            id = search_space.id
            base_model_path = self.models_dir + id
            base_model_class = search_space.model_class
            if os.path.exists(base_model_path):
                base_model = load_base_model(base_model_class, base_model_path)
                parameters = base_model.parameters
                parameters_values = search_space.parameters_values
                for key, value in zip(parameters.keys(), parameters.values()):
                    if key not in parameters_values:
                        warnings.warn(
                            'Parameter ' + str(key) + ', set with value ' +
                            str(value) + ', from checkpoint is not on the ' +
                            'provided search space for the id ' + str(id),
                            RuntimeWarning
                        )
                    else:
                        if value not in parameters_values[key]:
                            warnings.warn(
                                'Value ' + str(value) + ' for parameter ' + str(key) +
                                ' from checkpoint is not on the provided ' +
                                'search space for the id ' + str(id),
                                RuntimeWarning
                            )
            else:
                base_model = self.mirai_seeker.seek(search_space.id)
                dump_base_model(base_model, base_model_path)
            self.base_models[id] = base_model

            train_predictions, test_predictions, score = base_model.predict(
                self.train_data, self.train_target, self.test_data, self.config)
            self.train_predictions_df[id] = train_predictions
            self.test_predictions_df[id] = test_predictions
            self.scores[id] = score

            self.__update_best__(self.scores[id], id)

        total_cycles_duration = time.time() - start

        will_ensemble = len(self.base_models) > 1 and\
            self.config.ensemble_id is not None

        if will_ensemble:
            self.ensembler = Ensembler(
                list(self.base_models),
                self.train_target,
                self.train_predictions_df,
                self.test_predictions_df,
                self.scores,
                self.config
            )

            ensemble_id = self.config.ensemble_id

            if self.ensembler.optimize(total_cycles_duration):
                self.__update_best__(self.scores[ensemble_id], ensemble_id)

        self.__improvement_trigger__()

        self.n_cycles = 1

        while not self.must_interrupt:
            gc.collect()
            start = time.time()

            for search_space in self.config.search_spaces:
                self.__check_stagnation__()
                if self.must_interrupt:
                    break
                id = search_space.id

                base_model = self.mirai_seeker.seek(id)

                train_predictions, test_predictions, score = base_model.predict(
                    self.train_data, self.train_target,
                    self.test_data, self.config)

                self.mirai_seeker.register_base_model(id, base_model, score)

                if score > self.scores[id] or (
                            score == self.scores[id] and
                            len(base_model.features) < len(self.base_models[id].features)
                        ):
                    self.scores[id] = score
                    self.train_predictions_df[id] = train_predictions
                    self.test_predictions_df[id] = test_predictions
                    self.__update_best__(score, id)

                    if will_ensemble:
                        self.ensembler.update()
                        self.__update_best__(self.scores[ensemble_id], ensemble_id)

                    self.__improvement_trigger__()

                    dump_base_model(base_model, self.models_dir + id)
                else:
                    del train_predictions, test_predictions

            total_cycles_duration += time.time() - start
            self.n_cycles += 1

            if will_ensemble:
                if self.ensembler.optimize(total_cycles_duration/self.n_cycles):
                    self.__update_best__(self.scores[ensemble_id], ensemble_id)

                    self.__improvement_trigger__()

        self.__is_running__ = False

    def request_status(self):
        """
        Queries the current status of the engine.

        :rtype: miraiml.Status
        :returns: The current status of the engine in the form of a dictionary.
            If no score has been computed yet, returns ``None``.
        """
        if self.best_id is None:
            return None

        train_predictions = None
        if self.train_predictions_df is not None:
            train_predictions = self.train_predictions_df.copy()

        test_predictions = None
        if self.test_data is not None and self.test_predictions_df is not None:
            test_predictions = self.test_predictions_df.copy()

        ensemble_weights = None
        if self.ensembler is not None:
            ensemble_weights = self.ensembler.weights.copy()

        base_models = {}
        for id in self.base_models:
            base_model = self.base_models[id]
            base_models[id] = dict(
                model_class=base_model.model_class.__name__,
                parameters=base_model.parameters.copy()
            )

            base_models[id]['features'] = [
                self.columns_renaming_unmap[col] for col in base_model.features
            ]

        histories = None
        if self.mirai_seeker is not None:
            histories = {}
            for id in self.mirai_seeker.histories:
                histories[id] = self.mirai_seeker.histories[id].copy()

        return Status(
            best_id=self.best_id,
            scores=self.scores.copy(),
            train_predictions=train_predictions,
            test_predictions=test_predictions,
            ensemble_weights=ensemble_weights,
            base_models=base_models,
            histories=histories
        )


class Status:
    """
    Represents the current status of the engine. Objects of this class are
    not supposed to be instantiated by the user. Rather, they are returned
    by the :func:`miraiml.Engine.request_status()` method.

    The following attributes are accessible:

    * ``best_id``: the id of the best base model (or ensemble)

    * ``scores``: a dictionary containing the current score of each id

    * ``train_predictions``: a ``pandas.DataFrame`` object containing the predictions\
        for the train data for each id

    * ``test_predictions``: a ``pandas.DataFrame`` object containing the predictions\
        for the test data for each id

    * ``ensemble_weights``: a dictionary containing the ensemble weights for\
        each base model id

    * ``base_models``: a dictionary containing the characteristics of each base\
        model (accessed by its respective id)

    * ``histories``: a dictionary of ``pandas.DataFrame`` objects for each id,\
        containing the history of base models attempts and their respective scores.\
        Hyperparameters columns end with the ``'__(hyperparameter)'`` suffix and\
        features columns end with the ``'__(feature)'`` suffix. The score column\
        can be accessed with the key ``'score'``. For more information, please\
        check the :ref:`User Guide <mirai_seeker>`.

    The characteristics of each base model are represent by dictionaries, containing
    the following keys:

    * ``'model_class'``: The name of the base model's modeling class

    * ``'parameters'``: The dictionary of hyperparameters values

    * ``'features'``: The list of features used
    """
    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def build_report(self, include_features=False):
        """
        Returns the report of the current status of the engine in a formatted
        string.

        :type include_features: bool, optional, default=False
        :param include_features: Whether to include the list of features on the
            report or not (may cause some visual mess).

        :rtype: str
        :returns: The formatted report.
        """
        output = '########################\n'

        output += ('best id: {}\n'.format(self.best_id))
        output += ('best score: {}\n'.format(self.scores[self.best_id]))

        if self.ensemble_weights is not None:
            output += ('########################\n')
            output += ('ensemble weights:\n')
            weights_ = {}
            for id in self.ensemble_weights:
                weights_[self.ensemble_weights[id]] = id
            for weight in reversed(sorted(weights_)):
                id = weights_[weight]
                output += ('    {}: {}\n'.format(id, weight))

        output += ('########################\n')
        output += ('all scores:\n')
        scores_ = {}
        for id in self.scores:
            scores_[self.scores[id]] = id
        for score in reversed(sorted(scores_)):
            id = scores_[score]
            output += ('    {}: {}\n'.format(id, score))

        for id in sorted(self.base_models):
            base_model = self.base_models[id]
            features = sorted([str(feature) for feature in base_model['features']])
            output += ('########################\n')
            output += ('id: {}\n'.format(id))
            output += ('model class: {}\n'.format(base_model['model_class']))
            output += ('n features: {}\n'.format(len(features)))
            output += ('parameters:\n')
            parameters = base_model['parameters']
            for parameter in sorted(parameters):
                value = parameters[parameter]
                output += ('    {}: {}\n'.format(parameter, value))
            if include_features:
                output += ('features: {}\n'.format(', '.join(features)))

        return output
