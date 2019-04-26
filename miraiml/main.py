from threading import Thread
import pandas as pd
import time
import os

from .util import load, dump, is_valid_filename
from .core import MiraiSeeker, Ensembler


class HyperSearchSpace:
    """
    This class represents the search space of hyperparameters for a base model.

    :type model_class: type
    :param model_class: Any class that represents a statistical model. It must
        implement the methods ``fit`` as well as ``predict`` for regression or
        ``predict_proba`` for classification problems.

    :type id: str
    :param id: The id that will be associated with the models generated within
        this search space.

    :type parameters_values: dict, optional, default=None
    :param parameters_values: A dictionary containing lists of values to be
        tested as parameters when instantiating objects of ``model_class``.

    :type parameters_rules: function, optional, default=lambda x: None
    :param parameters_rules: A function that constrains certain parameters because
        of the values assumed by others. It must receive a dictionary as input and
        doesn't need to return anything. Not used if ``parameters_values`` has no
        keys.

        .. warning::
             Make sure that the parameters accessed in ``parameters_rules`` exist
             in the set of parameters defined on ``parameters_values``, otherwise
             the engine will attempt to access an invalid key.

    :raises: ``NotImplementedError``, ``TypeError``, ``ValueError``

    :Example:

    ::

        from sklearn.linear_model import LogisticRegression
        from miraiml import HyperSearchSpace

        def logistic_regression_parameters_rules(parameters):
            if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
                parameters['penalty'] = 'l2'

        hyper_search_space = HyperSearchSpace(
            model_class = LogisticRegression,
            id = 'Logistic Regression',
            parameters_values = {
                'penalty': ['l1', 'l2'],
                'C': np.arange(0.1, 2, 0.1),
                'max_iter': np.arange(50, 300),
                'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'random_state': [0]
            },
            parameters_rules = logistic_regression_parameters_rules
        )

    .. warning::
        **Do not** allow ``random_state`` assume multiple values. If ``model_class``
        has a ``random_state`` parameter, force the engine to always choose the
        same value by providing a list with a single element.

        Allowing ``random_state`` to assume multiple values will confuse the engine
        because the scores will be unstable even with the same choice of
        hyperparameters and features.
    """
    def __init__(self, model_class, id, parameters_values=None,
            parameters_rules=lambda x: None):
        self.__validate__(model_class, id, parameters_values, parameters_rules)
        self.model_class = model_class
        self.id = id
        if parameters_values is None:
            parameters_values = {}
        self.parameters_values = parameters_values
        self.parameters_rules = parameters_rules

    @classmethod
    def __validate__(cls, model_class, id, parameters_values, parameters_rules):
        """
        Validates the constructor parameters.
        """
        dir_model_class = dir(model_class)
        if 'fit' not in dir_model_class:
            raise NotImplementedError('model_class must implement fit')
        if not isinstance(id, str):
            raise TypeError('id must be a string')
        if not is_valid_filename(id):
            raise ValueError('Invalid id: {}'.format(id))
        if parameters_values is not None and not isinstance(parameters_values, dict):
            raise TypeError('parameters_values must be None or a dictionary')
        if not callable(parameters_rules):
            raise TypeError('parameters_rules must be a function')

class Config:
    """
    This class defines the general behavior of the engine.

    :type local_dir: str
    :param local_dir: The path for the engine to save its internal files. If the
        directory doesn't exist, it will be created automatically.

    :type problem_type: str
    :param problem_type: ``'classification'`` or ``'regression'``. The problem
        type. Multi-class classification problems are not supported.

    :type hyper_search_spaces: list
    :param hyper_search_spaces: The list of :class:`miraiml.HyperSearchSpace`
        objects to optimize. If ``hyper_search_spaces`` has length 1, the engine
        will not run ensemble cycles.

    :type score_function: function
    :param score_function: A function that receives the "truth" and the predictions
        (in this order) and returns the score. Bigger scores must mean better models.

    :type n_folds: int, optional, default=5
    :param n_folds: The number of folds for the fitting/predicting process.

    :type stratified: bool, optional, default=True
    :param stratified: Whether to stratify folds on target or not. Only used if
        ``problem_type == 'classification'``.

    :type ensemble_id: str, optional, default=None
    :param ensemble_id: The id for the ensemble. If none is given, the engine will
        not ensemble base models.

    :raises: ``NotImplementedError``, ``TypeError``, ``ValueError``

    :Example:

    ::

        from sklearn.metrics import roc_auc_score
        from miraiml import Config

        config = Config(
            local_dir = 'miraiml_local',
            problem_type = 'classification',
            hyper_search_spaces = hyper_search_spaces,
            score_function = roc_auc_score,
            n_folds = 5,
            stratified = True,
            ensemble_id = 'Ensemble'
        )
    """
    def __init__(self, local_dir, problem_type, hyper_search_spaces, score_function,
            n_folds=5, stratified=True, ensemble_id=None):
        self.__validate__(local_dir, problem_type, hyper_search_spaces, score_function,
                          n_folds, stratified, ensemble_id)
        self.local_dir = local_dir
        if self.local_dir[-1] != '/':
            self.local_dir += '/'
        self.problem_type = problem_type
        self.hyper_search_spaces = hyper_search_spaces
        self.score_function = score_function
        self.n_folds = n_folds
        self.stratified = stratified
        self.ensemble_id = ensemble_id

    @classmethod
    def __validate__(cls, local_dir, problem_type, hyper_search_spaces,
            score_function, n_folds, stratified, ensemble_id):
        """
        Validates the constructor parameters.
        """
        if not isinstance(local_dir, str):
            raise TypeError('local_dir must be a string')

        for dir_name in local_dir.split('/'):
            if not is_valid_filename(dir_name):
                raise ValueError('Invalid directory name: {}'.format(dir_name))

        if not isinstance(problem_type, str):
            raise TypeError('problem_type must be a string')
        if problem_type not in ('classification', 'regression'):
            raise ValueError('Invalid problem type')

        if not isinstance(hyper_search_spaces, list):
            raise TypeError('hyper_search_spaces must be a list')
        if len(hyper_search_spaces) == 0:
            raise ValueError('No search spaces')

        ids = []
        for hyper_search_space in hyper_search_spaces:
            if not isinstance(hyper_search_space, HyperSearchSpace):
                raise TypeError('All hyper search spaces must be objects of '+\
                    'miraiml.HyperSearchSpace')
            id = hyper_search_space.id
            if id in ids:
                raise ValueError('Duplicated search space id: {}'.format(id))
            ids.append(id)
            dir_model_class = dir(hyper_search_space.model_class)
            if problem_type == 'classification' and 'predict_proba' not in dir_model_class:
                raise NotImplementedError('Model class of id {} '.format(id)+\
                    'must implement predict_proba for classification problems')
            if problem_type == 'regression' and 'predict' not in dir_model_class:
                raise NotImplementedError('Model class of id {} '.format(id)+\
                    'must implement predict for regression problems')

        if not callable(score_function):
            raise TypeError('score_function must be a function')

        if not isinstance(n_folds, int):
            raise TypeError('n_folds must be an integer')
        if n_folds < 2:
            raise ValueError('n_folds greater than 1')

        if not isinstance(stratified, bool):
            raise TypeError('stratified must be a boolean')

        if ensemble_id is not None and not isinstance(ensemble_id, str):
            raise TypeError('ensemble_id must be None or a string')
        if isinstance(ensemble_id, str) and not is_valid_filename(ensemble_id):
            raise ValueError('invalid ensemble_id')
        if ensemble_id in ids:
            raise ValueError('ensemble_id cannot have the same id of a hyper '+\
                'search space')


class Engine:
    """
    This class offers the controls for the engine.

    :type config: miraiml.Config
    :param config: The configurations for the behavior of the engine.

    :type on_improvement: function, optional, default=None
    :param on_improvement: A function that will be executed everytime the engine
        finds an improvement for some id. It must receive a ``status`` parameter,
        which is the return of the method :func:`request_status`.

    :raises: ``TypeError``

    :Example:

    ::

        from miraiml import Engine

        def on_improvement(status):
            print('Scores:', status['scores'])

        engine = Engine(config, on_improvement=on_improvement)
    """
    def __init__(self, config, on_improvement=None):
        self.__validate__(config, on_improvement)
        self.config = config
        self.on_improvement = on_improvement
        self.__is_running__ = False
        self.must_interrupt = False
        self.mirai_seeker = None
        self.models_dir = config.local_dir + 'models/'
        self.train_data = None
        self.ensembler = None

    @classmethod
    def __validate__(cls, config, on_improvement):
        """
        Validates the constructor parameters.
        """
        if not isinstance(config, Config):
            raise TypeError('miraiml.Engine\'s constructor requires an object'+\
                ' of miraiml.Config')
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
            This method is **not** asynchronous. It will wait for the engine to
            stop.
        """
        self.must_interrupt = True
        if self.ensembler is not None:
            self.ensembler.interrupt()
        while self.__is_running__:
            time.sleep(.1)
        self.must_interrupt = False

    def load_data(self, train_data, target_column, test_data=None, restart=False):
        """
        Interrupts the engine and loads a new pair of train/test datasets.

        :type train_data: pandas.DataFrame
        :param train_data: The training data.

        :type target_column: object
        :param target_column: The target column identifier.

        :type test_data: pandas.DataFrame, optional, default=None
        :param test_data: The testing data. Use the default value if you don't
            need to make predictions for data with unknown labels.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after updating data or not.

        :raises: ``TypeError``, ``ValueError``
        """
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError('Training data must be an object of pandas.DataFrame')

        if test_data is not None and not isinstance(test_data, pd.DataFrame):
            raise TypeError('Testing data must be None or an object of pandas.DataFrame')

        if target_column not in train_data.columns:
            raise ValueError('target_column must be a column of train_data')

        self.interrupt()
        self.train_data = train_data
        self.train_target = self.train_data.pop(target_column)
        self.all_features = list(train_data.columns)
        self.test_data = test_data
        if self.mirai_seeker is not None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def shuffle_train_data(self, restart=False):
        """
        Interrupts the engine and shuffles the training data.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after shuffling data or not.

        :raises: ``RuntimeError``

        .. note::
            It's a good practice to shuffle the training data periodically to avoid
            overfitting on a certain folding pattern.
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
        Interrupts the engine and starts again from last checkpoint (if any).

        :raises: ``RuntimeError``, ``KeyError``
        """
        if self.train_data is None:
            raise RuntimeError('No data to train')
        self.interrupt()

        def starter():
            try:
                self.__main_loop__()
            except:
                self.__is_running__ = False
                raise

        Thread(target=starter).start()

    def __improvement_trigger__(self):
        """
        Called when an improvement happens.
        """
        if self.on_improvement is not None:
            self.on_improvement(self.request_status())

    def __update_best__(self, score, id):
        """
        Updates the best id of the engine.
        """
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_id = id

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

        self.mirai_seeker = MiraiSeeker(
            self.config.hyper_search_spaces,
            self.all_features,
            self.config
        )

        start = time.time()
        for hyper_search_space in self.config.hyper_search_spaces:
            if self.must_interrupt:
                break
            id = hyper_search_space.id
            base_model_path = self.models_dir + id
            if os.path.exists(base_model_path):
                base_model = load(base_model_path)
            else:
                base_model = self.mirai_seeker.seek(hyper_search_space.id)
                dump(base_model, base_model_path)
            self.base_models[id] = base_model

            train_predictions, test_predictions, score = base_model.predict(
                self.train_data, self.train_target, self.test_data, self.config)
            self.train_predictions_df[id] = train_predictions
            self.test_predictions_df[id] = test_predictions
            self.scores[id] = score

            self.__update_best__(self.scores[id], id)

        total_cycles_duration = time.time() - start
        n_cycles = 1

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

        while not self.must_interrupt:

            start = time.time()
            for hyper_search_space in self.config.hyper_search_spaces:
                if self.must_interrupt:
                    break
                id = hyper_search_space.id

                base_model = self.mirai_seeker.seek(id)

                train_predictions, test_predictions, score = base_model.predict(
                    self.train_data, self.train_target,
                    self.test_data, self.config)

                self.mirai_seeker.register_base_model(id, base_model, score)

                if score > self.scores[id] or (score == self.scores[id] and\
                        len(base_model.features) < len(self.base_models[id].features)):
                    self.scores[id] = score
                    self.train_predictions_df[id] = train_predictions
                    self.test_predictions_df[id] = test_predictions
                    self.__update_best__(score, id)

                    if will_ensemble:
                        self.ensembler.update()
                        self.__update_best__(self.scores[ensemble_id], ensemble_id)

                    self.__improvement_trigger__()

                    dump(base_model, self.models_dir + id)

            total_cycles_duration += time.time() - start
            n_cycles += 1

            if will_ensemble:
                if self.ensembler.optimize(total_cycles_duration/n_cycles):
                    self.__update_best__(self.scores[ensemble_id], ensemble_id)

                    self.__improvement_trigger__()

        self.__is_running__ = False

    def request_status(self):
        """
        Queries the current status of the engine.

        :rtype: dict or None
        :returns: The current status of the engine in the form of a dictionary.
            If no score has been computed yet, returns ``None``. The available
            keys and their respective values on the status dictionary are:

            * ``'best_id'``: The current best id

            * ``'scores'``: A dictionary containing the score of each id

            * ``'predictions'``: A ``pandas.DataFrame`` object containing the\
                predictions from each id for the testing dataset. If no testing\
                dataset was provided, the value associated with this key is\
                ``None``

            * ``'ensemble_weights'``: A dictionary containing the ensemble weights\
                for each base model. If no ensembling cycle has been executed,\
                the value associated with this key is ``None``

            * ``'base_models'``: A dictionary containing the current description\
                of each base model, which can be accessed by their ids

            The dictionary associated with the ``'base_models'`` key contains the
            following keys and respective values:

            * ``'model_class'``: The name of the base model's class

            * ``'parameters'``: The dictionary of hyperparameters

            * ``'features'``: The list of features
        """
        if self.best_id is None:
            return None

        predictions = None
        if self.test_data is not None:
            predictions = self.test_predictions_df.copy()

        ensemble_weights = None
        if self.ensembler is not None:
            ensemble_weights = self.ensembler.weights.copy()

        base_models = {}
        for id in self.base_models:
            base_model = self.base_models[id]
            base_models[id] = dict(
                model_class = base_model.model_class.__name__,
                parameters = base_model.parameters.copy(),
                features = base_model.features.copy()
            )

        return dict(
            best_id = self.best_id,
            scores = self.scores.copy(),
            predictions = predictions,
            ensemble_weights = ensemble_weights,
            base_models = base_models
        )
