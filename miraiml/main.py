from threading import Thread
import pandas as pd
import time
import os

from .util import load, dump, is_valid_filename
from .core import MiraiSeeker, Ensembler

class SearchSpace:
    """
    This class represents the search hyperspace for a base statistical model. As
    an analogy, it represents all possible sets of clothes that someone can use.

    :type model_class: type
    :param model_class: Any class that represents a statistical model. It must
        implement the methods ``fit`` as well as ``predict`` for regression or
        ``predict_proba`` for classification problems.

    :type id: str
    :param id: The id that will be associated with the models generated within
        this search space.

    :type parameters_values: dict, optional, default={ }
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

    :Example:

    ::

        from sklearn.linear_model import LogisticRegression
        from miraiml import SearchSpace

        def logistic_regression_parameters_rules(parameters):
            if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
                parameters['penalty'] = 'l2'

        search_space = SearchSpace(
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
        has a ``random_state`` parameter, force ``SearchSpace`` to always choose
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

class Config:
    """
    This class defines the general behavior of the engine.

    :type local_dir: str
    :param local_dir: The path for the engine to save its internal files. If the
        directory doesn't exist, it will be created automatically.

    :type problem_type: str
    :param problem_type: ``'classification'`` or ``'regression'``. The problem
        type. Multi-class classification problems are not supported.

    :type search_spaces: list
    :param search_spaces: The list of :class:`miraiml.SearchSpace` objects to optimize.
        If ``search_spaces`` has length 1, the engine will not run ensemble cycles.

    :type score_function: function
    :param score_function: A function that receives the "truth" and the predictions
        (in this order) and returns the score. Bigger scores must mean better models.

    :type n_folds: int, optional, default=5
    :param n_folds: The number of folds for cross-validations.

    :type stratified: bool, optional, default=True
    :param stratified: Whether to stratify folds on target or not. Only used if
        ``problem_type == 'classification'``.

    :type random_exploration_ratio: float, optional, default=0.5
    :param random_exploration_ratio: The proportion of attempts in which the engine
        will explore the search space by blindly random attempts. Must be a number
        in the interval [0, 1).

    :type ensemble_id: str, optional, default=None
    :param ensemble_id: The id for the ensemble. If none is given, the engine will
        not ensemble base models.

    :type n_ensemble_cycles: int, optional, default=None
    :param n_ensemble_cycles: The number of times that the engine will attempt to
        improve the ensemble weights in each loop after optimizing all base models.
        If none or a value that's less than 1 is given, the engine will not ensemble
        base models.

    :Example:

    ::

        from sklearn.metrics import roc_auc_score
        from miraiml import Config

        config = Config(
            local_dir = 'miraiml_local',
            problem_type = 'classification',
            search_spaces = search_spaces,
            score_function = roc_auc_score,
            n_folds = 5,
            stratified = True,
            random_exploration_ratio = 0.5,
            ensemble_id = 'Ensemble',
            n_ensemble_cycles = 1000
        )
    """
    def __init__(self, local_dir, problem_type, search_spaces, score_function,
            n_folds=5, stratified=True, random_exploration_ratio=0.5,
            ensemble_id=None, n_ensemble_cycles=None):
        self.__validate__(local_dir, problem_type, search_spaces, score_function,
            n_folds, stratified, random_exploration_ratio, ensemble_id,
            n_ensemble_cycles)
        self.local_dir = local_dir
        if self.local_dir[-1] != '/':
            self.local_dir += '/'
        self.problem_type = problem_type
        self.search_spaces = search_spaces
        self.score_function = score_function
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_exploration_ratio = random_exploration_ratio
        self.ensemble_id = ensemble_id
        self.n_ensemble_cycles = n_ensemble_cycles

    def __validate__(self, local_dir, problem_type, search_spaces, score_function,
            n_folds, stratified, random_exploration_ratio, ensemble_id,
            n_ensemble_cycles):
        pass


class Engine:
    """
    This class offers the controls for the engine.

    :type config: miraiml.Config
    :param config: The configurations for the behavior of the engine.

    :raises: ``TypeError``

    :Example:

    ::

        from miraiml import Engine

        engine = Engine(config)
    """
    def __init__(self, config):
        self.__validate__(config)
        self.config = config
        self.__is_running__ = False
        self.must_interrupt = False
        self.mirai_seeker = None
        self.models_dir = config.local_dir + 'models/'
        self.X_train = None
        self.ensembler = None

    def __validate__(self, config):
        if type(config) != Config:
            raise TypeError('miraiml.Engine\'s constructor requires an object'+\
                ' of miraiml.Config')

    def is_running(self):
        """
        Tells whether the engine is running or not.

        :rtype: bool
        :returns: ``True`` if the engine is running and ``False`` otherwise.
        """
        return self.__is_running__

    def interrupt(self):
        """
        Sets an internal flag to make the engine stop on the first opportunity.
        """
        self.must_interrupt = True
        if not self.ensembler is None:
            self.ensembler.interrupt()
        while self.__is_running__:
            time.sleep(.1)
        self.must_interrupt = False

    def load_data(self, train_data, test_data, target, restart=False):
        """
        Interrupts the engine and loads a new pair of train/test datasets.

        :type train_data: pandas.DataFrame
        :param train_data: The training data.

        :type test_data: pandas.DataFrame
        :param test_data: The testing data.

        :type target: str
        :param target: The label of the target column.

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after updating data or not.
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

        :type restart: bool, optional, default=False
        :param restart: Whether to restart the engine after shuffling data or not.

        :raises: ``RuntimeError``

        .. note::
            It's a good practice to shuffle the training data periodically to avoid
            overfitting on a certain folding pattern.
        """
        if self.X_train is None:
            raise RuntimeError('No data to shuffle.')

        self.interrupt()
        if not self.X_train is None:
            seed = int(time.time())
            self.X_train = self.X_train.sample(frac=1, random_state=seed)
            self.y_train = self.y_train.sample(frac=1, random_state=seed)
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
        if not self.mirai_seeker is None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def restart(self):
        """
        Interrupts the engine and starts again from last checkpoint (if any).

        :raises: ``RuntimeError``, ``KeyError``
        """
        if self.X_train is None:
            raise RuntimeError('No data to train.')
        self.interrupt()

        def starter():
            try:
                self.__main_loop__()
            except:
                print('Stopping the engine.')
                self.__is_running__ = False
                raise

        Thread(target=lambda: starter()).start()

    def __main_loop__(self):
        """
        Main optimization loop.
        """
        self.__is_running__ = True
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.base_models = {}
        self.train_predictions_dict = {}
        self.test_predictions_dict = {}
        self.scores = {}
        self.best_score = None
        self.best_id = None

        self.mirai_seeker = MiraiSeeker(
            self.config.search_spaces,
            self.all_features,
            self.config
        )

        for search_space in self.config.search_spaces:
            if self.must_interrupt:
                break
            id = search_space.id
            base_model_path = self.models_dir + id
            if os.path.exists(base_model_path):
                base_model = load(base_model_path)
            else:
                base_model = self.mirai_seeker.seek(search_space.id)
                dump(base_model, base_model_path)
            self.base_models[id] = base_model

            self.train_predictions_dict[id], self.test_predictions_dict[id],\
                self.scores[id] = base_model.predict(self.X_train, self.y_train,
                    self.X_test, self.config)

            if self.best_score is None or self.scores[id] > self.best_score:
                self.best_score = self.scores[id]
                self.best_id = id

        base_models_ids = list(self.base_models)

        will_ensemble = len(base_models_ids) > 1\
            and not self.config.ensemble_id is None\
            and not self.config.n_ensemble_cycles is None\
            and self.config.n_ensemble_cycles > 0

        if will_ensemble:
            self.ensembler = Ensembler(
                base_models_ids,
                self.y_train,
                self.train_predictions_dict,
                self.test_predictions_dict,
                self.scores,
                self.config
            )

            ensemble_id = self.config.ensemble_id

            if self.ensembler.optimize():
                score = self.scores[ensemble_id]
                if score > self.best_score:
                    self.best_score = score
                    self.best_id = ensemble_id

        while not self.must_interrupt:
            for search_space in self.config.search_spaces:
                if self.must_interrupt:
                    break
                id = search_space.id

                base_model = self.mirai_seeker.seek(id)

                train_predictions, test_predictions, score = base_model.\
                    predict(self.X_train, self.y_train, self.X_test,
                        self.config)

                self.mirai_seeker.register_base_model(id, base_model, score)

                if score > self.scores[id] or (score == self.scores[id] and\
                        len(base_model.features) < len(self.base_models[id].features)):
                    self.scores[id] = score
                    self.train_predictions_dict[id] = train_predictions
                    self.test_predictions_dict[id] = test_predictions
                    if score > self.best_score:
                        self.best_score = score
                        self.best_id = id

                    if will_ensemble:
                        self.train_predictions_dict[ensemble_id],\
                            self.test_predictions_dict[ensemble_id],\
                            self.scores[ensemble_id] = self.ensembler.update()
                        if self.scores[ensemble_id] > self.best_score:
                            self.best_score = self.scores[ensemble_id]
                            self.best_id = ensemble_id

                    dump(base_model, self.models_dir + id)

            if will_ensemble:
                if self.ensembler.optimize():
                    score = self.scores[ensemble_id]
                    if score > self.best_score:
                        self.best_score = score
                        self.best_id = ensemble_id

        self.__is_running__ = False

    def request_score(self):
        """
        Queries the score of the best id on the training data.

        :rtype: float or None
        :returns: The score of the best model. If no score has been computed yet,
            returns ``None``.
        """
        if len(self.scores) > 0:
            return self.scores[self.best_id]
        return None

    def request_predictions(self):
        """
        Queries the predictions of the best id for the testing
        data.

        :rtype: numpy.ndarray or None
        :returns: The predictions of the best model (or ensemble) for the testing
            data. If no predictions has been computed yet, returns ``None``.
        """
        if len(self.test_predictions_dict) > 0:
            return self.test_predictions_dict[self.best_id]
        return None

    def request_scores(self):
        """
        Returns the score for each id in a ``pandas.DataFrame``.

        :rtype: pandas.DataFrame
        :returns: The score for each id in a ``pandas.DataFrame`` with the columns
            ``'id'`` and ``'score'``, in a descending order (sorted by ``'score'``).
        """
        status = []
        for id in self.scores:
            status.append({'id': id, 'score': self.scores[id]})

        return pd.DataFrame(status).\
            sort_values('score', ascending=False).\
            reset_index(drop=True)
