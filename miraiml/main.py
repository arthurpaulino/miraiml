from threading import Thread
from time import sleep, time
import pandas as pd
import os

from .core import MiraiSeeker, Ensembler
from .util import load, par_dump

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
    :param parameters_values: Optional, ``default={}``. A dictionary containing
        a list of values to be tested as parameters to instantiate objects of
        ``model_class``.

    :type parameters_rules: function
    :param parameters_rules: Optional, ``default=lambda x: None``. A function that
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
        If ``base_layouts`` has length 1, the engine will not run ensemble cycles.
    :type base_layouts: list

    :param score_function: A function that receives the "truth" and the predictions
        (in this order) and returns the score. Bigger scores must mean better models.
    :type score_function: function

    :param n_folds: Optional, ``default=5``. The number of folds for cross-validations.
    :type n_folds: int

    :param stratified: Optional, ``default=True``. Whether to stratify folds on
        target or not. Only used if ``problem_type == 'classification'``.
    :type stratified: bool

    :param random_exploration_ratio: Optional, ``default=0.5``. The proportion of
        attempts in which the engine will explore the search space by blindly random
        attempts. Must be a number in the interval [0, 1).
    :type random_exploration_ratio: float

    :param ensemble_id: Optional, ``default=None``. The id for the ensemble. If none
        is given, the engine will not ensemble base models.
    :type ensemble_id: str

    :param n_ensemble_cycles: Optional, ``default=None``. The number of times that
        the engine will attempt to improve the ensemble weights in each loop after
        optimizing all base models. If none is given, the engine will not ensemble
        base models.
    :type n_ensemble_cycles: int

    :Example:

    ::

        from sklearn.metrics import roc_auc_score
        from miraiml import Config
        config = Config(
            local_dir = 'miraiml_local',
            problem_type = 'classification',
            base_layouts = base_layouts,
            score_function = roc_auc_score,
            n_folds = 5,
            stratified = True,
            random_exploration_ratio = 0.5,
            ensemble_id = 'Ensemble',
            n_ensemble_cycles = 1000
        )
    """
    def __init__(self, local_dir, problem_type, base_layouts, score_function,
            n_folds=5, stratified=True, random_exploration_ratio=0.5,
            ensemble_id=None, n_ensemble_cycles=None):
        self.local_dir = local_dir
        if self.local_dir[-1] != '/':
            self.local_dir += '/'
        self.problem_type = problem_type
        self.base_layouts = base_layouts
        self.score_function = score_function
        self.n_folds = n_folds
        self.stratified = stratified
        self.random_exploration_ratio = random_exploration_ratio
        self.ensemble_id = ensemble_id
        self.n_ensemble_cycles = n_ensemble_cycles

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
        self.__is_running__ = False
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
        return self.__is_running__

    def interrupt(self):
        """
        Sets an internal flag to make the engine stop on the first opportunity.
        """
        self.must_interrupt = True
        if not self.ensembler is None:
            self.ensembler.interrupt()
        while self.__is_running__:
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
        self.__is_running__ = True
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        self.base_models = {}
        self.train_predictions_dict = {}
        self.test_predictions_dict = {}
        self.scores = {}
        self.best_score = None
        self.best_id = None

        ensemble_id = self.config.ensemble_id

        self.mirai_seeker = MiraiSeeker(self.config.base_layouts, self.all_features,
            self.config)

        for base_layout in self.config.base_layouts:
            if self.must_interrupt:
                break
            id = base_layout.id
            base_model_path = self.models_dir + id
            if os.path.exists(base_model_path):
                base_model = load(base_model_path)
            else:
                base_model = self.mirai_seeker.seek(base_layout.id)
                par_dump(base_model, base_model_path)
            self.base_models[id] = base_model

        will_ensemble = len(self.base_models) > 1\
            and not self.config.ensemble_id is None\
            and not self.config.n_ensemble_cycles is None

        if will_ensemble:
            self.ensembler = Ensembler(base_models_ids, self.y_train,
                self.train_predictions_dict, self.test_predictions_dict,
                self.scores, self.config)

        while not self.must_interrupt:
            for base_layout in self.config.base_layouts:
                if self.must_interrupt:
                    break
                id = base_layout.id

                base_model = self.mirai_seeker.seek(id)

                train_predictions, test_predictions, score = base_model.\
                    predict(self.X_train, self.y_train, self.X_test,
                        self.config)

                self.mirai_seeker.register_base_model(id, base_model, score)

                if id not in self.scores or score > self.scores[id]:
                    self.scores[id] = score
                    self.train_predictions_dict[id] = train_predictions
                    self.test_predictions_dict[id] = test_predictions
                    if self.best_score is None or score > self.best_score:
                        self.best_score = score
                        self.best_id = id

                    if will_ensemble:
                        self.train_predictions_dict[ensemble_id],\
                            self.test_predictions_dict[ensemble_id],\
                            self.scores[ensemble_id] = self.ensembler.update()
                        if self.scores[ensemble_id] > self.best_score:
                            self.best_score = self.scores[ensemble_id]
                            self.best_id = ensemble_id

                    par_dump(base_model, self.models_dir + id)

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

        :rtype: float
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

        :rtype: numpy.ndarray
        :returns: The predictions of the best model (or ensemble) for the testing
            data. If no predictions has been computed yet, returns ``None``.
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
            status.append({'id': id, 'score': self.scores[id]})

        print(
            pd.DataFrame(status).\
                sort_values('score', ascending=False).\
                reset_index(drop=True).\
                to_string()
        )