from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from time import sleep
import pandas as pd
import numpy as np
import warnings

from miraiml import SearchSpace, Config, Engine

warnings.filterwarnings('ignore')

# In this example, we will use a Logistic Regression, a K-NN, a Naive Bayes Classifier and
# their ensemble on the first layer and a LightGBM wrapper on the second layer.

### FIRST LAYER ###
def logistic_regression_parameters_rules(parameters):
    if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
        parameters['penalty'] = 'l2'

search_spaces_first_layer = [
    SearchSpace(
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
    ),

    SearchSpace(model_class=KNeighborsClassifier, id='K-NN', parameters_values= {
        'n_neighbors': np.arange(1, 15),
        'weights': ['uniform', 'distance'],
        'p': np.arange(1, 5)
    }),

    SearchSpace(model_class=GaussianNB, id='Gaussian NB')
]

config_first_layer = Config(
    local_dir = 'miraiml_local_stacking_first_layer',
    problem_type = 'classification',
    search_spaces = search_spaces_first_layer,
    score_function = roc_auc_score,
    ensemble_id = 'Ensemble',
    n_ensemble_cycles = 1000
)

### SECOND LAYER ###
class LightGBM:
    def __init__(self, n_folds, max_leaves, colsample_bytree, learning_rate):
        self.n_folds = n_folds
        self.parameters = dict(
            max_leaves = max_leaves,
            colsample_bytree = colsample_bytree,
            learning_rate = learning_rate,
            boosting_type = 'gbdt',
            objective = 'binary',
            metric = 'auc',
            verbosity = -1
        )
        self.models = None

    def fit(self, X, y):
        # list to save trained models
        self.models = []

        folds = StratifiedKFold(n_splits = self.n_folds)

        for _, (index_train, index_valid) in enumerate(folds.split(X, y)):
            X_train, y_train = X.iloc[index_train], y.iloc[index_train]
            X_valid, y_valid = X.iloc[index_valid], y.iloc[index_valid]

            dtrain = lgb.Dataset(X_train, y_train)
            dvalid = lgb.Dataset(X_valid, y_valid)

            model = lgb.train(
                params = self.parameters,
                train_set = dtrain,
                num_boost_round = 1000000, # a big number. it will use early stop
                valid_sets = dvalid,
                early_stopping_rounds = 30,
                verbose_eval = False
            )

            self.models.append(model)

    def predict_proba(self, X):

        y_test = None

        # averaging predictions
        for model in self.models:
            if y_test is None:
                y_test = model.predict(X)
            else:
                y_test += model.predict(X)

        y_test /= self.n_folds

        # returning a 2-columns numpy.ndarray
        return np.array([1-y_test, y_test]).transpose()

search_spaces_second_layer = [
    SearchSpace(
        model_class = LightGBM,
        id = 'LightGBM',
        parameters_values = dict(
            n_folds = [5],
            max_leaves = [3, 7, 15, 31],
            colsample_bytree = [0.2, 0.4, 0.6, 0.8, 1],
            learning_rate = [0.1]
        )
    )
]

config_second_layer = Config(
    local_dir = 'miraiml_local_stacking_second_layer',
    problem_type = 'classification',
    search_spaces = search_spaces_second_layer,
    score_function = roc_auc_score
)

# Gathering data
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)
train_target = train_data.pop('target_class')

# Important: it's necessary to reset indexes otherwise pd.concat will create extra rows for
# indexes that do not relate to each other
for df in train_data, test_data:
    df.reset_index(drop=True, inplace=True)

engine_second_layer = Engine(config_second_layer)

# The second engine will be triggered here
def load_second_layer_data(status):
    train_data_second_layer = pd.concat([train_data, status['train_predictions']], axis=1)
    test_data_second_layer = pd.concat([test_data, status['test_predictions']], axis=1)

    engine_second_layer.load_data(
        train_data = train_data_second_layer,
        train_target = train_target,
        test_data = test_data_second_layer,
        restart = True # this will trigger the second engine
    )

engine_first_layer = Engine(config_first_layer, on_improvement=load_second_layer_data)

# Loading data on the first layer of models
engine_first_layer.load_data(train_data, train_target, test_data)

# Fire!
engine_first_layer.restart()

# Let's wait 20 seconds and then interrupt the engines
print('Training...')
sleep(60)
engine_first_layer.interrupt()
engine_second_layer.interrupt()

# Checking layers scores
status_first_layer = engine_first_layer.request_status()
status_second_layer = engine_second_layer.request_status()

# Let's see the scores
print('\nFirst layer scores:', status_first_layer['scores'])
print('\nSecond layer scores:', status_second_layer['scores'])
