from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

from core import MiraiLayout, MiraiConfig, MiraiML

mirai_layouts = [
    MiraiLayout(RandomForestClassifier, 'rf', {
        'n_estimators': np.arange(5, 30),
        'max_depth': np.arange(2, 20),
        'min_samples_split': np.arange(0.1, 1.1, 0.1),
        'min_weight_fraction_leaf': np.arange(0, 0.6, 0.1),
        'random_state': [0]
    }),

    MiraiLayout(ExtraTreesClassifier, 'et', {
        'n_estimators': np.arange(5, 30),
        'max_depth': np.arange(2, 20),
        'min_samples_split': np.arange(0.1, 1.1, 0.1),
        'min_weight_fraction_leaf': np.arange(0, 0.6, 0.1),
        'random_state': [0]
    }),

    MiraiLayout(GradientBoostingClassifier, 'gb', {
        'n_estimators': np.arange(10, 130),
        'learning_rate': np.arange(0.05, 0.15, 0.01),
        'subsample': np.arange(0.5, 1, 0.01),
        'max_depth': np.arange(2, 20),
        'min_weight_fraction_leaf': np.arange(0, 0.6, 0.1),
        'random_state': [0]
    }),

    MiraiLayout(LogisticRegression, 'logr', {
        #'penalty': ['l1', 'l2'],
        'C': np.arange(0.1, 2, 0.1),
        'max_iter': np.arange(50, 300),
        #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'random_state': [0]
    }),

    MiraiLayout(GaussianNB, 'gaus'),

    MiraiLayout(KNeighborsClassifier, 'knn', {
        'n_neighbors': np.arange(1, 15),
        'p': np.arange(1, 5)
    })
]

config = MiraiConfig(dict(
    n_folds=5,
    problem_type='classification',
    stratified=True,
    score_function=roc_auc_score,
    mirai_layouts=mirai_layouts,
    ensemble_id='ens',
    n_ensemble_cycles=40,
    report=False
))

mirai_ml = MiraiML(config)

data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=0)

mirai_ml.update_data(train_data, test_data, 'target_class')
mirai_ml.restart()
