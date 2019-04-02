from sklearn.model_selection import StratifiedKFold, KFold
from random import triangular, choices, random, sample
from threading import Thread
from time import sleep
from math import ceil
import pandas as pd
import numpy as np
import pickle
import os

LOCAL_DIR = 'mirai_ml_local/'
MODELS_DIR = 'models/'

def load(path):
    return pickle.load(open(path, 'rb'))

def dump(obj, path):
    while True:
        try:
            pickle.dump(obj, open(path, 'wb'))
            return
        except:
            sleep(.1)

def par_dump(obj, path):
    Thread(target=lambda: dump(obj, path)).start()

def sample_random_len(lst):
    return sample(lst, max(1, ceil(random()*len(lst))))

class MiraiModel:
    def __init__(self, model_class, parameters, features):
        self.model_class = model_class
        self.parameters = parameters
        self.features = features

    def predict(self, X_train, y_train, X_test, config):
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

def gen_weights(scores, ids):
    weights = {}
    min_score, max_score = np.inf, -np.inf
    for id in ids:
        score = scores[id]
        min_score = min(min_score, score)
        max_score = max(max_score, score)
    diff_score = max_score - min_score
    for id in ids:
        weights[id] = triangular(0, 1, (scores[id]-min_score)/diff_score)
    return weights

def ensemble(train_predictions_dict, test_predictions_dict, weights, y_train,
        config):
    ids = sorted(weights)
    id = ids[0]
    train_predictions = weights[id]*train_predictions_dict[id]
    test_predictions = weights[id]*test_predictions_dict[id]
    weights_sum = weights[id]
    for id in ids[1:]:
        train_predictions += weights[id]*train_predictions_dict[id]
        test_predictions += weights[id]*test_predictions_dict[id]
        weights_sum += weights[id]
    train_predictions /= weights_sum
    test_predictions /= weights_sum
    return (train_predictions, test_predictions, config.score_function(y_train,
        train_predictions))

class MiraiSeeker:
    # Implements a smart way of seeking for parameters and feature sets.
    def __init__(self, ids, all_features):
        self.ids = ids
        self.all_features = all_features
        self.history_path = LOCAL_DIR + 'history'

        if os.path.exists(self.history_path):
            self.history = load(self.history_path)
        else:
            self.reset()

    def reset(self):
        self.history = {}
        for id in self.ids:
            self.history[id] = pd.DataFrame()
        par_dump(self.history, self.history_path)

    def register_mirai_model(self, id, mirai_model, score):
        event = {
            'model_class': mirai_model.model_class,
            'score':score
        }
        for parameter in mirai_model.parameters:
            event[parameter+'(parameter)'] = mirai_model.parameters[parameter]
        for feature in self.all_features:
            event[feature+'(feature)'] = feature in mirai_model.features

        self.history[id] = pd.concat([self.history[id],
            pd.DataFrame([event])]).drop_duplicates()
        par_dump(self.history, self.history_path)

    def is_ready(self, id):
        return self.history[id].shape[0] > 1

    def gen_mirai_model(self, id):
        # The magic happens here. For each parameter and feature, its value
        # (True or False for features) is chosen stochastically depending on the
        # mean score of the validations in which the value was chosen before.
        # Better parameters and features have higher chances of being chosen.
        history = self.history[id]
        model_class = history['model_class'].values[0]
        parameters = {}
        features = []
        for column in history.columns:
            if column in ['score', 'model_class']:
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
        return MiraiModel(model_class, parameters, features)
