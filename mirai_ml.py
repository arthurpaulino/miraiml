from random import triangular, choices, random, sample, choice, uniform
from sklearn.model_selection import StratifiedKFold, KFold
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
        event = {'score':score}
        for parameter in mirai_model.parameters:
            event[parameter+'(parameter)'] = mirai_model.parameters[parameter]
        for feature in self.all_features:
            event[feature+'(feature)'] = feature in mirai_model.features

        self.history[id] = pd.concat([self.history[id],
            pd.DataFrame([event])]).drop_duplicates()
        par_dump(self.history, self.history_path)

    def is_ready(self, id):
        return self.history[id].shape[0] > 1

    def gen_parameters_features(self, id):
        # The magic happens here. For each parameter and feature, its value
        # (True or False for features) is chosen stochastically depending on the
        # mean score of the validations in which the value was chosen before.
        # Better parameters and features have higher chances of being chosen.
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


class MiraiLayout:
    def __init__(self, model_class, id, parameters_values={},
            parameters_rules=lambda x: x):
        self.model_class = model_class
        self.id = id
        self.parameters_values = parameters_values
        self.parameters_rules = parameters_rules

    def gen_parameters_features(self, all_features):
        model_class = self.model_class
        parameters = {}
        for parameter in self.parameters_values:
            parameters[parameter] = choice(self.parameters_values[parameter])
        features = sample_random_len(all_features)
        return (parameters, features)

class MiraiConfig:
    def __init__(self, parameters):
        self.n_folds = parameters['n_folds']
        self.problem_type = parameters['problem_type']
        self.stratified = parameters['stratified']
        self.score_function = parameters['score_function']
        self.mirai_layouts = parameters['mirai_layouts']
        self.mirai_exploration_ratio = parameters['mirai_exploration_ratio']
        self.ensemble_id = parameters['ensemble_id']
        self.n_ensemble_cycles = parameters['n_ensemble_cycles']
        self.report = parameters['report']

class MiraiML:
    def __init__(self, config):
        self.config = config
        self.is_running = False
        self.must_interrupt = False
        self.mirai_seeker = None

    def interrupt(self):
        self.must_interrupt = True
        while self.is_running:
            sleep(.1)
        self.must_interrupt = False

    def update_data(self, train_data, test_data, target, restart=False):
        self.interrupt()
        self.X_train = train_data.drop(columns=target)
        self.all_features = list(self.X_train.columns)
        self.y_train = train_data[target]
        self.X_test = test_data
        if not self.mirai_seeker is None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def reconfig(self, config, restart=False):
        self.interrupt()
        self.config = config
        if not self.mirai_seeker is None:
            self.mirai_seeker.reset()
        if restart:
            self.restart()

    def restart(self):
        self.interrupt()
        Thread(target=lambda: self.main_loop()).start()

    def main_loop(self):
        self.is_running = True
        if not os.path.exists(LOCAL_DIR + MODELS_DIR):
            os.makedirs(LOCAL_DIR + MODELS_DIR)
        self.mirai_models = {}
        self.mirai_models_ids = []
        self.train_predictions_dict = {}
        self.test_predictions_dict = {}
        self.scores = {}
        self.best_score = None
        self.best_id = None
        self.weights = {}

        for mirai_layout in self.config.mirai_layouts:
            if self.must_interrupt:
                break
            id = mirai_layout.id
            self.mirai_models_ids.append(id)
            mirai_model_path = LOCAL_DIR + MODELS_DIR + id
            if os.path.exists(mirai_model_path):
                mirai_model = load(mirai_model_path)
            else:
                parameters, features = mirai_layout.gen_parameters_features(self.all_features)
                mirai_layout.parameters_rules(parameters)
                mirai_model = MiraiModel(mirai_layout.model_class, parameters, features)
                par_dump(mirai_model, mirai_model_path)
            self.mirai_models[id] = mirai_model

            self.train_predictions_dict[id], self.test_predictions_dict[id],\
                self.scores[id] = self.mirai_models[id].predict(self.X_train,
                    self.y_train, self.X_test, self.config)

            if self.best_score is None or self.scores[id] > self.best_score:
                self.best_score = self.scores[id]
                self.best_id = id

        self.mirai_seeker = MiraiSeeker(self.mirai_models_ids, self.all_features)

        ensemble_id = self.config.ensemble_id
        weights_path = LOCAL_DIR + MODELS_DIR + ensemble_id
        if os.path.exists(weights_path):
            self.weights = load(weights_path)
        else:
            self.weights = self.gen_weights()
            par_dump(self.weights, weights_path)

        self.train_predictions_dict[ensemble_id], self.test_predictions_dict[ensemble_id],\
            self.scores[ensemble_id] = self.ensemble(self.weights)

        if self.scores[ensemble_id] > self.best_score:
            self.best_score = self.scores[ensemble_id]
            self.best_id = ensemble_id

        self.search_weights()

        if self.config.report:
            self.report()
        while not self.must_interrupt:
            for mirai_layout in self.config.mirai_layouts:
                if self.must_interrupt:
                    break
                id = mirai_layout.id

                if self.mirai_seeker.is_ready(id) and\
                    uniform(0, 1) < self.config.mirai_exploration_ratio:
                    parameters, features = self.mirai_seeker.gen_parameters_features(id)
                else:
                    parameters, features = mirai_layout.gen_parameters_features(self.all_features)
                mirai_layout.parameters_rules(parameters)
                mirai_model = MiraiModel(mirai_layout.model_class, parameters, features)

                train_predictions, test_predictions, score = mirai_model.\
                    predict(self.X_train, self.y_train, self.X_test,
                        self.config)

                self.mirai_seeker.register_mirai_model(id, mirai_model, score)

                if score > self.scores[id]:
                    self.scores[id] = score
                    self.train_predictions_dict[id] = train_predictions
                    self.test_predictions_dict[id] = test_predictions
                    if score > self.best_score:
                        self.best_score = score
                        self.best_id = id
                    self.train_predictions_dict[ensemble_id],\
                        self.test_predictions_dict[ensemble_id],\
                        self.scores[ensemble_id] = self.ensemble(self.weights)
                    if self.scores[ensemble_id] > self.best_score:
                        self.best_score = self.scores[ensemble_id]
                        self.best_id = ensemble_id
                    if self.config.report:
                        self.report()
                    par_dump(mirai_model, LOCAL_DIR + MODELS_DIR + id)

            self.search_weights()

        self.is_running = False

    def gen_weights(self):
        weights = {}
        min_score, max_score = np.inf, -np.inf
        for id in self.mirai_models_ids:
            score = self.scores[id]
            min_score = min(min_score, score)
            max_score = max(max_score, score)
        diff_score = max_score - min_score
        for id in self.mirai_models_ids:
            weights[id] = triangular(0, 1, (self.scores[id]-min_score)/diff_score)
        return weights

    def search_weights(self):
        ensemble_id = self.config.ensemble_id
        for _ in range(self.config.n_ensemble_cycles):
            if self.must_interrupt:
                break
            weights = self.gen_weights()
            train_predictions, test_predictions, score = self.ensemble(weights)
            if score > self.scores[ensemble_id]:
                self.scores[ensemble_id] = score
                self.weights = weights
                self.train_predictions_dict[ensemble_id] = train_predictions
                self.test_predictions_dict[ensemble_id] = test_predictions
                if score > self.best_score:
                    self.best_score = score
                    self.best_id = ensemble_id
                par_dump(weights, LOCAL_DIR + MODELS_DIR + ensemble_id)
                if self.config.report:
                    self.report()

    def ensemble(self, weights):
        ids = sorted(weights)
        id = ids[0]
        train_predictions = weights[id]*self.train_predictions_dict[id]
        test_predictions = weights[id]*self.test_predictions_dict[id]
        weights_sum = weights[id]
        for id in ids[1:]:
            train_predictions += weights[id]*self.train_predictions_dict[id]
            test_predictions += weights[id]*self.test_predictions_dict[id]
            weights_sum += weights[id]
        train_predictions /= weights_sum
        test_predictions /= weights_sum
        return (train_predictions, test_predictions,
            self.config.score_function(self.y_train, train_predictions))

    def request_score(self):
        if len(self.scores) > 0:
            return self.scores[self.best_id]
        return None

    def request_predictions(self):
        if len(self.test_predictions_dict) > 0:
            return self.test_predictions_dict[self.best_id]
        return None

    def report(self):
        status = []
        for id in self.scores:
            status.append({
                'id': id,
                'score': self.scores[id],
                'weight': self.weights[id] if id in self.weights else np.nan
            })
        print()
        print(pd.DataFrame(status).sort_values('score',
            ascending=False).reset_index(drop=True))
