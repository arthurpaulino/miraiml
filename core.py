from random import choice, uniform
from threading import Thread
from time import sleep
import pandas as pd
import os

from util import *

class MiraiLayout:
    def __init__(self, model_class, id, parameters_values={},
            parameters_rules=lambda x: x):
        self.model_class = model_class
        self.id = id
        self.parameters_values = parameters_values
        self.parameters_rules = parameters_rules

    def gen_mirai_model(self, all_features):
        model_class = self.model_class
        parameters = {}
        for parameter in self.parameters_values:
            parameters[parameter] = choice(self.parameters_values[parameter])
        self.parameters_rules(parameters)
        features = sample_random_len(all_features)
        return MiraiModel(model_class, parameters, features)

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
                mirai_model = mirai_layout.gen_mirai_model(self.all_features)
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
            self.weights = gen_weights(self.scores, self.mirai_models_ids)
            par_dump(self.weights, weights_path)

        self.train_predictions_dict[ensemble_id], self.test_predictions_dict[ensemble_id],\
            self.scores[ensemble_id] = ensemble(self.train_predictions_dict,
                self.test_predictions_dict, self.weights, self.y_train, self.config)

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
                    mirai_model = self.mirai_seeker.gen_mirai_model(id)
                else:
                    mirai_model = mirai_layout.gen_mirai_model(self.all_features)

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
                        self.test_predictions_dict[ensemble_id], self.scores[ensemble_id] = \
                        ensemble(self.train_predictions_dict, self.test_predictions_dict,
                            self.weights, self.y_train, self.config)
                    if self.scores[ensemble_id] > self.best_score:
                        self.best_score = self.scores[ensemble_id]
                        self.best_id = ensemble_id
                    if self.config.report:
                        self.report()
                    par_dump(mirai_model, LOCAL_DIR + MODELS_DIR + id)

            self.search_weights()

        self.is_running = False

    def search_weights(self):
        ensemble_id = self.config.ensemble_id
        for _ in range(self.config.n_ensemble_cycles):
            if self.must_interrupt:
                break
            weights = gen_weights(self.scores, self.mirai_models_ids)
            train_predictions, test_predictions, score = \
                ensemble(self.train_predictions_dict, self.test_predictions_dict,
                    weights, self.y_train, self.config)
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
