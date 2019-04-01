from sklearn.model_selection import StratifiedKFold, KFold
from random import triangular
from threading import Thread
import numpy as np
import pickle

LOCAL_DIR = 'miraiml_local/'
MODELS_DIR = 'models/'

def load(path):
    return pickle.load(open(path, 'rb'))

def par_dump(obj, path):
    Thread(target=lambda: pickle.dump(obj, open(path, 'wb'))).start()

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

def gen_weights(scores, ensemble_id):
    weights = {}
    ids = [id for id in scores if id != ensemble_id]
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
