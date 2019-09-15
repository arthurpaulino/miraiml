from sklearn.linear_model import LinearRegression, Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from time import sleep
import pandas as pd

from miraiml import HyperSearchSpace, Config, Engine

TEST_FOLDER = '.pytest_cache'


def test_run():
    X, y = fetch_california_housing(data_home=TEST_FOLDER, return_X_y=True)
    data = pd.DataFrame(X)
    data['target'] = y

    hyper_search_spaces = [
        HyperSearchSpace(model_class=LinearRegression, id='Linear Regression'),
        HyperSearchSpace(model_class=Lasso, id='Lasso')
    ]

    config = Config(
        local_dir=TEST_FOLDER,
        problem_type='regression',
        hyper_search_spaces=hyper_search_spaces,
        score_function=r2_score,
        ensemble_id='Ensemble'
    )

    engine = Engine(config)

    train_data, test_data = train_test_split(data, test_size=0.2)

    engine.load_data(train_data, 'target', test_data)

    if engine.is_running():
        raise AssertionError()

    engine.restart()

    sleep(2)

    if not engine.is_running():
        raise AssertionError()

    sleep(5)

    status = engine.request_status()

    if len(status['scores']) != 3 or len(status['ensemble_weights']) != 2:
        raise AssertionError()

    if status['predictions'].shape[0] != test_data.shape[0]:
        raise AssertionError()

    for base_model in status['base_models'].values():
        for feature in base_model['features']:
            if feature not in test_data.columns or feature not in train_data.columns:
                raise AssertionError()

    engine.interrupt()

    if engine.is_running():
        raise AssertionError()

    engine.load_data(train_data, 'target', restart=True)

    sleep(2)

    if not engine.is_running():
        raise AssertionError()

    sleep(5)

    status = engine.request_status()

    if status["predictions"] is not None:
        raise AssertionError()

    engine.interrupt()
