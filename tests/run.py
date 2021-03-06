from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from time import sleep
import pandas as pd

from miraiml import SearchSpace, Config, Engine
from miraiml.pipeline import compose

TEST_FOLDER = '.pytest_cache'


def test_run():
    X, y = fetch_california_housing(data_home=TEST_FOLDER, return_X_y=True)
    data = pd.DataFrame(X)
    data['target'] = y

    Pipeline = compose(
        [('scaler', StandardScaler), ('lin_reg', LinearRegression)]
    )

    search_spaces = [
        SearchSpace(id='Linear Regression', model_class=LinearRegression),
        SearchSpace(id='Lasso', model_class=Lasso),
        SearchSpace(
            id='Pipeline',
            model_class=Pipeline,
            parameters_values=dict(
                scaler__with_mean=[True, False],
                scaler__with_std=[True, False],
                lin_reg__fit_intercept=[True, False],
                lin_reg__normalize=[True, False]
            )
        )
    ]

    config = Config(
        local_dir=TEST_FOLDER,
        problem_type='regression',
        score_function=r2_score,
        search_spaces=search_spaces,
        ensemble_id='Ensemble',
        stagnation=1
    )

    engine = Engine(config)

    train_data, test_data = train_test_split(data, test_size=0.2)

    train_data_original, test_data_original = train_data.copy(), test_data.copy()

    engine.load_train_data(train_data, 'target')
    engine.load_test_data(test_data)

    if engine.is_running():
        raise AssertionError()

    engine.restart()

    sleep(2)

    if not engine.is_running():
        raise AssertionError()

    sleep(5)

    status = engine.request_status()

    if len(status.scores) != len(search_spaces) + 1 or \
            len(status.ensemble_weights) != len(search_spaces):
        raise AssertionError()

    if status.train_predictions.shape[0] != train_data.shape[0]:
        raise AssertionError()

    if status.test_predictions.shape[0] != test_data.shape[0]:
        raise AssertionError()

    for base_model in status.base_models.values():
        for feature in base_model['features']:
            if feature not in test_data.columns or feature not in train_data.columns:
                raise AssertionError()

    engine.interrupt()

    if engine.is_running():
        raise AssertionError()

    engine.clean_test_data(restart=True)

    sleep(5)

    if not engine.is_running():
        raise AssertionError()

    engine.shuffle_train_data(restart=True)

    sleep(5)

    status = engine.request_status()

    if status.test_predictions is not None:
        raise AssertionError()

    engine.interrupt()

    status.build_report()
    status.build_report(include_features=True)

    pd.testing.assert_frame_equal(train_data, train_data_original)
    pd.testing.assert_frame_equal(test_data, test_data_original)
