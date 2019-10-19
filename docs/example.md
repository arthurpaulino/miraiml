# Example notebook

This notebook will cover a regression case using scikit-learn's *California Housing* dataset.


```python
from sklearn.datasets import fetch_california_housing
import pandas as pd

X, y = fetch_california_housing(data_home='miraiml_local', return_X_y=True)
data = pd.DataFrame(X)
data['target'] = y
```

    Downloading Cal. housing from https://ndownloader.figshare.com/files/5976036 to miraiml_local


Let's split the data into training and testing data. In a real case scenario, we'd only have labels for training data.


```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, test_size=0.2)
```

## Building the search spaces

Let's compare (and ensemble) a `KNeighborsRegressor` and a pipeline composed by `StandardScaler` and a `LinearRegression`.


```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from miraiml import SearchSpace
from miraiml.pipeline import compose

Pipeline = compose(
    [('scaler', StandardScaler), ('lin_reg', LinearRegression)]
)

search_spaces = [
    SearchSpace(
        id='k-NN Regressor',
        model_class=KNeighborsRegressor,
        parameters_values=dict(
            n_neighbors=range(2, 9),
            weights=['uniform', 'distance'],
            p=range(2, 5)
        )
    ),
    SearchSpace(
        id='Pipeline',
        model_class=Pipeline,
        parameters_values=dict(
            scaler__with_mean=[True, False],
            scaler__with_std=[True, False],
            lin_reg__fit_intercept=[True, False]
        )
    )
]
```

## Configuring the Engine

For this test, let's use `r2_score` to evaluate our modeling.


```python
from sklearn.metrics import r2_score

from miraiml import Config

config = Config(
    local_dir='miraiml_local',
    problem_type='regression',
    score_function=r2_score,
    search_spaces=search_spaces,
    ensemble_id='Ensemble'
)
```

## Triggering the Engine

Let's also print the scores everytime the Engine finds a better solution for any base model.


```python
from miraiml import Engine

def on_improvement(status):
    scores = status.scores
    for key in sorted(scores.keys()):
        print('{}: {}'.format(key, round(scores[key], 3)), end='; ')
    print()

engine = Engine(config=config, on_improvement=on_improvement)
```

Now we're ready to load the data.


```python
engine.load_train_data(train_data, 'target')
engine.load_test_data(test_data)
```

Let's leave it running for 2 minutes and then interrupt it.


```python
from time import sleep

engine.restart()

sleep(120)

engine.interrupt()
```

    Ensemble: 0.606; Pipeline: 0.402; k-NN Regressor: 0.567; 
    Ensemble: 0.606; Pipeline: 0.402; k-NN Regressor: 0.567; 
    Ensemble: 0.622; Pipeline: 0.601; k-NN Regressor: 0.567; 
    Ensemble: 0.638; Pipeline: 0.601; k-NN Regressor: 0.567; 
    Ensemble: 0.638; Pipeline: 0.601; k-NN Regressor: 0.567; 
    Ensemble: 0.67; Pipeline: 0.601; k-NN Regressor: 0.598; 
    Ensemble: 0.671; Pipeline: 0.609; k-NN Regressor: 0.598; 
    Ensemble: 0.672; Pipeline: 0.609; k-NN Regressor: 0.598; 
    Ensemble: 0.672; Pipeline: 0.609; k-NN Regressor: 0.598; 
    Ensemble: 0.672; Pipeline: 0.609; k-NN Regressor: 0.598; 
    Ensemble: 0.733; Pipeline: 0.609; k-NN Regressor: 0.727; 
    Ensemble: 0.749; Pipeline: 0.609; k-NN Regressor: 0.727; 
    Ensemble: 0.749; Pipeline: 0.609; k-NN Regressor: 0.727; 
    Ensemble: 0.759; Pipeline: 0.609; k-NN Regressor: 0.759; 
    Ensemble: 0.763; Pipeline: 0.609; k-NN Regressor: 0.759; 
    Ensemble: 0.763; Pipeline: 0.609; k-NN Regressor: 0.759; 
    Ensemble: 0.763; Pipeline: 0.609; k-NN Regressor: 0.759; 
    Ensemble: 0.769; Pipeline: 0.609; k-NN Regressor: 0.768; 
    Ensemble: 0.769; Pipeline: 0.609; k-NN Regressor: 0.768; 
    Ensemble: 0.769; Pipeline: 0.609; k-NN Regressor: 0.768; 
    Ensemble: 0.769; Pipeline: 0.609; k-NN Regressor: 0.768; 


## Status analysis


```python
status = engine.request_status()
```

Let's see the status report.


```python
print(status.build_report(include_features=True))
```

    ########################
    best id: Ensemble
    best score: 0.769317652930817
    ########################
    ensemble weights:
        k-NN Regressor: 0.4315924504433665
        Pipeline: 0.04186130925207794
    ########################
    all scores:
        Ensemble: 0.769317652930817
        k-NN Regressor: 0.7678076788534821
        Pipeline: 0.6088760916515874
    ########################
    id: Pipeline
    model class: MiraiPipeline
    n features: 7
    parameters:
        lin_reg__fit_intercept: True
        scaler__with_mean: False
        scaler__with_std: True
    features: 1, 2, 3, 4, 5, 6, 7
    ########################
    id: k-NN Regressor
    model class: KNeighborsRegressor
    n features: 4
    parameters:
        n_neighbors: 5
        p: 4
        weights: uniform
    features: 0, 1, 2, 7
    


How does the k-NN Regressor's score changes with `n_neighbors`, on average?


```python
import matplotlib.pyplot as plt
%matplotlib inline

knn_history = status.histories['k-NN Regressor']

knn_history\
.groupby('n_neighbors__(hyperparameter)').mean()\
.reset_index()[['n_neighbors__(hyperparameter)', 'score']]\
.plot.scatter(x='n_neighbors__(hyperparameter)', y='score')

plt.show()
```


![png](example_files/example_19_0.png)


Again, in practice we wouldn't have labels for `test_data`, but how would the Engine perform on the test dataset?


```python
r2_score(test_data['target'], status.test_predictions['Ensemble'])
```




    0.7686513420142499


