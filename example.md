# MiraiML usage example

## The dataset

Let's use the csv file `pulsar_stars.csv` to explore MiraiML functionalities. It's a dataset downloaded from Kaggle ([Predicting a Pulsar Star](https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star)).

```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('pulsar_stars.csv')
data.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mean of the integrated profile</th>
      <th>Standard deviation of the integrated profile</th>
      <th>Excess kurtosis of the integrated profile</th>
      <th>Skewness of the integrated profile</th>
      <th>Mean of the DM-SNR curve</th>
      <th>Standard deviation of the DM-SNR curve</th>
      <th>Excess kurtosis of the DM-SNR curve</th>
      <th>Skewness of the DM-SNR curve</th>
      <th>target_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>140.562500</td>
      <td>55.683782</td>
      <td>-0.234571</td>
      <td>-0.699648</td>
      <td>3.199833</td>
      <td>19.110426</td>
      <td>7.975532</td>
      <td>74.242225</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>102.507812</td>
      <td>58.882430</td>
      <td>0.465318</td>
      <td>-0.515088</td>
      <td>1.677258</td>
      <td>14.860146</td>
      <td>10.576487</td>
      <td>127.393580</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>103.015625</td>
      <td>39.341649</td>
      <td>0.323328</td>
      <td>1.051164</td>
      <td>3.121237</td>
      <td>21.744669</td>
      <td>7.735822</td>
      <td>63.171909</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>136.750000</td>
      <td>57.178449</td>
      <td>-0.068415</td>
      <td>-0.636238</td>
      <td>3.642977</td>
      <td>20.959280</td>
      <td>6.896499</td>
      <td>53.593661</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>88.726562</td>
      <td>40.672225</td>
      <td>0.600866</td>
      <td>1.123492</td>
      <td>1.178930</td>
      <td>11.468720</td>
      <td>14.269573</td>
      <td>252.567306</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 17898 entries, 0 to 17897
    Data columns (total 9 columns):
     Mean of the integrated profile                  17898 non-null float64
     Standard deviation of the integrated profile    17898 non-null float64
     Excess kurtosis of the integrated profile       17898 non-null float64
     Skewness of the integrated profile              17898 non-null float64
     Mean of the DM-SNR curve                        17898 non-null float64
     Standard deviation of the DM-SNR curve          17898 non-null float64
     Excess kurtosis of the DM-SNR curve             17898 non-null float64
     Skewness of the DM-SNR curve                    17898 non-null float64
    target_class                                     17898 non-null int64
    dtypes: float64(8), int64(1)
    memory usage: 1.2 MB

It's a pretty clean and simple dataset related to a classification problem and the target column is called `target_class`. Let's suppose that we have a training dataset (labeled) and a testing dataset, for which we don't have labels.

```python
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, stratify=data['target_class'], test_size=0.2, random_state=0)
```

## The Engine

### `BaseLayout`

First, let's define our list of `BaseLayout`s.

```python
from miraiml import BaseLayout
import numpy as np

base_layouts = []
```

#### Random Forest and Extra Trees

Let's use the same search space for both of them.

```python
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

parameters = {
    'n_estimators': np.arange(5, 30),
    'max_depth': np.arange(2, 20),
    'min_samples_split': np.arange(0.1, 1.1, 0.1),
    'min_weight_fraction_leaf': np.arange(0, 0.6, 0.1),
    'random_state': [0]
}

base_layouts += [
    BaseLayout(model_class=RandomForestClassifier, id='Random Forest', parameters_values=parameters),
    BaseLayout(ExtraTreesClassifier, 'Extra Trees', parameters),
]
```

#### Gradient Boosting

Let's use similar parameters.

```python
from sklearn.ensemble import GradientBoostingClassifier

base_layouts.append(BaseLayout(
    GradientBoostingClassifier, 'Gradient Boosting', {
        'n_estimators': np.arange(10, 130),
        'learning_rate': np.arange(0.05, 0.15, 0.01),
        'subsample': np.arange(0.5, 1, 0.01),
        'max_depth': np.arange(2, 20),
        'min_weight_fraction_leaf': np.arange(0, 0.6, 0.1),
        'random_state': [0]
    }
))
```

#### Logistic Regression

Let's try something new here. Let's constrain a parameter.

```python
from sklearn.linear_model import LogisticRegression

def logistic_regression_parameters_rules(parameters):
    if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
        parameters['penalty'] = 'l2'

base_layouts.append(BaseLayout(LogisticRegression, 'Logistic Regression', {
        'penalty': ['l1', 'l2'],
        'C': np.arange(0.1, 2, 0.1),
        'max_iter': np.arange(50, 300),
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'random_state': [0]
    },
    parameters_rules=logistic_regression_parameters_rules
))
```

#### Gaussian Naive Bayes

No parameters here. The engine will just search for an interesting set of features.

```python
from sklearn.naive_bayes import GaussianNB

base_layouts.append(BaseLayout(GaussianNB, 'Gaussian NB'))
```

#### K-Nearest Neighbors

```python
from sklearn.neighbors import KNeighborsClassifier

base_layouts.append(BaseLayout(KNeighborsClassifier, 'K-NN', {
    'n_neighbors': np.arange(1, 15),
    'weights': ['uniform', 'distance'],
    'p': np.arange(1, 5)
}))
```

Alright. Good enough for now.

### `Config`

Now we define the general behavior of the engine.

```python
from sklearn.metrics import roc_auc_score
from miraiml import Config

config = Config(
    local_dir = 'miraiml_local',
    problem_type = 'classification',
    base_layouts = base_layouts,
    n_folds = 5,
    stratified = True,
    score_function = roc_auc_score,
    mirai_exploration_ratio = 0.5,
    ensemble_id = 'Ensemble',
    n_ensemble_cycles = 1000,
    report = False
)
```

Ok, that was easy.

### `Engine`

Let's see it running.

```python
from miraiml import Engine

engine = Engine(config)
```

Let's load the training and testing datasets.

```python
engine.update_data(train_data, test_data, target='target_class')
```

Ready to roll. In order to keep this notebook clean, let's show the scores every 20 seconds three times and then interrupt the engine.

```python
from time import sleep

engine.restart()

for _ in range(3):
    sleep(20)
    engine.report()

engine.interrupt()
```

                        id     score    weight
    0  Logistic Regression  0.974392  0.995657
    1             Ensemble  0.968524       NaN
    2          Extra Trees  0.959366  0.665604
    3          Gaussian NB  0.953072  0.155789
    4    Gradient Boosting  0.949128  0.711959
    5                 K-NN  0.939656  0.226841
    6        Random Forest  0.916895  0.062783

                        id     score    weight
    0  Logistic Regression  0.974392  0.980699
    1             Ensemble  0.969277       NaN
    2          Extra Trees  0.959366  0.495164
    3          Gaussian NB  0.953072  0.238956
    4    Gradient Boosting  0.949128  0.413303
    5        Random Forest  0.942440  0.017965
    6                 K-NN  0.939656  0.037094

                        id     score    weight
    0             Ensemble  0.974481       NaN
    1  Logistic Regression  0.974392  0.909078
    2    Gradient Boosting  0.967560  0.670053
    3          Extra Trees  0.959474  0.081531
    4          Gaussian NB  0.957497  0.086895
    5                 K-NN  0.945346  0.149539
    6        Random Forest  0.942440  0.438359

We can also request the predictions for the testing data anytime we want:

```python
test_predictions = engine.request_predictions()
test_predictions
```

    array([0.83163376, 0.00570235, 0.01726076, ..., 0.02851306, 0.00362415,
           0.00400597])

For the sake of curiosity, let's see how we were able to perform.

```python
roc_auc_score(test_data['target_class'], test_predictions)
```

    0.9749966249662496

That's it for now. There's more to come!
