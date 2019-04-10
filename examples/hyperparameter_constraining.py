from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from miraiml import BaseLayout, Config, Engine
from sklearn.metrics import roc_auc_score
from time import sleep
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Logistic Regression cannot use the 'l1' penalty for the solvers 'newton-cg',
# 'sag' and 'lbfgs' and the engine is designed to work with general model classes.

# So, the way that we can let it know is by defining a `parameters_rules` function,
# which receives a dictionary of parameters and changes them if needed.

def logistic_regression_parameters_rules(parameters):
    if parameters['solver'] in ['newton-cg', 'sag', 'lbfgs']:
        parameters['penalty'] = 'l2'

# We just need to make sure that those parameters will exist in the set of
# parameters tested by the engine, otherwise it will scream some messages to let
# us know that we asked it to access invalid keys on the dictionary.

# Now we create the list of base models containing only a Logistic Regression
# base model.

base_layouts = [
    BaseLayout(
        LogisticRegression,
        'Logistic Regression',
        parameters_values = {
            'penalty': ['l1', 'l2'],
            'C': np.arange(0.1, 2, 0.1),
            'max_iter': np.arange(50, 300),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'random_state': [0]
        },
        parameters_rules=logistic_regression_parameters_rules
    )
]

# Quick configuration
config = Config(
    local_dir = 'miraiml_local_hyperparameter_constraining',
    problem_type = 'classification',
    base_layouts = base_layouts,
    score_function = roc_auc_score
)

# Instantiating the engine
engine = Engine(config)

# Load data
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)
engine.update_data(train_data, test_data, target='target_class')

# Fire!
engine.restart()

# Let's wait 5 seconds and interrupt it
sleep(5)
engine.interrupt()

print('Score:', engine.request_score())
