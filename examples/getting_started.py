from sklearn.model_selection import train_test_split
from miraiml import SearchSpace, Config, Engine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from time import sleep
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# First, let's define our list of SearchSpaces. We're going to allow the engine
# to work with a single Gaussian Naive Bayes classifier for this example. There
# is no hyperparameter search in this case, but the engine still searches for a
# good set of features to use.
search_spaces = [
    SearchSpace(model_class=GaussianNB, id='Gaussian NB')
]

# Now we configure the behavior of the engine.
config = Config(
    local_dir = 'miraiml_local_getting_started',
    problem_type = 'classification',
    search_spaces = search_spaces,
    score_function = roc_auc_score
)

# To instantiate the engine, we just need the `config` object.
engine = Engine(config)

# Suppose we have a training dataset (labeled) and a testing dataset, for which
# we don't have labels.
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)

# Now we load the data and inform the name of the target column.
engine.update_data(train_data, test_data, target='target_class')

# Ready to roll. To check if it's running asynchronously, we will start it and
# then call `is_running` after 1 second.
engine.restart()
print('Waiting 1 second...')
sleep(1)
print('\nIs the engine running?', engine.is_running())

# We can request the score at any time. Let's just wait 5 seconds to make sure
# that at least one cycle of cross-validation has finished.
print('\nWaiting 5 seconds...')
sleep(5)
print('\nScore of the engine:', engine.request_score())

# We can also request the predictions for the testing data anytime we want
print('\nPredictions for the testing data:', engine.request_predictions())

# Okay, let's shut it down and check if it's running every second. The print
# inside the loop may not be called.
engine.interrupt()
while engine.is_running():
    print('\nEngine still running')
    sleep(1)

# Predictions and scores are still available, but may have improved.
print('\nScore of the engine:', engine.request_score())
print('\nPredictions for the testing data:', engine.request_predictions())
