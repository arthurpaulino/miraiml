from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

from time import sleep
import pandas as pd
import numpy as np
import warnings

from miraiml import SearchSpace, Config, Engine

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
train_target = train_data.pop('target_class')

# Now we load the data and inform the name of the target column.
engine.load_data(train_data, train_target, test_data)

# Ready to roll. To check if it's running asynchronously, we will start it and
# then call `is_running` after 1 second.
engine.restart()
print('Waiting 1 second...')
sleep(1)
print('\nIs the engine running?', engine.is_running())

# We can request the engine's status at any time. Let's just wait 5 seconds to
# make sure that at least one cycle of cross-validation has finished.
print('\nWaiting 5 seconds...')
sleep(5)
status = engine.request_status()

# Let's print the score of the engine
print('\nScore:', status['score'])

# Okay, let's wait 5 more seconds and shut it down.
print('\nWaiting 5 seconds...')
sleep(5)
engine.interrupt()

# The engine's status is still available and the score may have improved
status = engine.request_status()
print('\nScore:', status['score'])
