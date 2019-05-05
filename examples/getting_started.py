from time import sleep
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

from miraiml import HyperSearchSpace, Config, Engine

# First, let's define our list of HyperSearchSpaces. We're going to allow the engine
# to work with a single Gaussian Naive Bayes classifier for this example. There
# is no hyperparameter search in this case, but the engine still searches for a
# good set of features to use.
hyper_search_spaces = [
    HyperSearchSpace(model_class=GaussianNB, id='Gaussian NB')
]

# Now we configure the behavior of the engine.
config = Config(
    local_dir='miraiml_local_getting_started',
    problem_type='classification',
    hyper_search_spaces=hyper_search_spaces,
    score_function=roc_auc_score
)

# To instantiate the engine, we just need the `config` object.
engine = Engine(config)

# Suppose we have a training dataset (labeled) and a testing dataset, for which
# we don't have labels.
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
                                         test_size=0.2, random_state=0)

# Now we load the data and inform the name of the target column.
engine.load_data(train_data, 'target_class', test_data)

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

# Let's print the scores of the engine's base models
status = engine.request_status()
print('\nScores:', status['scores'])

# Okay, let's wait 5 more seconds and shut it down.
print('\nWaiting 5 seconds...')
sleep(5)
engine.interrupt()

# The engine's status is still available and the scores may have improved
status = engine.request_status()
print('\nScores:', status['scores'])
