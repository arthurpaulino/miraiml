from sklearn.model_selection import train_test_split
from miraiml import SearchSpace, Config, Engine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from time import sleep
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Let's use a single Naive Bayes classifier for this example.
search_spaces = [SearchSpace(model_class=GaussianNB, id='Gaussian NB')]

config = Config(
    local_dir = 'miraiml_local_on_improvement',
    problem_type = 'classification',
    search_spaces = search_spaces,
    score_function = roc_auc_score
)

# Simply printing the best score on improvement. This function must receive a
# dictionary, which is the return of the request_status method.
def on_improvement(status):
    best_id = status['best_id']
    scores = status['scores']
    print('Best score:', scores[best_id])

# Instantiating the engine
engine = Engine(config, on_improvement=on_improvement)

# Loading data
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)
train_target = train_data.pop('target_class')
engine.load_data(train_data, train_target, test_data)

# Starting the engine
engine.restart()

# Let's watch the engine print the best score for 10 seconds
sleep(10)
engine.interrupt()
