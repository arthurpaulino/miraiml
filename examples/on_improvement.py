from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

from time import sleep
import pandas as pd
import warnings

from miraiml import HyperSearchSpace, Config, Engine

warnings.filterwarnings('ignore')

# Let's use a single Naive Bayes classifier for this example.
hyper_search_spaces = [HyperSearchSpace(model_class=GaussianNB, id='Gaussian NB')]

config = Config(
    local_dir = 'miraiml_local_on_improvement',
    problem_type = 'classification',
    hyper_search_spaces = hyper_search_spaces,
    score_function = roc_auc_score
)

# Simply printing the best score on improvement. This function must receive a
# dictionary, which is the return of the request_status method.
def on_improvement(status):
    print('Score:', status['score'])

# Instantiating the engine
engine = Engine(config, on_improvement=on_improvement)

# Loading data
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)
engine.load_data(train_data, 'target_class', test_data)

# Starting the engine
engine.restart()

# Let's watch the engine print the best score for 10 seconds
sleep(10)
engine.interrupt()
