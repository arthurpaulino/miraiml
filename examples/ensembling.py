from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from miraiml import BaseLayout, Config, Engine
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score
from time import sleep
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# We're going to ensemble a Naive Bayes classifier and a K-NN classifier.
base_layouts = [
    BaseLayout(model_class=GaussianNB, id='Gaussian NB'),
    BaseLayout(model_class=KNeighborsClassifier, id='K-NN', parameters_values= {
        'n_neighbors': np.arange(1, 15),
        'weights': ['uniform', 'distance'],
        'p': np.arange(1, 5)
    })
]

# We have to signal it on the config, otherwise the engine will not attempt to
# ensemble them.
config = Config(
    local_dir = 'miraiml_local_ensembling',
    problem_type = 'classification',
    base_layouts = base_layouts,
    score_function = roc_auc_score,
    ensemble_id = 'Ensemble',
    n_ensemble_cycles = 1000
)

# Instantiating the engine
engine = Engine(config)

# Load data
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)
engine.update_data(train_data, test_data, target='target_class')

# Starting the engine
engine.restart()

# Let's wait 10 seconds and interrupt it
sleep(10)
engine.interrupt()

# Now we can requests all scores
print('Scores:')
print(engine.request_scores())