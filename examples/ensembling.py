from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

from time import sleep
import pandas as pd
import numpy as np
import warnings

from miraiml import HyperSearchSpace, Config, Engine

warnings.filterwarnings('ignore')

# We're going to ensemble a Naive Bayes classifier and a K-NN classifier.
hyper_search_spaces = [
    HyperSearchSpace(model_class=GaussianNB, id='Gaussian NB'),
    HyperSearchSpace(model_class=KNeighborsClassifier, id='K-NN', parameters_values= {
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
    hyper_search_spaces = hyper_search_spaces,
    score_function = roc_auc_score,
    ensemble_id = 'Ensemble'
)

# Instantiating the engine
engine = Engine(config)

# Load data
data = pd.read_csv('pulsar_stars.csv')
train_data, test_data = train_test_split(data, stratify=data['target_class'],
    test_size=0.2, random_state=0)
engine.load_data(train_data, 'target_class', test_data)

# Starting the engine
print('Training...')
engine.restart()

# Let's wait 20 seconds and interrupt it
sleep(20)
engine.interrupt()

# Let's explore the status object
status = engine.request_status()

# As seen before, we can see the engine's scores
print('\nScores:', status['scores'])

# We have access to a shortcut that informs the best id
print('\nBest id:', status['best_id'])

# We have the predictions for each id on a pandas.DataFrame object
predictions = status['predictions']
print('\nPredictions:')
print(predictions.head())

# We can inquire the ensemble weights
print('\Ensemble weights:', status['ensemble_weights'])

# Let's see the details of each base model
base_models = status['base_models']
for id in base_models:
    base_model = base_models[id]
    print('\nBase model id:', id)
    print(' Model class:', base_model['model_class'])
    print(' Parameters:', base_model['parameters'])
    print(' Features:', base_model['features'])
