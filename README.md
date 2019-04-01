# MiraiML

> Mirai: _future_ in japanese.

MiraiML is an asynchronous engine for autonomous machine learning built for
real-time usage.

MiraiML attempts to improve the chosen metric by searching good hyperparameters
and sets of features to feed statistical models that implement the methods `fit`
and `predict_proba` (or just `predict` for regressions).

To use MiraiML, we need to understand the classes `MiraiLayout`, `MiraiConfig`
and `MiraiML`. Feel free to go straight to the [example notebook](example.ipynb).

## `MiraiLayout`

This is where we define the search hyperspace for each base statistical model.
In order to instantiate a `MiraiLayout`, we need:

- A statistical model class;

- An id;

- A dictionary containing a list of values to be tested for each hyperparameter
  (optional);

- A function to modify hyperparameters if there are constraints that prohibits
  two or more hyperparameter values from being used at the same time (optional).
  [For instance](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html):

  > penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
  >
  > Used to specify the norm used in the penalization. The ‘newton-cg’, ‘sag’
  > and ‘lbfgs’ solvers support only l2 penalties.

## `MiraiConfig`

It defines the general behavior of the engine. Its attributes are:

- The number of folds for the cross-validation;

- The problem type (classification or regression);

- Whether the folds should be stratified or not;

- The scoring function;

- A list of `MiraiLayouts`;

- An id for the ensemble;

- The number of cycles to attempt improvements on the ensemble;

- A flag that tells whether we want the engine to print improvements or not.

## `MiraiML`

This class provides the controls for the engine. To instantiate it, we just need
to provide a `MiraiConfig` object. Let's see what we can do by studying its
methods.

- `interrupt`: sets a flag to make the engine stop on the first opportunity;

- `update_data`: interrupts the engine and loads a new pair of train/test
  datasets. It has an optional parameter to restart the engine afterwards;

- `reconfig`: interrupts the engine and loads a new configuration. It also has
  an optional parameter to restart the engine afterwards;

- `restart`: interrupts the engine and start again;

- `request_score`: queries the score of the best model (or ensemble);

- `request_predictions`: queries the predictions of the best model (or ensemble)
  for the test data.
