User Guide
==========

Introduction
------------

    Mirai: `future` in japanese.

MiraiML is an asynchronous engine for continuous & autonomous machine learning,
built for real-time usage.

- It's asynchronous because it runs in background, allowing you to execute custom
  Python code as you interact with the engine;

- It's continuous because it can run "forever", always looking for solutions that
  can achieve better performances;

- It's autonomous because it does not wander on the search space blindly and does
  not perform exhaustive grid searches. Instead, it combines past attempts to guide
  its next steps, always allowing itself to jump out of local minima.

MiraiML improves the chosen metric by searching good hyperparameters and sets of
features for base statistical models, whilst finding smart ways to combine the
predictions of those models in order to achieve even higher scores.

    `But how can MiraiML help me? And how does it even work?`

We're going to address these questions on the next subsections.

MiraiML usability
-----------------

Tired of coding the same grid searches, cross-validations, training and predicting
scripts over and over? I was. MiraiML does it all with a simple API, so you can
spend less time on such mechanical tasks. MiraiML works on the typical train/test
scenario, when the data can fit in the RAM. Let's explore the API from a bottom-up
perspective.

The basic usage flow is represented in the image below:

.. https://docs.google.com/drawings/d/1NC2A2YtpNGOx8Tle0ElODIifIPXaQ-Ex5vi_pUvT_Vc/edit?usp=sharing
.. image:: https://docs.google.com/drawings/d/e/2PACX-1vQOnR9eb3bAKqrPA9UqSrhdk17iXtFgb8ukqpqdUAal8wYHH3BFj2JowmqOaI1_xBrjn01fqw0lcMn-/pub?w=1051&h=150

Search spaces
*************

MiraiML requires that you define the search spaces in which it will look for
solution candidates. In order to instantiate a search space, you need to use the
:class:`miraiml.SearchSpace` class. A search space is a combination of an id, a
model class and a dictionary of hyperparameters values to be tested. The only
requirement is that the model class must implement a ``fit`` method as well as a
``predict`` method for regression problems or a ``predict_proba`` for
classification problems. For instance, you can use scikit-learn's models:

::

    >>> from sklearn.linear_model import LinearRegression
    >>> from miraiml import SearchSpace

    >>> search_space = SearchSpace(
    ...     id = 'Linear Regression',
    ...     model_class = LinearRegression,
    ...     parameters_values = dict(
    ...         fit_intercept = [True, False],
    ...         normalize = [True, False]
    ...     )
    ... )

:class:`miraiml.SearchSpace` also allows you to provide a `parameters_rules`
function to deal with prohibitive combinations of hyperparameters. Please refer
to its documentation for further understanding.

After you've defined your search spaces, the next step is building the
configuration object.

Configuration
*************

The configuration for MiraiML's Engine is defined by an instance of the
:class:`miraiml.Config` class, which tells the Engine where to save its local
files, the problem type, the function to score the candidate solutions, the search
spaces that should be used and a few other things. For instance:

::

    >>> from sklearn.metrics import r2_score
    >>> from miraiml import Config

    >>> config = Config(
    ...     local_dir = 'miraiml_local',
    ...     problem_type = 'regression',
    ...     score_function = r2_score,
    ...     search_spaces = [search_space]
    ... )

Alright, now we're all set to use the Engine.

The Engine
**********

:class:`miraiml.Engine` provides a straightforward interface to access its
functionalities. The instantiation only requires a configuration object:

::

    >>> from miraiml import Engine

    >>> engine = Engine(config)

.. note::
    You can also provide a ``on_improvement`` function that will be executed
    everytime the engine finds a better modeling solution. Check out the API
    documentation for more information.

Let's use scikit-learn's classic `California Housing` dataset as an example:

::

    >>> from sklearn.datasets import fetch_california_housing
    >>> import pandas as pd

    >>> X, y = fetch_california_housing(return_X_y=True)
    >>> data = pd.DataFrame(X)
    >>> data['target'] = y

    >>> engine.load_train_data(train_data=data, target_column='target')

After the training data is loaded, you can trigger the optimization process with:

::

    >>> engine.restart()

And to interrupt it:

::

    >>> engine.interrupt()

The :class:`miraiml.Engine` documentation contains the full set of functionalities
that are available for you.

MiraiML internals
-----------------

MiraiML works in cycles. In each cycle, the Engine tries to find better solutions
for each search space and for the ensemble. There are three main concepts at play
here:

* *Base models* represent solutions in the search space
* *Mirai Seeker* manages the walk through the search spaces
* *Ensembler* attempts weighted combinations of base models

Base models
***********

.. _base_model:

    `Fit, predict and validate with a single button.`

Base models are the fundamental bricks of the optimization process. A base model
is a combination of a model class, a set of parameters and a set of features.

Base models implement a versatile method for predictions, which return predictions
for the training data and for the testing data, as well as the score achieved on
the training data.

The mechanics of this process is similar to a cross-validation, with a slight
difference: the final score is not the mean score of each fold. Instead, the array
of predictions is built iteratively and then fully compared to the target column.
More precisely:

1. Filter training and testing features
2. Split the training data in N folds
3. For each fold:
    - Train the model on the bigger part
    - Make predictions for the smaller part
    - Make predictions for the testing dataset
4. Compute the score for the entire column of predictions on the training dataset
5. Compute the average of the predictions for the testing dataset

Averaging the predictions for the testing dataset may result in slightly better
accuracies than expected.

.. rubric:: Pipelines

Pipelines can be used as base models when you want to test various ways of
pre-processing your data before fitting it with an estimator.

If that's your case, please check out the :mod:`miraiml.pipeline` module.

Mirai Seeker
************

.. _mirai_seeker:

There can be too many base models in the search space and we may not be able to
afford exhausive searches. Thus, a smart strategy to search good base models is
mandatory.

The engine registers optimization attempts on dataframes called `histories`. These
dataframes have columns for each hyperparameter and each feature, as well as a
column for the reported scores. The values of the hyperparameters' columns are the
values of the hyperparameters themselves. The values of the features' columns are
either 0 or 1, which indicate whether the features were used or not. An example
of history dataframe for a K-NN classifier with three registries would be:

=========== ========== === ====== =====
Hyperparameters        Features   ---
---------------------- ---------- -----
n_neighbors weights    age gender score
=========== ========== === ====== =====
3           'uniform'  1   0      0.82
2           'distance' 0   1      0.76
4           'uniform'  1   1      0.84
=========== ========== === ====== =====

As the history grows, it can be used to generate potentially good base models for
future optimization attempts. Currently, the available strategies to create base
models are:

- Random
    Generates a completely random base model.

- Naive
    The naive strategy iterates over the history columns (except the score) and
    groups the data by the current column values using the `mean` aggregation
    function on the score column. Each value present on the current column can be
    chosen with a probability that is proportional to the mean score from the
    `group by` aggregation.

    For instance, if we aggregate the history dataframe above by the column `age`,
    the mean score of attempts in which the feature `age` was chosen is 0.83 and
    the mean score of the attempts in which the feature `age` was **not** chosen
    is 0.76. Now, we choose to use `age` on the next base model with a probability
    that's proportional to 0.83 and we choose **not to** with a probability that's
    proportional to 0.76.

    It's called `Naive` because it assumes the strong hypothesis that the columns
    of history dataframes affect the score independently.

- Linear Regression
    Uses a simple linear regression to model the score as a function of the other
    history columns. Categorical columns are processed with One-Hot-Encoding. This
    strategy makes `n`/2 guesses and chooses the best guess according to the linear
    regression model, where `n` is the size of the history dataframe.

The strategy is chosen stochastically according to the following priority rule:

    `The random strategy will be chosen with a probability of 0.5. If it's not,
    the other strategies will be chosen with equal probabilities.`

Ensembler
*********

.. _ensemble:

It is possible to combine the predictions of various base models in order to reach
even higher scores. This process is done by computing a straightforward linear
combination of the base models' predictions.

More precisely, suppose we have a set of base models. For each base model :math:`i`,
let :math:`tr_i` and :math:`ts_i` be its predictions for the training and testing
dataset, respectively. The ensemble of the base models is based on a set of
coefficients :math:`w` (weights), for which we can compute the combined predictions
:math:`E_{tr}` and :math:`E_{ts}` for the training and testing datasets, respectively,
according to the formula:

.. math::
    (E_{tr}, E_{ts}) = \left(\frac{\sum w_i tr_i}{\sum w_i},
    \frac{\sum w_i ts_i}{\sum w_i}\right)

With a smart choice of :math:`w`, the score for :math:`E_{tr}` may be better than
the score of any :math:`tr_i`.

Now, the obvious question is: how to find a good :math:`w`? This is where the
concept of `ensembling cycles` comes into play.

An ensembling cycle is an attempt to generate good weights stochastically, based
on the the score of each base model individually. This is done by using `triangular
distributions <https://en.wikipedia.org/wiki/Triangular_distribution>`_.

The weight of the best base model is drawn from the triangular distribution that
varies from 0 to 1, with mode 1.

For every other base model :math:`i` (not a base model with the highest score),
the weight is drawn from a triangular distribution that varies from 0 to `range`,
with mode 0. It means that its weight will most likely be close to 0 and its
upperbound is defined by the `range` variable.

The value of `range` should depend on the relative score of the base model. But
preventing it from reaching 1 would be too prohibitive. A solution for this is:
`range` is chosen from a triangular distribution that varies from 0 to 1, with mode
`normalized`. The variable `normalized` measures the relative quality of the base
model.

The value of `normalized` is computed by the formula :math:`(s_i-s_\textrm{min})/
(s_\textrm{max}-s_\textrm{min})`, where :math:`s_i` is the score of the base model
and :math:`s_\textrm{min}` and :math:`s_\textrm{max}` are the scores of the worst
and the best base models, respectively.

In the end, bad base models can still influence the ensemble, but their
probabilities of having high weights are relatively low.

The number of ensembling cycles depend on the time consumed by the other models.
The current rule is:

    `The time consumed by the ensemble is limited by the total time consumed by
    all base models, on average.`


Optimization workflow
---------------------

The optimization cycle starts by looking for better base models for each search
space. Mirai Seeker is responsible for keeping track of old base models attempts
in order to provide good guesses for new attempts. If a better base model is found
for some search space, the ensembler output is updated with the new predictions.
Then, after a new solution is attempted for each search space, the Engine executes
the ensembling cycles, looking for better ensembling weights.

Wrapping it all up, the following diagram represents the workflow within an
optimization loop:

.. https://docs.google.com/drawings/d/1C1fwMzYXkawVbn_jloLIX_VNI_jl2bwq8wR3ogCckaQ/edit?usp=sharing
.. image:: https://docs.google.com/drawings/d/e/2PACX-1vQP_qMIXETTJo7h04IfcHA9_N_GaO0hGZueBXbkpJcz1Of3cdZSaVkJejl4EKHIzDxDSVk2IPgGW7sh/pub?w=1689&h=797
