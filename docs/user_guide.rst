MiraiML User Guide
==================

Introduction
------------

    Mirai: `future` in japanese.

MiraiML is an asynchronous engine for continuous & autonomous machine learning,
built for real-time usage.

- It's asynchronous because it runs in background, allowing you to execute custom
  Python code as you interact with the engine;

- It's continuous because it can run "forever", always looking for solutions that
  can achieve better accuracies;

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
spend less time on such mechanical tasks.

MiraiML works on the typical train/test scenario, when the data can fit in the
RAM. No models are provided, thus you need to import external models or implement
your own. Didactic tutorials can be found on the
`examples <https://github.com/arthurpaulino/miraiml/tree/master/examples>`_
directory.

Base models: fitting, predicting and scoring
--------------------------------------------

.. _base_model:

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
4. Compute the score for the entire column of predictions
5. Compute the average of the predictions for the testing dataset

Averaging the predictions for the testing dataset may result in slightly better
accuracies than expected.

Seeking good base models
------------------------

.. _mirai_seeker:

There can be too many base models in the search space and we may not be able to
afford exhausive searches. Thus, a smart strategy to search good base models is
mandatory.

The engine is able to register optimization attempts on dataframes called
`histories`. These dataframes have columns for each hyperparameter and each
feature, as well as a column for the reported score. The values of the
hyperparameters' columns are the values of the hyperparameters themselves. The
values of the features' columns are either 0 or 1, which indicate whether the
features were used or not. An example of history dataframe for a K-NN classifier
with three registries would be:

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
    history columns. Makes `n`/2 guesses and chooses the best guess according to
    the model, where `n` is the size of the history dataframe.

The strategy is chosen stochastically according to the following priority rule:

    `With a probability of 0.5, the random strategy will be chosen. If it's not,
    the other strategies will be chosen with equal probabilities.`

Ensembling base models
----------------------

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

    :math:`(E_{tr}, E_{ts}) = \left(\frac{\sum w_i tr_i}{\sum w_i},
    \frac{\sum w_i ts_i}{\sum w_i}\right)`

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
with mode 0. It means that its weight will most likely be close to 0. The upperbound
is defined by the `range` variable.

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
