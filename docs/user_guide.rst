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
of predictions is built incrementally and then fully compared to the target column.
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

Currently, the available strategies are:

- Random
    Generates a completely random sets of hyperparameters and features.
- Naive
    On the history of tested base models, hyperparameters can assume their
    respective values and features can assume the value 1 if they were present
    on the validation and 0 otherwise.

    The naive strategy iterates over each history column (except the score) and
    performs a `group by` using the `mean` aggregation function on the score.
    Each value present on the current column can be chosen with a probability
    that is proportional to the score from the `group by` aggregation.

Ensembling base models
----------------------

.. _ensemble:

It is possible to combine the predictions of various base models in order to reach
even higher scores. This process is done by computing a straightforward linear
combination of the base models' predictions. The score of the ensemble is computed
on the training target and the same linear combination is performed on the
predictions for the testing dataset.

Now, the obvious question is: how to find smart coefficients (or weights) for the
linear combination? This is where the concept of `ensembling cycles` comes into
play.

An ensembling cycle is an attempt to generate good weights based on the the score
of each base model individually. This is done by using `triangular distributions
<https://en.wikipedia.org/wiki/Triangular_distribution>`_.

The weight of the best base model is drawn from the triangular distribution that
varies from 0 to 1, with mode 1.

For the other base models :math:`i`, the weight is drawn from triangular
distributions that varies from 0 to `range`, with mode 0. `range` is chosen from
a triangular distribution that varies from 0 to 1, with mode `normalized`.

`normalized` is computed by the formula :math:`(s_i-s_\textrm{min})/
(s_\textrm{max}-s_\textrm{min})`, where :math:`s_i` is the score of the current
base model, :math:`s_\textrm{min}` and :math:`s_\textrm{max}` are the scores of
the worst and the best base models, respectively.

It means that bad base models can still influence the ensemble, but their
probabilities of having high weights are relatively low if compared to better
base models.
