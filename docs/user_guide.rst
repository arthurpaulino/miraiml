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

For each fold of the training dataset, the model trains on the bigger part and
then make predictions for the smaller part and for the testing dataset. After
iterating over all folds, the predictions for the training dataset will be
complete and there will be ``config.n_folds`` sets of predictions for the testing
dataset. The final set of predictions for the testing dataset is the mean of the
``config.n_folds`` predictions.

This mechanic may produce more stable predictions for the testing dataset than for
the training dataset, resulting in slightly better accuracies than expected.

Seeking good base models
------------------------

.. _mirai_seeker:

For each hyperparameter and feature, its value (True or False for features) is
chosen stochastically depending on the mean score of the registered entries in
which the value was chosen before. Better parameters and features have higher
chances of being chosen.

Ensembling base models
----------------------

.. _ensemble:
