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

- It's autonomous because it does not wanders on the search space blindly. Instead,
  it combines past attempts to guide its next steps, always allowing itself to jump
  out of local minima.

MiraiML attempts to improve the chosen metric by searching good hyperparameters
and sets of features for base statistical models, whilst finding smart ways to
combine the predictions of these models in order to achieve even higher scores.

    `But how can MiraiML help me? And how does it even work?`

We're going to address these questions on the subsections to come.

MiraiML usability
-----------------

Base models: fitting, predicting and scoring
--------------------------------------------

Seeking for good base models
----------------------------

Ensembling base models
----------------------
