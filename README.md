# MiraiML

> Mirai: _future_ in japanese.

MiraiML is an asynchronous engine for autonomous & continuous machine learning,
built for real-time usage.

MiraiML attempts to improve the chosen metric by searching good hyperparameters
and sets of features to feed statistical models.

Even though the engine is autonomous, it offers (and requires) a certain degree
of configuration, in a way that the user has good control over what's happening
behind the scenes.

A step-by-step tutorial can be found on the [example notebook](example.ipynb).
Or [read the docs](https://miraiml.readthedocs.io/en/latest/) for further
understanding.

## To do

- Modularize an Ensembler
- Stacking
- Provide a way for the user to design his own exploration strategies
- Provide rich visualizations for the optimization process
- Provide an `on_improvement` function for the user
- Provide support for multi-class classification problems
- Create unit tests
