.. -*- mode: rst -*-

|docs|_

.. |docs| image:: https://readthedocs.org/projects/miraiml/badge/?version=latest
.. _docs: https://readthedocs.org/projects/miraiml/

MiraiML
=======

    Mirai: `future` in japanese.

MiraiML is an asynchronous engine for autonomous & continuous machine learning,
built for real-time usage.

MiraiML attempts to improve the chosen metric by searching good hyperparameters
and sets of features to feed statistical models.

Even though the engine is autonomous, it offers (and requires) a certain degree
of configuration, in a way that the user has good control over what's happening
behind the scenes.

A step-by-step tutorial can be found on the `example notebook <example.ipynb>`_.
Or `read the docs <https://miraiml.readthedocs.io/en/latest/>`_ for further
understanding of MiraiML's potential.

Usage
-----

1. Download: ``$ git clone https://github.com/arthurpaulino/miraiml.git``
2. Change directory: ``$ cd miraiml``
3. Install: ``$ python setup.py install``
4. Inside a Python environment, import the main classes:

>>> from miraiml import BaseLayout, Config, Engine

To do
-----

- Add a collaboration tutorial
- Autogenerate html reports
- Infer ``MiraiSeeker`` scores for smarter guesses
- Document an `User Guide`
- Create more examples
- Validate constructors' parameters
- Protect engine from trying to save files with invalid characters from ids
