[![][docs_img]][docs_proj]

# MiraiML

> Mirai: `future` in japanese.

MiraiML is an asynchronous engine for autonomous & continuous machine learning,
built for real-time usage.

MiraiML attempts to improve the chosen metric by searching good hyperparameters
and sets of features to feed statistical models.

Even though the engine is autonomous, it offers (and requires) a certain degree
of configuration, in a way that the user has good control over what's happening
behind the scenes.

Some didactic tutorials can be found on the [example](example) folder. Or [Read
the Docs][docs_url] for a better understanding of MiraiML's potential.

## Usage

1. Download: ``$ git clone https://github.com/arthurpaulino/miraiml.git``
2. Change directory: ``$ cd miraiml``
3. Install: ``$ python setup.py install``
4. Now, inside a Python environment, import the main classes:

```python
>>> from miraiml import BaseLayout, Config, Engine
```

## Contributing

Please, follow the `guidelines <CONTRIBUTING.md>`_ if you want to be part of this
project.

[docs_img]: https://readthedocs.org/projects/miraiml/badge/?version=latest
[docs_proj]: https://readthedocs.org/projects/miraiml/
[docs_url]: https://miraiml.readthedocs.io/en/latest/
