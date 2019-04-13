[![][docs_img]][docs_proj]

# MiraiML

> Mirai: _future_ in japanese.

MiraiML is an asynchronous engine for autonomous & continuous machine learning,
built for real-time usage.

## Usage

1. Download: `$ git clone https://github.com/arthurpaulino/miraiml.git`
2. Change directory: `$ cd miraiml`
3. Install: `$ make install` or `$ python setup.py install`
   Optional: remove the local `build` directory
4. Now, inside a Python environment, import the main classes:

```python
>>> from miraiml import SearchSpace, Config, Engine
```

Some didactic tutorials can be found in the [example](examples) directory. Or
[Read the Docs][docs_url] for a better understanding of MiraiML.

**Note**: For more info on `make` directives, call `$ make help`

## Contributing

Please, follow the [guidelines](CONTRIBUTING.md) if you want to be part of this
project.

[docs_img]: https://readthedocs.org/projects/miraiml/badge/?version=latest
[docs_proj]: https://readthedocs.org/projects/miraiml/
[docs_url]: https://miraiml.readthedocs.io/en/latest/
