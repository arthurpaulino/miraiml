"""
:mod:`miraiml` provides the following components:

- :class:`miraiml.HyperSearchSpace` represents the search space of hyperparameters
- :class:`miraiml.Config` defines the general behavior of :class:`miraiml.Engine`
- :class:`miraiml.Engine` manages the optimization process
- :mod:`miraiml.extras` has some cool features

You can import them by doing

>>> from miraiml import HyperSearchSpace, Config, Engine, extras
"""

__version__ = '2.2.0'
__all__ = ['HyperSearchSpace', 'Config', 'Engine', 'extras']

from .main import HyperSearchSpace, Config, Engine
from . import extras
