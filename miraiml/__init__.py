"""
:mod:`miraiml` provides the following classes:

- :class:`miraiml.SearchSpace` represents the search space for the optimization
- :class:`miraiml.Config` defines the general behavior of
  :class:`miraiml.Engine`
- :class:`miraiml.Engine` manages the optimization process

You can import them by doing

>>> from miraiml import SearchSpace, Config, Engine
"""

from .main import SearchSpace, Config, Engine

__all__ = ['SearchSpace', 'Config', 'Engine']

__version__ = '1.0.3'
