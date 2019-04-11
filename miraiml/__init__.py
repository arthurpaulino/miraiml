"""
Provides the classes to use MiraiML.

- :class:`miraiml.SearchSpace` represents the search space for instances of
  :class:`miraiml.core.BaseModel`
- :class:`miraiml.Config` defines the general behavior of
  :class:`miraiml.Engine`
- :class:`miraiml.Engine` manages the optimization process

You can import them by doing

>>> from miraiml import SearchSpace, Config, Engine
"""

from .main import SearchSpace, Config, Engine

__all__ = ['SearchSpace', 'Config', 'Engine']
