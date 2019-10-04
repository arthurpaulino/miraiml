"""
:mod:`miraiml` provides the following components:

- :class:`miraiml.HyperSearchSpace` represents the search space for a base model
- :class:`miraiml.Config` defines the general behavior for :class:`miraiml.Engine`
- :class:`miraiml.Engine` manages the optimization process
- :mod:`miraiml.pipeline` has some features related to pipelines
"""

__version__ = '2.2.0'
__all__ = ['HyperSearchSpace', 'Config', 'Engine', 'pipeline']

from .main import HyperSearchSpace, Config, Engine
from . import pipeline
