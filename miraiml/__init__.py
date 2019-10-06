"""
:mod:`miraiml` provides the following components:

- :class:`miraiml.SearchSpace` represents the search space for a base model
- :class:`miraiml.Config` defines the general behavior for :class:`miraiml.Engine`
- :class:`miraiml.Engine` manages the optimization process
- :mod:`miraiml.pipeline` has some features related to pipelines **(hot!)**
"""

__version__ = '3.0.0'
__all__ = ['SearchSpace', 'Config', 'Engine', 'Status', 'pipeline']

from miraiml.main import SearchSpace, Config, Engine, Status
from miraiml import pipeline
