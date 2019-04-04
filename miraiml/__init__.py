"""
MiraiML is an asynchronous engine for autonomous & continuous machine learning,
built for real-time usage.

MiraiML attempts to improve the chosen metric by searching good hyperparameters
and sets of features to feed statistical models.
"""

from .core import BaseLayout, Config, Engine

__all__ = ['BaseLayout', 'Config', 'Engine']
