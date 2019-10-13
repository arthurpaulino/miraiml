"""
:mod:`miraiml.pipeline` contains a function that lets you build your own
pipeline classes. It also contains a few pre-defined pipelines for baselines.
"""

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression

from miraiml.util import is_valid_pipeline_name
from miraiml.core import BasePipelineClass


def compose(steps):
    """
    A function that defines pipeline classes dinamically. It builds a pipeline
    class that can be instantiated with particular parameters for each of its
    transformers/estimator without needing to call ``set_params`` as you would
    do with scikit-learn's Pipeline when performing hyperparameters optimizations.

    Similarly to scikit-learn's Pipeline, ``steps`` is a list of tuples
    containing an alias and the respective pipeline element. Although, since
    this function is a class factory, you shouldn't instantiate the
    transformer/estimator as you would do with scikit-learn's Pipeline. Thus,
    this is how :func:`compose` should be called:

    ::

        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.preprocessing import StandardScaler

        >>> from miraiml.pipeline import compose

        >>> MyPipelineClass = compose(
        ...     steps = [
        ...         ('scaler', StandardScaler), # StandardScaler instead of StandardScaler()
        ...         ('rfc', RandomForestClassifier) # No instantiation either
        ...     ]
        ... )

    And then, in order to instantiate ``MyPipelineClass`` with the desired
    parameters, you just need to refer to them as a concatenation of their
    respective class aliases and their names, separated by ``'__'``.

    ::

        >>> pipeline = MyPipelineClass(scaler__with_mean=False, rfc__max_depth=3)

    If you want to know which parameters you're allowed to play with, just call
    ``get_params``:

    ::

        >>> params = pipeline.get_params()
        >>> print("\\n".join(params))
        scaler__with_mean
        scaler__with_std
        rfc__bootstrap
        rfc__class_weight
        rfc__criterion
        rfc__max_depth
        rfc__max_features
        rfc__max_leaf_nodes
        rfc__min_impurity_decrease
        rfc__min_impurity_split
        rfc__min_samples_leaf
        rfc__min_samples_split
        rfc__min_weight_fraction_leaf
        rfc__n_estimators
        rfc__n_jobs
        rfc__oob_score
        rfc__random_state
        rfc__verbose
        rfc__warm_start

    You can check the available methods for your instantiated pipelines on the
    documentation for :class:`miraiml.core.BasePipelineClass`, which is the
    class from which the composed classes inherit from.

    **The intended purpose** of such pipeline classes is that they can work as
    base models to build instances of :class:`miraiml.SearchSpace`.

    ::

        >>> from miraiml import SearchSpace

        >>> search_space = SearchSpace(
        ...     id='MyPipelineClass',
        ...     model_class=MyPipelineClass,
        ...     parameters_values=dict(
        ...         scaler__with_mean=[True, False],
        ...         scaler__with_std=[True, False],
        ...         rfc__max_depth=[3, 4, 5, 6]
        ...     )
        ... )

    :type steps: list
    :param steps: The list of pairs (alias, class) to define the pipeline.

        .. warning::
            Repeated aliases are not allowed and none of the aliases can start
            with numbers or contain ``'__'``.

            The classes used to compose a pipeline **must** implement ``get_params``
            and ``set_params``, such as scikit-learn's classes, or :func:`compose`
            **will break**.

    :rtype: type
    :returns: The composed pipeline class

    :raises: ``TypeError`` if an alias is not a string.

    :raises: ``ValueError`` if an alias has an invalid name.

    :raises: ``NotImplementedError`` if some class of the pipeline does not implement
        the required methods.
    """

    aliases = []

    for alias, class_type in steps:
        if not isinstance(alias, str):
            raise TypeError('{} is not a string'.format(alias))

        if not is_valid_pipeline_name(alias):
            raise ValueError('{} is not allowed for an alias'.format(alias))

        class_content = dir(class_type)

        if 'fit' not in class_content:
            raise NotImplementedError('{} must implement fit'.format(class_type.__name__))

        aliases.append(alias)

        if len(aliases) < len(steps):
            if 'transform' not in class_content:
                raise NotImplementedError(
                    '{} must implement transform'.format(class_type.__name__)
                )
        else:
            if 'predict' not in class_content and 'predict_proba' not in class_content:
                raise NotImplementedError(
                    '{} must implement predict or predict_proba'.format(class_type.__name__)
                )

    if len(set(aliases)) != len(aliases):
        raise ValueError('Repeated aliases are not allowed')

    return type('MiraiPipeline', (BasePipelineClass,), dict(steps=steps))


__initial_steps__ = [
    ('ohe', OneHotEncoder),
    ('impute', SimpleImputer),
    ('min_max', MinMaxScaler)
]


class NaiveBayesBaseliner(compose(__initial_steps__ + [('naive', GaussianNB)])):
    """
    This is a baseline pipeline for classification problems. It's composed by
    the following transformers/estimator:

    1. ``sklearn.preprocessing.OneHotEncoder``
    2. ``sklearn.impute.SimpleImputer``
    3. ``sklearn.preprocessing.MinMaxScaler``
    4. ``sklearn.naive_bayes.GaussianNB``

    The available parameters to tweak are:

    ::

        >>> from miraiml.pipeline import NaiveBayesBaseliner

        >>> for param in NaiveBayesBaseliner().get_params():
        ...     print(param)
        ...
        ohe__categorical_features
        ohe__categories
        ohe__drop
        ohe__dtype
        ohe__handle_unknown
        ohe__n_values
        ohe__sparse
        impute__add_indicator
        impute__fill_value
        impute__missing_values
        impute__strategy
        impute__verbose
        min_max__feature_range
        naive__priors
        naive__var_smoothing
    """
    def __init__(self):
        super().__init__()


class LinearRegressionBaseliner(compose(__initial_steps__ + [('lin_reg', LinearRegression)])):
    """
    This is a baseline pipeline for regression problems. It's composed by the
    following transformers/estimator:

    1. ``sklearn.preprocessing.OneHotEncoder``
    2. ``sklearn.impute.SimpleImputer``
    3. ``sklearn.preprocessing.MinMaxScaler``
    4. ``sklearn.linear_model.LinearRegression``

    The available parameters to tweak are:

    ::

        >>> from miraiml.pipeline import LinearRegressionBaseliner

        >>> for param in LinearRegressionBaseliner().get_params():
        ...     print(param)
        ...
        ohe__categorical_features
        ohe__categories
        ohe__drop
        ohe__dtype
        ohe__handle_unknown
        ohe__n_values
        ohe__sparse
        impute__add_indicator
        impute__fill_value
        impute__missing_values
        impute__strategy
        impute__verbose
        min_max__feature_range
        lin_reg__fit_intercept
        lin_reg__n_jobs
        lin_reg__normalize
    """
    def __init__(self):
        super().__init__()
