from .util import is_valid_pipeline_name
from .core import BasePipelineClass


def compose_pipeline_class(class_name, steps):
    """
    Builds a pipeline class that can be instantiated with particular parameters
    for each of its transformers/estimator without needing to call ``set_params``
    as you would do with scikit-learn's Pipeline when performing hyperparameters
    optimizations.

    As this function returns a ``type``, you can create as many classes as you
    want as long as you provide a different ``class_name`` for each of them,
    otherwise the definitions would overlap and past definitions would be
    overwritten.

    Similarly to scikit-learn's Pipeline, ``steps`` is a list of tuples
    containing an alias and the respective pipeline element. Although, since
    this function is a class factory, you shouldn't instantiate the
    transformer/estimator as you would do with scikit-learn's Pipeline. Thus,
    this is how ``compose_pipeline_class`` should be called:

    :Example:

    ::

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from miraiml.extra import compose_pipeline_class

        MyPipelineClass = compose_pipeline_class(
            class_name = 'MyPipelineClass',
            steps = [
                ('scaler', StandardScaler), # StandardScaler instead of StandardScaler()
                ('rfc', RandomForestClassifier) # No instantiation either
            ]
        )

    And then, in order to instantiate ``MyPipelineClass`` with the desired
    parameters, you just need to refer to them as a concatenation of their
    respective class aliases and their names, separated by ``'__'``.

    :Example:

    ::

        pipeline = MyPipelineClass(scaler__with_mean=False, rfc__max_depth=3)

    **The purpose** of such pipeline classes is that they can work as base
    models to build instances of :class:`HyperSearchSpace`.

    :Example:

    ::

        hyper_search_space = HyperSearchSpace(
            model_class=MyPipelineClass,
            id='MyPipelineClass',
            parameters_values=dict(
                scaler__with_mean=[True, False],
                scaler__with_std=[True, False],
                rfc__max_depth=[3, 4, 5, 6]
            )
        )

    .. warning::
        None of the strings used to compose pipeline classes can start with
        numbers or contain ``'__'``. Also, repeated aliases are not allowed.

    .. note::
        You can check the documentation for the class from which the returned
        class inherits here: :class:`miraiml.core.BasePipelineClass`.

    :rtype: type
    :returns: The composed pipeline class

    :raises: ``ValueError``, ``TypeError``, ``NotImplementedError``
    """

    def validate_type_and_content(name):
        if not isinstance(name, str):
            raise TypeError('{} is not a string'.format(name))
        if not is_valid_pipeline_name(name):
            raise ValueError('{} is not allowed'.format(name))

    validate_type_and_content(class_name)

    aliases = []

    for alias, class_type in steps:
        validate_type_and_content(alias)

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

    return type(class_name, (BasePipelineClass,), dict(steps=steps))
