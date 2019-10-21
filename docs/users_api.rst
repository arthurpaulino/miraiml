The User's API
==============

.. automodule:: miraiml

miraiml.SearchSpace
------------------------

.. autoclass:: miraiml.SearchSpace
    :members:

miraiml.Config
--------------

.. autoclass:: miraiml.Config
    :members:

miraiml.Engine
--------------

.. autoclass:: miraiml.Engine
    :members:

    .. autosummary::
        :nosignatures:

        is_running
        interrupt
        load_train_data
        load_test_data
        shuffle_train_data
        reconfigure
        restart
        request_status

miraiml.Status
--------------

.. autoclass:: miraiml.Status
    :members:

miraiml.pipeline
----------------

.. automodule:: miraiml.pipeline
    :members:

    .. autosummary::
        :nosignatures:

        compose
        NaiveBayesBaseliner
        LinearRegressionBaseliner
