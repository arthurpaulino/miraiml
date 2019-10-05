The User's API
==============

.. automodule:: miraiml

miraiml.HyperSearchSpace
------------------------

.. autoclass:: miraiml.HyperSearchSpace
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
        load_data
        shuffle_train_data
        reconfigure
        restart
        request_status
        extract_model

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
