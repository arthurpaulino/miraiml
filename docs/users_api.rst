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

        is_running
        interrupt
        load_data
        shuffle_train_data
        reconfigure
        restart
        request_status
        request_report
        extract_model

miraiml.pipeline
----------------

.. autofunction:: miraiml.pipeline.compose_pipeline_class

.. autoclass:: miraiml.pipeline.Test
