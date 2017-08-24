.. _setup:

****************
Setup Your Model
****************

After you have built up a model by defining variables and components, then organizing them into a hierarchy, and connecting them together, \
you need to call the :code:`setup()` method to have the framework do some initialization work in preparation for execution.
You can control some details of that initialization with the arguments that you pass into :code:`setup()`,
and it is important to note that you can not set or get any variable values nor run until **after** you call :code:`setup()`.

.. automethod:: openmdao.core.problem.Problem.setup
    :noindex:

Related Features
-------------------
:ref:`Set/Get Variables<set-and-get-variables>`, drivers
