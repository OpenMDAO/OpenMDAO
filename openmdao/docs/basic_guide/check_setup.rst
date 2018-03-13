.. _check_setup_tutorial:

---------------------------------------
Sanity Checking Your Model
---------------------------------------

In the first two tutorials we showed you the basics of how to build up a model from a set of components,
group them together, connect them together, and optimize them.

Sometimes you put your model together and things don't work quite the way you would expect.
When this happens, OpenMDAO has a number of :ref:`debugging <debugging_features>` features to help you understand the structure of your model better and sort out the issue.
Many debugging features are all accessed via a :ref:`command line script <om-command>` that is installed along with OpenMDAO itself.
There are a lot of different tools that are accessible from that script, but in this tutorial we'll focus on the most important one:
:ref:`check setup <om-command-check>`.


check setup
----------------

check setup runs through a host of different tests to make sure your model is setup correctly and warn you about things that commonly cause problems.
It will:

    #. identify any unconnected inputs (forgetting to connect things is one of the most common mistakes).
    #. look for any cycles in your model that indicate the need for solvers (did you mean to create that cycle?).
    #. recurse down the model hierarchy and give every group and component a chance to perform its own custom checks.

For example, if you tried to build the :ref:`sellar problem using connections<guide_promote_vs_connect>`,
but forgot to issue one of the connections then your problem wouldn't run correctly and you'd get the wrong answer.

.. embed-code::
    openmdao.test_suite.scripts.sellar

When you see that incorrect answer, the first thing you should do is check your setup via the :ref:`openmdao script <om-command>`.

.. embed-shell-cmd::
    :cmd: openmdao check sellar.py
    :dir: ../test_suite/scripts


This output tells you several things:

    #. You have an unconnected input: `cycle.d1.y2`
    #. There are no reported cycles in your model, but there should be because this is supposed to be a coupled model!


Whenever you encounter a problem, before you look at anything else you should always run this check first and look over the output carefully.
