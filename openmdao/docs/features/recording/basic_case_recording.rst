.. _basic_case_recording:

************************
Basic Recording Example
************************

Basic Recording Example
------------------------

Below is an basic example of how to create a recorder, save the information, and parse through to inspect
the data. `list_outputs` is a quick way to show all of your outputs and their final values, and should you
need to isolate a single value OpenMDAO provides two ways to retrieve them. Using `__getitem__` will give you
the value of the output as is, but if you want to convert that value's units, use `__get_val__`. To view all
the design variables, constraints, and objectives, you can can their methods like the example below.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_basic_case_recording
    :layout: interleave


Generating a Plot with Case Recording Data
-------------------------------------------

It can be useful to see your design variables' paths to convergence. Below we show how to extract the
data from the recorder data and create a basic plot.

.. embed-code::
    openmdao.recorders.tests.test_sqlite_recorder.TestFeatureSqliteRecorder.test_feature_basic_case_plot
    :layout: interleave

.. image:: images/des_var_opt.jpeg
    :width: 600


