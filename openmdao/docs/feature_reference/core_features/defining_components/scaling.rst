.. _scale_outputs_and_resids:

*****************
Scaling Variables
*****************

As we saw in <Section declaring_variables>, we can specify scaling parameters for outputs and residuals.
Scaling can be important for the efficiency of some linear solvers and can have an impact on some gradient free
nonlinear solvers such as Broyden. Knowing when and how to use scaling can be tricky, but in general, it is a good
idea to scale outputs and residuals so that both have values that are :math:`\mathcal{O}(1)` and so that they have
roughly the same range of variation in your design space.

For example, consider a value that is expected to have a value around 2500. Then you might scale it by dividing
by 1000. However, if the value was going to have an expected range between 2400 and 2500, then you might want to subtract out
2400 then divide by 100.

OpenMDAO supports this kind of linear scaling of both output and residual values through a set of user defined reference
values specified when the variables are defined. These are described below.

.. note::

    When you apply scaling to your variables, it does not affect the inputs and outputs you work with in your components.
    These are still worked with in physical, dimensional quantities. The scaling is only applied internally when values
    are given to solvers.

Basics
------

For outputs, scaling can be specified using the :code:`ref` argument to :code:`add_output`.
This argument is named as such because it represents the reference physical value that will be scaled to 1.
The table below shows some example physical values and their scaled values for the a given :code:`ref`.

  ============  ==============  ================
  :code:`ref`   Physical value  Normalized value
  ============  ==============  ================
  10000         0               0.0
  --            10000           1.0
  --            25000           2.5
  0.0001        0.0000          0.0
  --            0.0001          1.0
  --            0.0003          3.0
  ============  ==============  ================

For residuals, scaling works the same way, except that the argument to :code:`add_output` is :code:`res_ref`.

Scaling with an offset
----------------------

It can be desirable to scale with an offset when the variable values are very large but they only vary by a small amount.
In these situations, we can specify a second argument, :code:`ref0`, to :code:`add_output`.
This argument is named as such because :code:`ref0` represents the physical value when the scaled value is 0.


  ============  ============  ==============  ================
  :code:`ref`   :code:`ref0`  Physical value  Normalized value
  ============  ============  ==============  ================
  10001         10000         9999.5          -0.5
  --            --            10000.0         0.0
  --            --            10001.0         1.0
  --            --            10003.2         3.2
  ============  ============  ==============  ================

Residual scaling works the same way with :code:`res_ref`, though there is no offset for residuals.
In explicit components, :code:`res_ref` defaults to :code:`ref`.

Using scaling with units
------------------------

Now, we address the situation in which we use scaling in conjunction with units.
Let us say we specify to :code:`add_output` the :code:`units` argument along with :code:`ref` and :code:`ref0`.
Then, the values pass in for :code:`ref` and :code:`ref0` are assumed to be in the units given by the :code:`units` argument.
For instance, if :code:`ref=10001.` and :code:`units='Pa'`, then a scaled value of 1 represents :code:`10001. Pa`.

  ==============  ============  ============  ==============  ================
  :code:`units`   :code:`ref`   :code:`ref0`  Physical value  Normalized value
  ==============  ============  ============  ==============  ================
  kPa             100           --            0 kPa           0.0
  --              --            --            100 kPa         1.0
  --              --            --            250 kPa         2.5
  Pa              100100        100000        99900 Pa        -0.1
  --              --            --            100000 Pa       0.0
  --              --            --            100100 Pa       0.1
  ==============  ============  ============  ==============  ================


.. note::

      residual scaling is separate and independent of output scaling in implicit components.
      In explicit components, the requested output scaling is applied to the residuals as well
      unless :code:`res_ref` is also specified.

Specifying a scaler on an output
--------------------------------

This example shows how to specify a scaler on outputs 'y1' and 'y2'. The scaling used here
assures that the outputs (which are states in this implicit component) are in the same order
of magnitude when the solver interacts with them.  Note that whenver a user function is called
(like `apply_nonlinear` here), all outputs and residuals are reverted to unscaled dimensiional
form.

  .. embed-code::
      openmdao.core.tests.test_scaling.ScalingExample1

Specifying a scaler and offset on an output
-------------------------------------------

This example shows how to specify a scaler and an offset on outputs 'y1' and 'y2'.

  .. embed-code::
      openmdao.core.tests.test_scaling.ScalingExample2

Specifying a scaler on a residual
---------------------------------

This example shows how to specify a scaler on the residuals for variables 'y1' and 'y2'.
This choice of scaler values assures that the residuals are of the same order of magnitude when
the solver interacts with them.

  .. embed-code::
      openmdao.core.tests.test_scaling.ScalingExample3

Specifying a vector of scalers
------------------------------

When you have a vector output, you can also specify a vector scaling factor with individually
selected elements.  For this, the `ref`, `ref0` or `res_ref` must have the same shape as the
variable value.

  .. embed-code::
      openmdao.core.tests.test_scaling.ScalingExampleVector

.. tags:: Scaling, Outputs