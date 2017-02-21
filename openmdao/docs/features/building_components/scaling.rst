:orphan:

.. _scaling:

Scaling variables
=================

As we saw in <Section declaring_variables>, we can specify scaling parameters for outputs and residuals.
Scaling can be important for the efficiency of some linear solvers.
Knowing when and how to use scaling can be tricky, but in general, it is a good idea to scale outputs and residuals so that both have values that are :math:`\mathcal{O}(1)`.

Basics
------

For outputs, scaling can be specified using the :code:`ref` argument to :code:`add_output`.
This argument is named as such because it represents the reference value.
As the examples below show, scaling can be useful when the output value is very small or very large.

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
On the other hand, :code:`ref` represents the physical value when the scaled value is 1.

  ============  ============  ==============  ================
  :code:`ref`   :code:`ref0`  Physical value  Normalized value
  ============  ============  ==============  ================
  10001         10000         9999.5          -0.5
  --            --            10000.0         0.0
  --            --            10001.0         1.0
  --            --            10003.2         3.2
  ============  ============  ==============  ================

Residual scaling works the same way with :code:`res_ref` and :code:`res_ref0`.

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
