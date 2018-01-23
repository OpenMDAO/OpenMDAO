.. _units:

Specifying Units for Variables
==============================

As we saw in :ref:`declaring-variables`, we can specify units for inputs, outputs, and residuals.
There is a :code:`units` argument on :code:`add_input` to specify input units,
and there are :code:`units` and :code:`res_units` arguments on :code:`add_output` to specify output and residual units, respectively.
A complete listing of all available units is given :ref:`here <feature_units>`.

.. note::

    Residual units, if not specified, default to the same units as the output variable.
    :code:`res_units` is very rarely specified.

Specifying units has the following result:

1. Unit conversions occur during data passing.
For instance, let us say we have a :code:`TimeComp` component that outputs :code:`time1` in hours and a :code:`SpeedComp` component takes :code:`time2` as an input but in seconds.
If we connect :code:`TimeComp.time1` to :code:`SpeedComp.time2` with hours/seconds specified during the corresponding :code:`add_output`/:code:`add_input` calls, then OpenMDAO automatically converts from hours to seconds.

2. The user always gets/sets the variable in the specified units.
Declaring an input, output, or residual to have certain units means that any value 'set' into the variable is assumed to be in the given units and any time the user asks to 'get' the variable, the value is return in the given units.
This is the case not only in <Component> methods such as :code:`compute`, :code:`apply_nonlinear`, and :code:`apply_linear`, but everywhere, including the user's run script.

3. In :code:`add_input` and :code:`add_output`, all arguments are assumed given in the specified units.
In the case of :code:`add_input`, if :code:`units` is specified, then :code:`val` is assumed to be given in those units.
In the case of :code:`add_output`, if :code:`units` is specified, then :code:`val`, :code:`lower`, :code:`upper`, :code:`ref`, and :code:`ref0` are all assumed to be given in those units.
Also in :code:`add_output`, if :code:`res_units` is specified, then :code:`res_ref` is assumed to be given in :code:`res_units`.

Units syntax
------------
Units are specified as a string that adheres to the following syntax.
The string is a composition of numbers and known units that are combined with multiplication (:code:`*`), division (:code:`/`), and exponentiation (:code:`**`) operators.
The known units can be prefixed by kilo (`k`), Mega (`M`), and so on.
The list of units and valid prefixes can be found in the :ref:`units library <feature_units>`.

For illustration, each of the following is a valid unit string representing the same quantity:

- :code:`N`
- :code:`0.224809 * lbf`
- :code:`kg * m / s ** 2`
- :code:`kg * m * s ** -2`
- :code:`kkg * mm / s ** 2`

.. note::

    If units are not specified or are specified as :code:`None` then the variable
    is a assumed to be unitless.  If such a variable is connected to a variable
    with units, the connection will be allowed but a warning will be issued.

Example
-------

This example illustrates how we can compute speed from distance and time given in :code:`km` and :code:`h` using a component that computes speed using :code:`m` and :code:`s`.

We first define the component.

.. embed-code::
    openmdao.core.tests.test_units.SpeedComp

In the overall problem, the first component, :code:`c1`, defines distance and time in :code:`m` and :code:`s`.
OpenMDAO handles the unit conversions when passing these two variables into :code:`c2`, our 'SpeedComp'.
There is a further unit conversion from :code:`c2` to :code:`c3` since speed must be converted now to :code:`m/s`.

.. embed-test::
    openmdao.core.tests.test_units.TestUnitConversion.test_speed
