.. _set-and-get-variables:

***************************************
Setting and Getting Component Variables
***************************************

You will both set and get the values in the dimensional and unscaled form via the
:ref:`Problem <openmdao.core.problem.py>` class.
If you have promoted both inputs and outputs to the same name,
then the output takes precedence and it determines the units you should work in.


Outputs and Independent Variables
---------------------------------

To set or get the output variable, you reference it by its promoted name.
In the regular :ref:`Sellar <openmdao.test_suite.components.sellar.py>` problem, all the variables
have been promoted to the top of the model.
So to get the value of the "y1" output defined in the
:ref:`SellarDis1WithDerivatives <openmdao.test_suite.components.sellar.py>` component, you would do
the following:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_promoted_sellar_set_get_outputs
    :layout: interleave


You use the same syntax when working with the independent variables of your problem.
Independent variables can be used as design variables by a
:ref:`Driver <openmdao.core.driver.py>` or set directly by a user.
OpenMDAO requires that every input variable must have a source.  The ultimate source for any
flow of data in an OpenMDAO model is a special component,
:ref:`IndepVarComp <openmdao.core.indepvarcomp.py>`, that does not have any inputs.  You can
leave some of your inputs unconnected and OpenMDAO will automatically create an
:ref:`IndepVarComp <openmdao.core.indepvarcomp.py>` called `_auto_ivc` with an output that connects
to each input, or you can create your own :ref:`IndepVarComp <openmdao.core.indepvarcomp.py>` manually.
For example, consider our paraboloid tutorial problem which has two independent
variables: `x` and `y`.

These could be defined manually and set as follows:

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_set_indeps
    :layout: interleave


or, the inputs `x` and `y` could be left unconnected and OpenMDAO would connect them to
outputs on `_auto_ivc`.  The names of the output variables on `_auto_ivc` are sequentially
named as they are created and are of the form `v0`, `v1`, etc., but you don't really need to know
those names.  You can just interact with the inputs that those outputs are connected to
and the framework will ensure that the proper values are set into the outputs (and into any other
inputs connected to the same output).  Here's what the paraboloid tutorial problem looks like
without declaring the :ref:`IndepVarComps <openmdao.core.indepvarcomp.py>`:


.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_set_indeps_auto
    :layout: interleave


As we said above, outputs are always referenced via their promoted name.
So if you built the Sellar problem using connections
(see :ref:`SellarDerivativesConnected <openmdao.test_suite.components.sellar.py>`),
instead of promoting everything, then you would access the variables like this:


.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_not_promoted_sellar_set_get_outputs
    :layout: interleave


Working with Array Variables
----------------------------

When you have an array variable, for convenience we allow you to set the value with any
properly-sized array, list, or tuple.
In other words, the shape of the list has to match the shape of the actual data.


.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_set_get_array
    :layout: interleave

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_set_2d_array
    :layout: interleave



Residuals
---------

If you want to look at the residual values associated with any particular output variable, you will
reference them using the same naming conventions the outputs.
Also like outputs, you will be given the residuals in the unscaled dimensional form.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_residuals
    :layout: interleave


Inputs
------

You can get or set the value of an input variable using either its promoted name or its absolute
name. If you reference it by its promoted name, however, and that
input is connected to an output because the input and output are promoted to the same name, then
the promoted name will be interpreted as that of the output, and the units will be assumed to be
those of the output as well.  If the input has not been connected to an output then the framework
will connect it automatically to an output of `_auto_ivc`.  In this case, setting or getting using
the input name will cause the framework to assume the units are those of the input, assuming
there is no abiguity in units for example.


Connected Inputs Without a Source
=================================

If multiple inputs have been promoted to the same name but *not* connected manually to an output or promoted
to the same name as an output, then again the framework will connect all of those inputs to an
`_auto_ivc` output.  If, however, there is any difference between the units or values of any of those inputs,
then you must tell the framework what units and/or values to use when creating the corresponding
`_auto_ivc` output.  You do this by calling the `set_input_defaults` function using the promoted
input name on a Group that contains all of the promoted inputs.


.. automethod:: openmdao.core.group.Group.set_input_defaults
    :noindex:

Below is an example of what you'll see if you do *not* call `set_input_defaults` to disambiguate
your units and/or values:

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_with_units_diff_err
    :layout: interleave


The next example shows a successful run after calling `set_input_defaults`:

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_with_units_diff
    :layout: interleave


Another possible scenario is to have multiple inputs promoted to the same name when those inputs have
different units, but then connecting them manually to an output using the :code:`connect` function.
In this case, the framework will not raise an exception during setup if `set_input_defaults` was not
called as it does in the case of multiple promoted inputs that connected to `_auto_ivc`.  However,
if the user attempts to set or get the input using the promoted name, the framework *will* raise an
exception if `set_input_defaults` has not been called to disambiguate the units of the promoted
input.  The reason for this difference is that in the unconnected case, the framework won't know
what value and units to assign to the `_auto_ivc` output if they're ambiguous.  In the manually
connected case, the value and units of the output have already been supplied by the user, and
the only time that there's an ambiguity is if the user tries to access the inputs using their
promoted name.

Specifying Units
----------------

You can also set an input or request the value of any variable in a different unit than its declared
unit, and OpenMDAO will
perform the conversion for you. This is done with the `Problem` methods `get_val` and `set_val`.

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_with_units
    :layout: interleave

When dealing with arrays, you can set or get specific indices or index ranges by adding the "indices"
argument to the calls:

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_array_with_units
    :layout: interleave

An alternate method of specifying the indices is by making use of the :code:`slicer` object. This
object serves as a
helper function allowing the user to specify the indices value using the same syntax as you would when
accessing a numpy array. This example shows that usage.

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_array_with_slicer
    :layout: interleave


Retrieving Remote Variables
---------------------------

If you're running under MPI, the `Problem.get_val` method also has a *get_remote* arg that allows
you to get the value of a variable even if it's not local to the current MPI process.  For example,
the code below will retrieve the value of `foo.bar.x` in all processes, whether the variable is
local or not.


.. code-block:: python

    val = prob.get_val('foo.bar.x', get_remote=True)


.. warning::

    If `get_remote` is True, `get_val` makes a collective MPI call, so make sure to call it
    in *all* ranks of the Problem's MPI communicator.  Otherwise, collective calls made
    in different ranks will get out of sync and result in cryptic MPI errors.



Testing if a Variable or System is Local
----------------------------------------

If you want to know if a given variable or system is local to the current process, the
`Problem.is_local` method will tell you.  For example:

.. code-block:: python

    if prob.is_local('foo.bar.x'):
        print("foo.bar.x is local!")


.. tags:: SetGet
