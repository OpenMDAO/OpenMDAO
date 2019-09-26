.. _set-and-get-variables:

***************************************
Setting and Getting Component Variables
***************************************

You will both set and get the values in the dimensional and unscaled form via the :ref:`Problem <openmdao.core.problem.py>` class.
If you have promoted both inputs and outputs to the same name,
then the output takes precedence and it determines the units you should work in.


Outputs and Independent Variables
---------------------------------

To set or get the output variable, you reference it by its promoted name.
In the regular :ref:`Sellar <openmdao.test_suite.components.sellar.py>` problem, all the variables have been promoted to the top of the model.
So to get the value of the "y1" output defined in the :ref:`SellarDis1WithDerivatives <openmdao.test_suite.components.sellar.py>` component, you would do the following:

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_promoted_sellar_set_get_outputs
    :layout: interleave


You use the same syntax when working with the independent variables of your problem.
Independent variables hold values set by a user or are used as design variables by a :ref:`Driver <openmdao.core.driver.py>`.
OpenMDAO requires that every variable must have an ultimate source, even independent variables.
We accomplish this by defining independent variables as outputs of a special component,
:ref:`IndepVarComp <openmdao.core.indepvarcomp.py>`, that does not have any inputs.
For example, consider our paraboloid tutorial problem problem which has two independent variables: `x` and `y`.

These would be defined and set as follows:

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_set_indeps
    :layout: interleave



As we said above, outputs are always referenced via their promoted name.
So if you built the Sellar problem using connections (see :ref:`SellarDerivativesConnected <openmdao.test_suite.components.sellar.py>`),
instead of promoting everything, then you would access the variables like this:


.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_not_promoted_sellar_set_get_outputs
    :layout: interleave


Working with Array Variables
----------------------------

When you have an array variable, for convenience we allow you to set the value with any properly-sized array, list, or tuple.
In other words, the shape of the list has to match the shape of the actual data.


.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_set_get_array
    :layout: interleave

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_set_2d_array
    :layout: interleave


Residuals
---------

If you want to look at the residual values associated with any particular output variable, you will reference them using the same naming conventions the outputs.
Also like outputs, you will be given the residuals in the unscaled dimensional form.

.. embed-code::
    openmdao.core.tests.test_problem.TestProblem.test_feature_residuals
    :layout: interleave


Inputs
------

.. note::
    99.9% of the time, you don't want to work with input variables.
    Instead you probably want to use the associated output variable.
    But if you really, really want to, this is how you do it.

To set or get the and input variable, you reference it by its absolute path name. The full path name is necessary, because
you could have an output (source) variable in units of meters, and then two connected inputs (targets) in units of millimeters and centimeters, respectively.
Hence you need a specific path to reference each of the two different inputs separately to get the value in that input's units.


.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_promoted_sellar_set_get_inputs
    :layout: interleave


Specifying Units
----------------

You can also set an input or request the valuable of any variable in a different unit than the one it is declared in, and OpenMDAO will
peform the conversion for you. This is done with the `Problem` methods `get_val` and `set_val`.

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_with_units
    :layout: interleave

When dealing with arrays, you can set or get specific indices or index ranges by adding the "indices" argument to the calls:

.. embed-code:: openmdao.core.tests.test_problem.TestProblem.test_feature_get_set_array_with_units
    :layout: interleave

An alternate method of specifying the indices is by making use of the :code:`slicer` object. This object serves as a
helper function allowing the user to specify the indices value using the same syntax as you would when
accessing a numpy array.

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
