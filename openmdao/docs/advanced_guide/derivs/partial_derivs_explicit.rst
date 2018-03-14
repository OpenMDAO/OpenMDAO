.. _advanced_guide_partial_derivs_explicit:

OpenMDAO considers component derivatives to be **partial derivatives**.
The framework uses these partial derivatives in order to compute the **total derivatives** across your whole model.
This tutorial is focused on how to define the partial derivatives for components that inherit from :ref:`ExplicitComponent <comp-type-2-explicitcomp>`.


***************************************************
Defining Partial Derivatives on Explicit Components
***************************************************

For any :ref:`ExplicitComponent <comp-type-2-explicitcomp>` you are going to provide derivatives of the **outputs with respect to the inputs**.
Whenever you are going to define derivatives, there are two things you're required to do:

    #. Declare the partial derivatives via :code:`declare_partials`.
    #. Specify their values via :code:`compute_partials`.

Here is an example, based on the :ref:`Betz Limit Example <betz_limit_tutorial>`:

.. embed-code::
    openmdao.test_suite.test_examples.test_betz_limit.ActuatorDisc


The calls to :code:`declare_partials` tell OpenMDAO which partial derivatives to expect.
This is always done inside the :code:`setup` method.
In this example, not all the outputs depend on all the inputs, and you'll see that if you look at the derivative declarations.
Any partial that is not declared is assumed to be zero.
You may declare all the partials in just one line as follows (see the :ref:`feature doc on specifying partials <feature_specify_partials>` for more details):

.. code::

    self.declare_partials('*', '*')

Declaring the partials in this fashion, however, indicates to OpenMDAO that all the partials are nonzero.
While you may save yourself a few lines of code using this method, the line savings could come at the expense of performance.
Generally, it is better to be more specific, and declare only the nonzero partials.

.. note::
    There are a few more options to :code:`declare_partials` that are worth taking a look at.
    There is support for when your derivatives are constant, and there is support for specifying derivatives in a sparse AIJ format.
    The full details can be found in the :ref:`feature doc on specifying partials <feature_specify_partials>`.

After you declare the nonzero partial derivatives, you need to implement the :code:`compute_partials` method to perform the actual
derivative computations.
OpenMDAO will call this method whenever it needs to work with the partial derivatives.
The values are stored in the Jacobian object, :code:`J`, and get used in the linear solutions that are necessary to compute model-level total derivatives.
This API results in the assembly of a Jacobian matrix in memory.
The :code:`compute_partials` API is the most appropriate way to declare derivatives in the vast majority of circumstances,
and you should use it unless you have a good reason not to.

Providing Derivatives Using the Matrix-Free API
***********************************************

Sometimes you don't want to assemble the full partial-derivative Jacobian of your component in memory.
The reasons why you might not want this are beyond the scope of this tutorial.
For now, let's assume that if matrix assembly won't work for your application, that you are likely already well aware of this issue.
So if you can't imagine why you would want to use a matrix-free API, you may disregard the following link.
If you do need to work matrix-free, there is a :code:`compute_jacvec_product` API, examples of which can be found
in the feature document for :ref:`ExplicitComponent <comp-type-2-explicitcomp>`.


How Do I Know If My Derivatives Are Correct?
********************************************

It is really important, if you are going to provide analytic derivatives, that you make sure they are correct.
It is hard to overstate the importance of accurate derivatives in the convergence of analysis and optimization problems.
OpenMDAO provides a helper function to make it easier to verify your partial derivatives.
Any time you implement analytic derivatives, or change the nonlinear equations of your analysis, you should check your partial derivatives this way.

.. embed-code::
    openmdao.test_suite.test_examples.test_betz_limit.TestBetzLimit.test_betz_derivatives
    :layout: interleave

.. note::

    :code:`check_partials` is **really** important when you're coding derivatives.
    It has some options to give you more detailed outputs for debugging and to let you limit which components get tested.
    You should look over the complete :ref:`check_partials doc <feature_check_partials>` before you start doing heavy development with derivatives.

There is a lot of information there, but for now, just take a look at the *r(fwd-chk)* column, which shows the norm of the relative difference
between the analytic derivatives Jacobian and one that was approximated using finite differences.
Here, all the numbers are really small, and that's what you want to see.
It's rare, except for linear functions, that the finite difference and analytic derivatives will match exactly, but they should be pretty close.
