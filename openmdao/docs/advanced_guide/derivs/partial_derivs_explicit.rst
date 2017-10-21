.. _advanced_guide_partial_derivs_explicit:

OpenMDAO considers component derivatives to be **partial derivatives**.
The framework uses these partial derivatives in order to compute the **total derivatives** across your whole model.
This tutorial is focused on how to define the partial derivatives for components that inherit from :ref:`ExplicitComponent <comp-type-2-explicitcomp>`.



****************************************************
Defining partial derivatives on Explicit Components
****************************************************

For any :ref:`ExplicitComponent <comp-type-2-explicitcomp>` you are going to provide derivatives of the **outputs with respect to the inputs**.
Whenever you are going to define derivatives, there are two things you're required to do:

    #. declare the partial derivatives via :code:`declare_partials`
    #. specify their values via :code:`compute_partials`

Here is an example, based on the :ref:`Betz Limit Example <betz_limit_tutorial>`:

.. embed-code::
    openmdao.test_suite.test_examples.test_betz_limit.ActuatorDisc


The calls to :code:`declare_partials` tell OpenMDAO which partial derivatives to expect.
This is always done inside the :code:`setup` method.
In this example not all the outputs depend on all the inputs, and you see that if you look at the derivative declarations.
Any partial that is not declared is assumed to be 0.
You could have declared all the partials in just one line as follows (see this :ref:`feature doc <feature_specify_partials>` for more details):

.. code::

    self.declare_partials('*', '*')

However, declaring the partials all in one line as shown above indicates to OpenMDAO that all the partials are non-zero.
While you saved yourself a few lines of code there, it would come at the potential expense of performance.
So generally it is better to be more specific and declare only the non-zero partials.

.. note::
    There are a few more options to :code:`declare_partials` that are worth taking a look at.
    There is support for when your derivatives are constant and if you want to specify derivatives in a sparse AIJ format.
    The full details are :ref:`here <feature_specify_partials>`.

After you declare the non-zero partial derivatives, you need to implement the :code:`compute_partials` method to perform the actual
derivative computations.
OpenMDAO will call this method whenever it needs to work with the partial derivatives.
The values are stored in the Jacobian object, :code:`J`, and get used in the linear solutions that are necessary to compute model-level total derivatives.
This API results in the assembly of a Jacobian matrix in memory.
The :code:`compute_partials` API is the most appropriate way to declare derivatives in the vast majority of circumstances,
and you should use it unless you have a good reason not to.

Providing derivatives using the matrix-free API
************************************************

Sometimes you don't want to assemble the full partial-derivative Jacobian of your component in memory.
We're not going to discuss why this might be the case in this tutorial.
For now, let's just say that if matrix-assembly won't work for your application you are likely already well aware of this issue.
If you can't imagine why you would want to use a matrix-free API, then don't worry about this.
If you do need to work matrix-free, there is a :code:`compute_jacvec_product` API that you can use.
See the :ref:`ExplicitComponent <comp-type-2-explicitcomp>` reference for an example of how to use it.


How Do I Know if My Derivatives Are Correct?
**************************************************

It is really important, if you are going to provide analytic derivatives, that you make sure they are correct.
Nothing else will screw up the convergence of your analysis or optimization more quickly than bad derivatives.
OpenMDAO provides a way to check your partial derivatives so you can make sure they are right.
Any time you implement analytic derivatives, or change the nonlinear equations of your analysis, you should check your partial derivatives this way.

.. embed-test::
    openmdao.test_suite.test_examples.test_betz_limit.TestBetzLimit.test_betz_derivatives

.. note::

    :code:`check_partials` is **really** important when you're coding derivatives.
    It has some options to give you more detailed outputs for debugging and to let you limit which components get tested.
    You should look over the complete :ref:`check_partials doc <feature_check_partials>` before you start doing heavy development with derivatives.

There is a lot of information there, including checks for both forward and reverse derivatives.
If you've taken our advice and stuck with the :code:`compute_partials` method, then you can ignore all the reverse stuff.
For now, just take a look at the third-to-last column, which shows the norm of the difference between the analytic derivatives Jacobian and one that was approximated using finite-difference.
Here, all the numbers are really small, and that's what you want to see.
Its rare, except for linear functions, that the finite-difference and analytic derivatives will match exactly, but they should be pretty close.

