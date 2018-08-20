:orphan:

.. _lnuserdefined:

LinearUserDefined
=================

LinearUserDefined is a solver that lets you define a custom method for performing a linear solve on a component. The default
method is named "solve_linear", but you can give it any name by passing in the function or method handle to
the "solve_function" attribute.

The function needs to have the following signature:

.. code-block:: python

    def my_solve_function(d_outputs, d_residuals, mode):
        r"""
        Apply inverse jac product. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        d_outputs : Vector
            unscaled, dimensional quantities read via d_outputs[key]
        d_residuals : Vector
            unscaled, dimensional quantities read via d_residuals[key]
        mode : str
            either 'fwd' or 'rev'

        Returns
        -------
        None or bool or (bool, float, float)
            The bool is the failure flag; and the two floats are absolute and relative error.
        """


Here is a rather contrived example where an identity preconditioner is used by giving the component's "mysolve"
method to a LinearUserDefined solver.

.. embed-code::
    openmdao.solvers.linear.tests.test_user_defined.TestUserDefinedSolver.test_feature
    :layout: interleave

LinearUserDefined Options
-------------------------

.. embed-options::
    openmdao.solvers.linear.user_defined
    LinearUserDefined
    options

.. tags:: Solver, LinearSolver
