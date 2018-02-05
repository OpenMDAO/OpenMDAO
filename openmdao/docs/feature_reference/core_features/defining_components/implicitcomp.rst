.. _comp-type-3-implicitcomp:

*****************
ImplicitComponent
*****************

Implicit variables are those that are computed as an implicit function of other variables.
For instance, :math:`y` would be an implicit variable, given that it is computed by solving :math:`\cos(x \cdot y) - z \cdot y = 0`.
In OpenMDAO, implicit variables are defined as the outputs of components that inherit from :ref:`ImplicitComponent <openmdao.core.implicitcomponent.py>`.

In the above implict expression, :math:`y` is the implicit variable while :math:`x` and :math:`z` would be considered inputs.

ImplicitComponent Methods
-------------------------

The implementation of each method will be illustrated using a simple implicit component that computes the output :math:`x` implicitly via a quadratic equation, :math:`ax^2 + bx + c =0`, where :math:`a`, :math:`b`, and :math:`c` are inputs to the component.

::

    class QuadraticComp(ImplicitComponent):
        """
        A Simple Implicit Component representing a Quadratic Equation.

        R(a, b, c, x) = ax^2 + bx + c

        Solution via Quadratic Formula:
        x = (-b + sqrt(b^2 - 4ac)) / 2a
        """

- :code:`setup()` :

  Declare input and output variables via :code:`add_input` and :code:`add_output`.
  Information like variable names, sizes, units, and bounds are declared. Also, declare partial derivatives that this component provides. Here we use the wild card to say that
  this component provides derivatives of all implicit residuals with respect to all inputs and outputs.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.QuadraticComp.setup

- :code:`apply_nonlinear(inputs, outputs, residuals)` :

  Compute the :code:`residuals`, given the :code:`inputs` and :code:`outputs`.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.QuadraticComp.apply_nonlinear

- :code:`solve_nonlinear(inputs, outputs)` :

  [Optional] Compute the :code:`outputs`, given the :code:`inputs`.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.QuadraticComp.solve_nonlinear

- :code:`linearize(inputs, outputs, partials)` :

  [Optional] The component's partial derivatives of interest are those of the :code:`residuals` with respect to the :code:`inputs` and the :code:`outputs`.
  If the user computes the partial derivatives explicitly, they are provided here.
  If the user wants to implement partial derivatives in a matrix-free way, this method provides a place to perform any necessary assembly or pre-processing for the matrix-vector products.
  Regardless of how the partial derivatives are computed, this method provides a place to perform any relevant factorizations for directly solving or preconditioning the linear system.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.QuadraticLinearize.linearize

- :code:`apply_linear(inputs, outputs, d_inputs, d_outputs, d_residuals, mode)` :

  [Optional] If the user wants to implement partial derivatives in a matrix-free way, this method performs the matrix-vector product. If mode is 'fwd', this method computes :math:`d\_{residuals} = J \cdot [ d\_{inputs} \quad d\_{outputs} ]^T`. If mode is 'rev', this method computes :math:`[ d\_{inputs} \quad d\_{outputs} ]^T = J^T \cdot d\_{residuals}`.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.QuadraticJacVec.apply_linear

- :code:`solve_linear(d_outputs, d_residuals, mode)` :

  [Optional] Solves a linear system where the matrix is :math:`d\_{residuals} / d\_{outputs}` or its transpose. If mode is 'fwd', the right-hand side vector is :math:`d\_{residuals}` and the solution vector is :math:`d\_{outputs}`. If mode is 'rev', the right-hand side vector is :math:`d\_{outputs}` and the solution vector is :math:`d\_{residuals}`.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.QuadraticJacVec.solve_linear

- :code:`guess_nonlinear(self, inputs, outputs, residuals)` :

  [Optional] This method allows the user to calculate and specify an initial guess for implicit states. It is called prior to every call to
  `solve_nonlinear`, so it is useful for when you would like to "reset" the initial conditions on an inner-nested solve whenever an outer
  loop solver or driver changes other values. Since it is a hook for custom code, you could also use it to monitor variables in the input,
  output, or residual vectors and change the initial guess when some condition is met. Here is a simple example where we use NewtonSolver to
  find one of the roots of a second-order quadratic equation. Which root you get depends on the initial guess.

  .. embed-code::
      openmdao.core.tests.test_impl_comp.ImplicitCompTestCase.test_guess_nonlinear_feature

.. tags:: Component, ImplicitComponent
