"""Test the DirectSolver linear solver class."""

from __future__ import division, print_function

import numpy
import unittest
from six import iterkeys

from openmdao.api import Problem, Group, ImplicitComponent, NewtonSolver, DirectSolver
from openmdao.api import GlobalJacobian, COOmatrix, CSRmatrix, DenseMatrix
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.parametric_suite import parametric_suite
from openmdao.devtools.testutil import assert_rel_error


class TestImplComp(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('x', numpy.ones(2))
        self.add_output('y', numpy.ones(2))
        self.mtx = numpy.array([
            [ 3., 4.],
            [ 2., 3.],
        ])
        # Inverse is
        # [ 3.,-4.],
        # [-2., 3.],

        #self.declare_partials('y', 'x', val=-numpy.eye(2))
        #self.declare_partials('y', 'y', val=self.mtx)

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['y'] = self.mtx.dot(outputs['y']) - inputs['x']

    def linearize(self, inputs, outputs, partials):
        partials['y', 'x'] = -numpy.eye(2)
        partials['y', 'y'] = self.mtx


class TestDirectSolver(unittest.TestCase):

    def test_direct_solver_comp(self):
        """
        Test the direct solver on a component.
        """
        for jac in ['dict', 'coo', 'csr', 'dense']:
            prob = Problem(model=TestImplComp())
            prob.model.nl_solver = NewtonSolver()
            prob.model.ln_solver = DirectSolver()

            if jac == 'dict':
                pass
            elif jac == 'coo':
                prob.model.jacobian = GlobalJacobian(matrix_class=COOmatrix)
            elif jac == 'csr':
                prob.model.jacobian = GlobalJacobian(matrix_class=CSRmatrix)
            elif jac == 'dense':
                prob.model.jacobian = GlobalJacobian(matrix_class=DenseMatrix)

            prob.setup(check=False)

            prob.run_model()
            assert_rel_error(self, prob['y'], [-1., 1.])

            prob.model._vectors['residual']['linear'].set_const(2.0)
            prob.model._vectors['output']['linear'].set_const(0.0)
            prob.model._solve_linear(['linear'], 'fwd')
            result = prob.model._vectors['output']['linear'].get_data()
            assert_rel_error(self, result, [-2., 2.])

            prob.model._vectors['output']['linear'].set_const(2.0)
            prob.model._vectors['residual']['linear'].set_const(0.0)
            prob.model._solve_linear(['linear'], 'rev')
            result = prob.model._vectors['residual']['linear'].get_data()
            assert_rel_error(self, result, [2., -2.])

    def test_direct_solver_group(self):
        """
        Test the direct solver on a group.
        """
        prob = Problem(model=TestImplicitGroup(lnSolverClass=DirectSolver))

        prob.setup(check=False)
        prob.model._linearize()

        prob.model._vectors['residual']['linear'].set_const(1.0)
        prob.model._vectors['output']['linear'].set_const(0.0)
        prob.model._solve_linear(['linear'], 'fwd')
        result = prob.model._vectors['output']['linear']._data
        assert_rel_error(self, result[0], prob.model.expected_solution[0], 1e-15)
        assert_rel_error(self, result[1], prob.model.expected_solution[1], 1e-15)

        prob.model._vectors['output']['linear'].set_const(1.0)
        prob.model._vectors['residual']['linear'].set_const(0.0)
        prob.model._solve_linear(['linear'], 'rev')
        result = prob.model._vectors['residual']['linear']._data
        assert_rel_error(self, result[0], prob.model.expected_solution[0], 1e-15)
        assert_rel_error(self, result[1], prob.model.expected_solution[1], 1e-15)

    @parametric_suite(
        vector_class=['default'],
        global_jac=[False, True],
        jacobian_type=['dense'],
        partial_type=['array', 'sparse', 'aij'],
        num_var=[2, 3],
        var_shape=[(2, 3), (2,)],
        component_class=['explicit'],
        connection_type=['implicit', 'explicit'],
        run_by_default=False,
    )
    def test_subset(self, param_instance):
        param_instance.linear_solver_class = DirectSolver

        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        expected_values = model.expected_values
        if expected_values:
            actual = {key: problem[key] for key in iterkeys(expected_values)}
            assert_rel_error(self, actual, expected_values, 1e-8)

        expected_totals = model.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_rel_error(self, totals, expected_totals, 1e-8)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_rel_error(self, totals, expected_totals, 1e-8)


if __name__ == "__main__":
    unittest.main()
