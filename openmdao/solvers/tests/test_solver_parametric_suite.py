"""Runs a parametric test over several of the linear solvers."""

from __future__ import division, print_function

import numpy as np
import unittest
from six import iterkeys

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.jacobians.assembled_jacobian import DenseJacobian, COOJacobian, \
                                                  CSRJacobian, CSCJacobian
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.parametric_suite import parametric_suite


class ImplComp4Test(ImplicitComponent):

    def setup(self):
        self.add_input('x', np.ones(2))
        self.add_output('y', np.ones(2))
        self.mtx = np.array([
            [ 3., 4.],
            [ 2., 3.],
        ])
        # Inverse is
        # [ 3.,-4.],
        # [-2., 3.],

        #self.declare_partials('y', 'x', val=-np.eye(2))
        #self.declare_partials('y', 'y', val=self.mtx)
        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['y'] = self.mtx.dot(outputs['y']) - inputs['x']

    def linearize(self, inputs, outputs, partials):
        partials['y', 'x'] = -np.eye(2)
        partials['y', 'y'] = self.mtx


class TestLinearSolverParametricSuite(unittest.TestCase):

    def test_direct_solver_comp(self):
        """
        Test the direct solver on a component.
        """
        for jac in ['dict', 'coo', 'csr', 'csc', 'dense']:
            prob = Problem(model=ImplComp4Test())
            prob.model.nonlinear_solver = NewtonSolver()
            prob.model.linear_solver = DirectSolver()
            prob.set_solver_print(level=0)

            if jac == 'dict':
                pass
            elif jac == 'csr':
                prob.model.jacobian = CSRJacobian()
            elif jac == 'csc':
                prob.model.jacobian = CSCJacobian()
            elif jac == 'coo':
                prob.model.jacobian = COOJacobian()
            elif jac == 'dense':
                prob.model.jacobian = DenseJacobian()

            prob.setup(check=False)

            if jac == 'coo':
                with self.assertRaises(Exception) as context:
                    prob.run_model()
                self.assertEqual(str(context.exception),
                                 "Direct solver is not compatible with matrix type COOMatrix in system ''.")
                continue

            prob.run_model()
            assert_rel_error(self, prob['y'], [-1., 1.])

            d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

            d_residuals.set_const(2.0)
            d_outputs.set_const(0.0)
            prob.model.run_solve_linear(['linear'], 'fwd')
            result = d_outputs.get_data()
            assert_rel_error(self, result, [-2., 2.])

            d_outputs.set_const(2.0)
            d_residuals.set_const(0.0)
            prob.model.run_solve_linear(['linear'], 'rev')
            result = d_residuals.get_data()
            assert_rel_error(self, result, [2., -2.])

    def test_direct_solver_group(self):
        """
        Test the direct solver on a group.
        """
        prob = Problem(model=TestImplicitGroup(lnSolverClass=DirectSolver))

        prob.setup(check=False)

        # Conclude setup but don't run model.
        prob.final_setup()

        prob.model.run_linearize()

        d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        prob.model.run_solve_linear(['linear'], 'fwd')
        result = d_outputs._data
        assert_rel_error(self, result[1], prob.model.expected_solution[0], 1e-15)
        assert_rel_error(self, result[5], prob.model.expected_solution[1], 1e-15)

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        prob.model.run_solve_linear(['linear'], 'rev')
        result = d_residuals._data
        assert_rel_error(self, result[1], prob.model.expected_solution[0], 1e-15)
        assert_rel_error(self, result[5], prob.model.expected_solution[1], 1e-15)

    @parametric_suite(
        vector_class=['default'],
        assembled_jac=[False, True],
        jacobian_type=['dense'],
        partial_type=['array', 'sparse', 'aij'],
        num_var=[2, 3],
        var_shape=[(2, 3), (2,)],
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
