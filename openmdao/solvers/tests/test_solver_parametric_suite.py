"""Runs a parametric test over several of the linear solvers."""

import numpy as np
import unittest

from openmdao.core.group import Group
from openmdao.core.problem import Problem
from openmdao.core.implicitcomponent import ImplicitComponent
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.test_suite.groups.implicit_group import TestImplicitGroup
from openmdao.test_suite.parametric_suite import parametric_suite


class ImplComp4Test(ImplicitComponent):

    def setup(self):
        self.add_input('x', np.ones(2))
        self.add_output('y', np.ones(2))
        self.mtx = np.array([
            [3., 4.],
            [2., 3.],
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
        for jac in [None, 'csc', 'dense']:
            prob = Problem(model=ImplComp4Test())
            prob.model.nonlinear_solver = NewtonSolver(solve_subsystems=False)
            if jac in ('csc', 'dense'):
                prob.model.options['assembled_jac_type'] = jac
            prob.model.linear_solver = DirectSolver(assemble_jac=jac in ('csc','dense'))
            prob.set_solver_print(level=0)

            prob.setup()

            prob.run_model()
            assert_near_equal(prob['y'], [-1., 1.])

            d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

            d_residuals.set_const(2.0)
            d_outputs.set_const(0.0)
            prob.model.run_solve_linear(['linear'], 'fwd')
            result = d_outputs._data
            assert_near_equal(result, [-2., 2.])

            d_outputs.set_const(2.0)
            d_residuals.set_const(0.0)
            prob.model.run_solve_linear(['linear'], 'rev')
            result = d_residuals._data
            assert_near_equal(result, [2., -2.])

    def test_direct_solver_group(self):
        """
        Test the direct solver on a group.
        """
        prob = Problem(model=TestImplicitGroup(lnSolverClass=DirectSolver))

        prob.setup()

        # Set this to False because we have matrix-free component(s).
        prob.model.linear_solver.options['assemble_jac'] = False

        # Conclude setup but don't run model.
        prob.final_setup()

        prob.model.run_linearize()

        d_inputs, d_outputs, d_residuals = prob.model.get_linear_vectors()

        d_residuals.set_const(1.0)
        d_outputs.set_const(0.0)
        prob.model.run_solve_linear(['linear'], 'fwd')
        result = d_outputs._data
        assert_near_equal(result, prob.model.expected_solution, 1e-15)

        d_outputs.set_const(1.0)
        d_residuals.set_const(0.0)
        prob.model.run_solve_linear(['linear'], 'rev')
        result = d_residuals._data
        assert_near_equal(result, prob.model.expected_solution, 1e-15)

    @parametric_suite(
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
        param_instance.linear_solver_options = {}  # defaults not valid for DirectSolver

        param_instance.setup()
        problem = param_instance.problem
        model = problem.model

        expected_values = model.expected_values
        if expected_values:
            actual = {key: problem[key] for key in expected_values}
            assert_near_equal(actual, expected_values, 1e-8)

        expected_totals = model.expected_totals
        if expected_totals:
            # Forward Derivatives Check
            totals = param_instance.compute_totals('fwd')
            assert_near_equal(totals, expected_totals, 1e-8)

            # Reverse Derivatives Check
            totals = param_instance.compute_totals('rev')
            assert_near_equal(totals, expected_totals, 1e-8)

if __name__ == "__main__":
    unittest.main()
