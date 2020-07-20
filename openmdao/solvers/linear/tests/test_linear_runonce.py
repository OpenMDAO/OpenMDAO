"""Test the LinearRunOnce linear solver class."""

import unittest

import openmdao.api as om
from openmdao.test_suite.components.paraboloid import Paraboloid
from openmdao.test_suite.groups.parallel_groups import ConvergeDivergeGroups
from openmdao.utils.assert_utils import assert_near_equal


class TestLinearRunOnceSolver(unittest.TestCase):

    def test_converge_diverge_groups(self):
        # Test derivatives for converge-diverge-groups topology.
        prob = om.Problem()
        model = prob.model = ConvergeDivergeGroups()

        model.linear_solver = om.LinearRunOnce()
        model.g1.linear_solver = om.LinearRunOnce()
        model.g1.g2.linear_solver = om.LinearRunOnce()
        model.g3.linear_solver = om.LinearRunOnce()

        prob.set_solver_print(level=0)
        prob.setup(check=False, mode='fwd')
        prob.run_model()

        wrt = ['iv.x']
        of = ['c7.y1']

        # Make sure value is fine.
        assert_near_equal(prob['c7.y1'], -102.7, 1e-6)

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        J = prob.compute_totals(of=of, wrt=wrt, return_format='flat_dict')
        assert_near_equal(J['c7.y1', 'iv.x'][0][0], -40.75, 1e-6)

    def test_undeclared_options(self):
        # Test that using options that should not exist in class cause an error
        solver = om.LinearRunOnce()

        msg = "\"LinearRunOnce: Option '%s' cannot be set because it has not been declared.\""

        for option in ['atol', 'rtol', 'maxiter', 'err_on_non_converge']:
            with self.assertRaises(KeyError) as context:
                solver.options[option] = 1

            self.assertEqual(str(context.exception), msg % option)

    def test_feature_solver(self):
        import openmdao.api as om
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        model.linear_solver = om.LinearRunOnce()

        prob.setup(check=False, mode='fwd')

        prob.set_val('x', 0.0)
        prob.set_val('y', 0.0)

        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_totals(of=of, wrt=wrt, return_format='dict')

        assert_near_equal(derivs['f_xy']['x'], [[-6.0]], 1e-6)
        assert_near_equal(derivs['f_xy']['y'], [[8.0]], 1e-6)


if __name__ == "__main__":
    unittest.main()
