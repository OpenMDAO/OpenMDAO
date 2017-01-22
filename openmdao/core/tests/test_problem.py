""" Unit tests for the problem interface."""
from __future__ import print_function
import unittest

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid


class TestProblem(unittest.TestCase):

    def test_compute_total_derivs_basic(self):
        # Basic test for the method using default solvers on simple model.

        top = Problem()
        root = top.root = Group()
        root.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        root.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        root.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        top.setup(check=False, mode='fwd')
        top.root.suppress_solver_output = True
        top.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = top.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], 8.0, 1e-6)

        top.setup(check=False, mode='rev')
        top.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = top.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], 8.0, 1e-6)

    def test_feature_numpyvec_setup(self):

        from openmdao.api import Problem, Group, IndepVarComp
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        root = prob.root = Group()
        root.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        root.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        root.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob['x'] = 2.
        prob['y'] = 10.
        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 214.0, 1e-6)

        prob['x'] = 0.
        prob['y'] = 0.
        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 22.0, 1e-6)

        # skip the setup error checking
        prob.setup(check=False)
        prob['x'] = 4
        prob['y'] = 8.

        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 174.0, 1e-6)

    def test_feature_petsc_setup(self):

        from openmdao.api import Problem, Group, IndepVarComp, PETScVector
        from openmdao.test_suite.components.paraboloid import Paraboloid

        prob = Problem()
        root = prob.root = Group()
        root.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        root.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        root.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        # use PETScVector when using any PETSc linear solvers or running under MPI
        prob.setup(vector_class=PETScVector)
        prob['x'] = 2.
        prob['y'] = 10.

        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 214.0, 1e-6)

    def test_feature_check_total_derivatives_manual(self):

        raise unittest.SkipTest("check_total_derivatives not implemented yet")

        from openmdao.api import Problem, NonlinearBlockGS
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.root = SellarDerivatives()
        prob.root.nl_solver = NonlinearBlockGS()

        prob.setup()
        prob.run_model()

        # manually specify which derivatives to check
        prob.check_total_derivatives(of=['obj', 'con1'], wrt=['x', 'z'])
        # TODO: Need to devlop the group FD/CS api, so user can control how this
        #       happens by chaninging settings on the root node

    def test_feature_check_total_derivatives_from_driver(self):

        raise unittest.SkipTest("check_total_derivatives not implemented yet")

        from openmdao.api import Problem, NonlinearBlockGS, ScipyOpt
        from openmdao.test_suite.components.sellar import SellarDerivatives

        prob = Problem()
        prob.root = SellarDerivatives()
        prob.root.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob.driver = ScipyOpt()
        prob.driver.options['algorithm'] = 'slsqp'
        prob.root.add_design_var('x')
        prob.root.add_design_var('z')
        prob.root.add_objective('obj')
        prob.root.add_design_var('con1')
        prob.root.add_design_var('con2')
        # re-do setup since we changed the driver and problem inputs/outputs
        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives of all obj+constraints w.r.t all design variables
        prob.check_total_derivatives()

    def test_setup_bad_mode(self):
        # Test error message when passing bad mode to setup.

        top = Problem()
        root = top.root = Group()

        try:
            top.setup(mode='junk')
        except ValueError as err:
            msg = "Unsupported mode: 'junk'"
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting ValueError')


if __name__ == "__main__":
    unittest.main()
