""" Unit tests for the problem interface."""
from __future__ import print_function
import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, PETScVector, NonlinearBlockGS
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesConnected


class TestProblem(unittest.TestCase):

    def test_compute_total_derivs_basic(self):
        # Basic test for the method using default solvers on simple model.

        top = Problem()
        root = top.model = Group()
        root.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        root.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        root.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        top.setup(check=False, mode='fwd')
        top.model.suppress_solver_output = True
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

        prob = Problem()
        root = prob.model = Group()
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

    @unittest.skipUnless(PETScVector, "PETSc is required.")
    def test_feature_petsc_setup(self):

        prob = Problem()
        root = prob.model = Group()
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

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()
        prob.run_model()

        # manually specify which derivatives to check
        # TODO: need a decorator to capture this output and put it into the doc,
        #       or maybe just a new kind of assert?
        prob.check_total_derivatives(of=['obj', 'con1'], wrt=['x', 'z'])
        # TODO: Need to devlop the group FD/CS api, so user can control how this
        #       happens by chaninging settings on the root node

    def test_feature_check_total_derivatives_from_driver(self):

        raise unittest.SkipTest("check_total_derivatives not implemented yet")

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob.driver = ScipyOpt()
        prob.driver.options['method'] = 'slsqp'
        prob.driver.add_design_var('x')
        prob.driver.add_design_var('z')
        prob.driver.add_objective('obj')
        prob.driver.add_design_var('con1')
        prob.driver.add_design_var('con2')
        # re-do setup since we changed the driver and problem inputs/outputs
        prob.setup()

        # We don't call run_driver() here because we don't
        # actually want the optimizer to run
        prob.run_model()

        # check derivatives of all obj+constraints w.r.t all design variables
        prob.check_total_derivatives()
        # TODO: need a decorator to capture this output and put it into the doc,
        #       or maybe just a new kind of assert?

    def test_feature_run_driver(self):
        raise unittest.SkipTest("drivers not implemented yet")

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        # TODO: this api is not final, just a placeholder for now
        prob.driver = ScipyOpt()
        prob.driver.options['method'] = 'slsqp'
        # note: this might differ from clippy api, but is consistent with arg name in scipy.
        prob.model.add_design_var('x')
        prob.model.add_design_var('z')
        prob.model.add_objective('obj')
        prob.model.add_design_var('con1')
        prob.model.add_design_var('con2')

        prob.setup()
        prob.run_driver()

        self.assert_rel_error(self, prob['x'], 0.0, 1e-6)
        self.assert_rel_error(self, prob['y'], [3.160000, 3.755278], 1e-6)
        self.assert_rel_error(self, prob['z'], [1.977639, 0.000000], 1e-6)
        self.assert_rel_error(self, prob['obj'], 3.18339, 1e-6)

    def test_feature_simple_promoted_sellar_set_get_outputs(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob['x'] = 2.75
        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        prob.run_model()

        assert_rel_error(self, prob['y1'], 27.3049178437, 1e-6)

    def test_feature_simple_not_promoted_sellar_set_get_outputs(self):

        prob = Problem()
        prob.model = SellarDerivativesConnected()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob['px.x'] = 2.75
        assert_rel_error(self, prob['px.x'], 2.75, 1e-6)

        prob.run_model()

        assert_rel_error(self, prob['d1.y1'], 27.3049178437, 1e-6)

    def test_feature_simple_promoted_sellar_set_get_inputs(self):
        raise unittest.SkipTest("set/get inputs via full path name not supported yet")

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob['x'] = 2.75
        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        prob.run_model()

        # the output variable, referenced by the promoted name
        assert_rel_error(self, prob['y1'], 27.3049178437, 1e-6)
        # the connected input variable, referenced by the absolute path
        assert_rel_error(self, prob['d2.y1'], 27.3049178437, 1e-6)

    def test_feature_set_get(self):
        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        # default value from the class definition
        assert_rel_error(self, prob['x'], 1.0, 1e-6)
        prob['x'] = 2.75
        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        assert_rel_error(self, prob['z'], [5.0, 2.0], 1e-6)
        prob['z'] = [1.5, 1.5]  # for convenience we convert the list to an array.
        assert_rel_error(self, prob['z'], [1.5, 1.5], 1e-6)

        prob.run_model()
        assert_rel_error(self, prob['y1'], 5.43379016853, 1e-6)
        assert_rel_error(self, prob['y2'], 5.33104915618, 1e-6)

        prob['z'] = np.array([2.5, 2.5])
        assert_rel_error(self, prob['z'], [2.5, 2.5], 1e-6)

        prob.run_model()
        assert_rel_error(self, prob['y1'], 9.87161739688, 1e-6)
        assert_rel_error(self, prob['y2'], 8.14191301549, 1e-6)

    def test_setup_bad_mode(self):
        # Test error message when passing bad mode to setup.

        top = Problem(Group())

        try:
            top.setup(mode='junk')
        except ValueError as err:
            msg = "Unsupported mode: 'junk'"
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting ValueError')


if __name__ == "__main__":
    unittest.main()
