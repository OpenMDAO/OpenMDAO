""" Unit tests for the problem interface."""

import unittest
import warnings
from six import assertRaisesRegex

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp, PETScVector, NonlinearBlockGS
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.paraboloid import Paraboloid

from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesConnected


class TestProblem(unittest.TestCase):

    def test_set_2d_array(self):

        prob = Problem(model=Group())
        model = prob.model
        model.add_subsystem(name='indeps',
                            subsys=IndepVarComp(name='X_c', shape=(3, 1)))
        prob.setup()

        new_val = -5*np.ones((3, 1))
        prob['indeps.X_c'] = new_val
        assert_rel_error(self, prob['indeps.X_c'], new_val, 1e-10)

        new_val = 2.5*np.ones(3)
        prob['indeps.X_c'][:, 0] = new_val
        assert_rel_error(self, prob['indeps.X_c'], new_val.reshape((3,)), 1e-10)
        assert_rel_error(self, prob['indeps.X_c'][:, 0], new_val, 1e-10)

    def test_set_checks_shape(self):

        model = Group()

        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_output('num')
        indep.add_output('arr', shape=(10, 1))

        prob = Problem(model)
        prob.setup()

        msg = "Incompatible shape for '.*': Expected (.*) but got (.*)"

        # check valid scalar value
        new_val = -10.
        prob['indep.num'] = new_val
        assert_rel_error(self, prob['indep.num'], new_val, 1e-10)

        # check bad scalar value
        bad_val = -10*np.ones((10))
        with assertRaisesRegex(self, ValueError, msg):
            prob['indep.num'] = bad_val

        # check assign scalar to array
        arr_val = new_val*np.ones((10, 1))
        prob['indep.arr'] = new_val
        assert_rel_error(self, prob['indep.arr'], arr_val, 1e-10)

        # check valid array value
        new_val = -10*np.ones((10, 1))
        prob['indep.arr'] = new_val
        assert_rel_error(self, prob['indep.arr'], new_val, 1e-10)

        # check bad array value
        bad_val = -10*np.ones((10))
        with assertRaisesRegex(self, ValueError, msg):
            prob['indep.arr'] = bad_val

        # check valid list value
        new_val = new_val.tolist()
        prob['indep.arr'] = new_val
        assert_rel_error(self, prob['indep.arr'], new_val, 1e-10)

        # check bad list value
        bad_val = bad_val.tolist()
        with assertRaisesRegex(self, ValueError, msg):
            prob['indep.arr'] = bad_val

    def test_compute_total_derivs_basic(self):
        # Basic test for the method using default solvers on simple model.

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup(check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], 8.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt)

        assert_rel_error(self, derivs['f_xy', 'x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy', 'y'], 8.0, 1e-6)

    def test_compute_total_derivs_basic_return_dict(self):
        # Make sure 'dict' return_format works.

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup(check=False, mode='fwd')
        prob.model.suppress_solver_output = True
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, derivs['f_xy']['x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy']['y'], 8.0, 1e-6)

        prob.setup(check=False, mode='rev')
        prob.run_model()

        of = ['f_xy']
        wrt = ['x', 'y']
        derivs = prob.compute_total_derivs(of=of, wrt=wrt, return_format='dict')

        assert_rel_error(self, derivs['f_xy']['x'], -6.0, 1e-6)
        assert_rel_error(self, derivs['f_xy']['y'], 8.0, 1e-6)

    def test_feature_set_indeps(self):
        prob = Problem()

        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

        prob.setup()

        prob['x'] = 2.
        prob['y'] = 10.
        prob.run_model()
        assert_rel_error(self, prob['f_xy'], 214.0, 1e-6)

    def test_feature_numpyvec_setup(self):

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

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
        model = prob.model = Group()
        model.add_subsystem('p1', IndepVarComp('x', 0.0), promotes=['x'])
        model.add_subsystem('p2', IndepVarComp('y', 0.0), promotes=['y'])
        model.add_subsystem('comp', Paraboloid(), promotes=['x', 'y', 'f_xy'])

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

        prob.model.options['method'] = 'slsqp'
        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_design_var('con1')
        prob.model.add_design_var('con2')
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

        prob.driver = ScipyOpt()
        prob.driver.options['method'] = 'slsqp'

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_design_var('con1')
        prob.model.add_design_var('con2')

        prob.setup()
        prob.run_driver()

        assert_rel_error(self, prob['x'], 0.0, 1e-6)
        assert_rel_error(self, prob['y'], [3.160000, 3.755278], 1e-6)
        assert_rel_error(self, prob['z'], [1.977639, 0.000000], 1e-6)
        assert_rel_error(self, prob['obj'], 3.18339, 1e-6)

    def test_feature_promoted_sellar_set_get_outputs(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob['x'] = 2.75
        assert_rel_error(self, prob['x'], 2.75, 1e-6)

        prob.run_model()

        assert_rel_error(self, prob['y1'], 27.3049178437, 1e-6)

    def test_feature_not_promoted_sellar_set_get_outputs(self):

        prob = Problem()
        prob.model = SellarDerivativesConnected()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        prob['px.x'] = 2.75
        assert_rel_error(self, prob['px.x'], 2.75, 1e-6)

        prob.run_model()

        assert_rel_error(self, prob['d1.y1'], 27.3049178437, 1e-6)

    def test_feature_promoted_sellar_set_get_inputs(self):

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

    def test_feature_set_get_array(self):
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
        prob['z'] = [1.5, 1.5]  # for convenience we convert the list to an array.
        assert_rel_error(self, prob['z'], (1.5, 1.5), 1e-6)

        prob.run_model()
        assert_rel_error(self, prob['y1'], 5.43379016853, 1e-6)
        assert_rel_error(self, prob['y2'], 5.33104915618, 1e-6)

        prob['z'] = np.array([2.5, 2.5])
        assert_rel_error(self, prob['z'], [2.5, 2.5], 1e-6)

        prob.run_model()
        assert_rel_error(self, prob['y1'], 9.87161739688, 1e-6)
        assert_rel_error(self, prob['y2'], 8.14191301549, 1e-6)

    @unittest.skip('access via promoted names is not working yet')
    def test_feature_residuals(self):

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.setup()

        # default value from the class definition

        prob['z'] = [1.5, 1.5]  # for convenience we convert the list to an array.
        prob.run_model()

        inputs, outputs, residuals = prob.model.get_nonlinear_vectors()

        self.assertLess(residuals['y1'], 1e-6)
        self.assertLess(residuals['y2'], 1e-6)

    def test_setup_bad_mode(self):
        # Test error message when passing bad mode to setup.

        prob = Problem(Group())

        try:
            prob.setup(mode='junk')
        except ValueError as err:
            msg = "Unsupported mode: 'junk'"
            self.assertEqual(str(err), msg)
        else:
            self.fail('Expecting ValueError')

    def test_root_deprecated(self):
        # testing the root property
        msg = "The 'root' property provides backwards compatibility " \
            + "with OpenMDAO <= 1.x ; use 'model' instead."

        prob = Problem()

        # check deprecation on setter
        with warnings.catch_warnings(record=True) as w:
            prob.root = Group()

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

        # check deprecation on getter
        with warnings.catch_warnings(record=True) as w:
            prob.root

        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[0].category, DeprecationWarning))
        self.assertEqual(str(w[0].message), msg)

        # testing the root kwarg
        with self.assertRaises(ValueError) as cm:
            prob = Problem(root=Group(), model=Group())
        err = cm.exception
        self.assertEqual(str(err), "cannot specify both `root` and `model`. `root` has been "
                         "deprecated, please use model")

        with warnings.catch_warnings(record=True) as w:
            prob = Problem(root=Group)

        self.assertEqual(str(w[0].message), "The 'root' argument provides backwards "
                         "compatibility with OpenMDAO <= 1.x ; use 'model' instead.")


if __name__ == "__main__":
    unittest.main()
