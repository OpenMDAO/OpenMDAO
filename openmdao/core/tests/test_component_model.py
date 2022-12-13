""" Unit tests for using a Component as a Model."""

# This module can be removed when support for the "Component as a model" deprecation is removed

import unittest

import numpy as np
import openmdao.api as om

from openmdao.utils.om_warnings import OMDeprecationWarning
from openmdao.utils.assert_utils import assert_near_equal, assert_warning, assert_no_warning
from openmdao.utils.units import convert_units

model_type_msg = "The model for this Problem is of type '{}'. " \
                 "A future release will require that the model " \
                 "be a Group or a sub-class of Group."


class SellarOneComp(om.ImplicitComponent):

    def initialize(self):
        self.options.declare('solve_y1', types=bool, default=True)
        self.options.declare('solve_y2', types=bool, default=True)

    def setup(self):

        # Global Design Variable
        self.add_input('z', val=np.array([-1., -1.]))

        # Local Design Variable
        self.add_input('x', val=2.)

        self.add_output('y1', val=1.0)
        self.add_output('y2', val=1.0)

        self.add_output('R_y1')
        self.add_output('R_y2')

        if self.options['solve_y1']:
            self.declare_partials('y1', ['x', 'z', 'y1', 'y2'])
        else:
            self.declare_partials('y1', 'y1')

        if self.options['solve_y2']:
            self.declare_partials('y2', ['z', 'y1', 'y2'])
        else:
            self.declare_partials('y2', 'y2')

        self.declare_partials('R_y1', ['R_y1', 'x', 'z', 'y1', 'y2'])
        self.declare_partials('R_y2', ['R_y2','z', 'y1', 'y2'])

    def apply_nonlinear(self, inputs, outputs, residuals):

        z0 = inputs['z'][0]
        z1 = inputs['z'][1]
        x = inputs['x']
        y1 = outputs['y1']
        y2 = outputs['y2']

        if self.options['solve_y1']:
            residuals['y1'] = (z0**2 + z1 + x - 0.2*y2) - y1
        else:
            residuals['y1'] = 0

        if self.options['solve_y2']:
            residuals['y2'] = (y1**.5 + z0 + z1) - y2
        else:
            residuals['y2'] = 0

        residuals['R_y1'] = (z0**2 + z1 + x - 0.2*y2) - y1 - outputs['R_y1']
        residuals['R_y2'] = (y1**.5 + z0 + z1) - y2 - outputs['R_y2']

    def linearize(self, inputs, outputs, J):

        # this will look wrong in check_partials if solve_y2 = False, but its not: R['y1'] = y1^* - y1
        J['y1', 'y1'] = -1.
        J['R_y1','R_y1'] = -1

        if self.options['solve_y1']:
            J['y1', 'x'] = [1]
            J['y1', 'z'] = [2*inputs['z'][0], 1]
            J['y1', 'y2'] = -0.2

        J['R_y1', 'x'] = [1]
        J['R_y1', 'z'] = [2*inputs['z'][0], 1]
        J['R_y1', 'y1'] = -1.
        J['R_y1', 'y2'] = -0.2

        # this will look wrong in check_partials if solve_y2 = False, but its not" R['y1'] = y2^* - y2
        J['y2','y2'] = -1

        J['R_y2','R_y2'] = -1
        if self.options['solve_y2']:
            J['y2','z'] = [1, 1]
            J['y2','y1'] = 0.5*outputs['y1']**-0.5

        J['R_y2','y2'] = -1
        J['R_y2','z'] = [1, 1]
        J['R_y2','y1'] = 0.5*outputs['y1']**-0.5

    def solve_nonlinear(self, inputs, outputs):
        z0 = inputs['z'][0]
        z1 = inputs['z'][1]
        x = inputs['x']
        y1 = outputs['y1']
        y2 = outputs['y2']

        outputs['R_y1'] = (z0**2 + z1 + x - 0.2*y2) - y1
        outputs['R_y2'] = (y1**.5 + z0 + z1) - y2



class TestComponentAsModel(unittest.TestCase):
    """
    This code shows various issues we encounter when the Model is a Component.

    Others include the N2 being wonky (but still generated as a report) in this situation.

    This behavior is deprecated and will be removed in a future release.
    """

    def test_subclass_as_model(self):

        class MyModel(om.Group):
            pass

        prob = om.Problem(model=MyModel())

        msg = "The model for this Problem is of type 'MyModel'. " \
              "A future release will require that the model " \
              "be a Group or a sub-class of Group."

        with assert_no_warning(OMDeprecationWarning, msg):
            prob.setup()

    def test_component_as_model(self):

        prob = om.Problem()

        prob.model = om.ExecComp('z = x + y')

        with assert_warning(OMDeprecationWarning, model_type_msg.format('ExecComp')):
            prob.setup()

        prob.set_val('x', 2)
        prob.set_val('y', 5)

        prob.run_model()

        assert_near_equal(7.0, prob.get_val('z'))

    def test_component_as_model_with_units(self):
        class TestComp(om.ExplicitComponent):
            def setup(self):
                self.add_input('foo', units='N')
                self.add_output('bar', units='N')
                self.declare_partials('bar', 'foo')

            def compute(self, inputs, outputs):
                outputs['bar'] = inputs['foo']

            def compute_partials(self, inputs, J):
                J['bar', 'foo'] = 1.

        p = om.Problem(model=TestComp())

        with assert_warning(OMDeprecationWarning, model_type_msg.format('TestComp')):
            p.setup()

        p.set_val('foo', 5, units='lbf')
        p.run_model()

        lbf_val = convert_units(5, 'lbf', 'N')
        self.assertEqual(p.get_val('foo'), lbf_val)
        self.assertEqual(p.get_val('bar'), lbf_val)

    def test_promotes_component_as_model(self):
        """
        This tests for an improved error message when 'promotes' args are inadvertantly passed
        to the ExecComp constructor rather than to 'add_subsystem'.

        The error came up in the context of using ExecComp as a model, but the error was unclear:

        Promotes error because there's nowhere to promote to, but that isn't necessarily clear.

        The error is

        RuntimeError: <model> <class ExecComp>: arg 'promotes_inputs' in call to ExecComp() does
        not refer to any variable in the expressions ['z = (x - 2)**2']

        Clearly that's not the case.
        """
        prob = om.Problem()

        prob.model = om.ExecComp('z = (x - 2)**2', promotes_inputs=['x'], promotes_outputs=['z'])

        with self.assertRaises(RuntimeError) as err:
            prob.setup()

        self.assertEqual(str(err.exception),
                         "<model> <class ExecComp>: arg 'promotes_inputs' in call to ExecComp() "
                         "does not refer to any variable in the expressions ['z = (x - 2)**2']. "
                         "Did you intend to promote variables in the 'add_subsystem' call?")

    def test_optimize_component_as_model(self):
        """
        The run_driver method requires that the model be a Group, so that it has an
        IndepVarComp to populate design variables.
        """
        prob = om.Problem(name='optimize_component_as_model')

        prob.model = om.ExecComp('z = (x - 2)**2')

        prob.model.add_design_var('x')
        prob.model.add_objective('z')

        with self.assertRaises(RuntimeError) as cm:
            prob.setup()

        msg = "\nCollected errors for problem 'optimize_component_as_model':\n   <model> <class ExecComp>: Output not found for design variable 'x'.\n" \
              "   <model> <class ExecComp>: The model is of type 'ExecComp'. Components must be placed in a Group " \
              "in order for unconnected inputs to be used as design variables. " \
              "A future release will require that the model be a Group or a sub-class of Group."
        self.assertEqual(str(cm.exception), msg)

    def test_optimize_empty_model(self):
        """
        Degenerate case, the model is an empty Group.

        No relevance or connection data will be initialized.
        """
        prob = om.Problem(driver=om.DOEDriver())
        prob.setup()
        prob.run_driver()

    def test_relevance_with_component_model(self):
        # Test relevance when model is a Component
        SOLVE_Y1 = False
        SOLVE_Y2 = True

        p_opt = om.Problem()

        p_opt.model = SellarOneComp(solve_y1=SOLVE_Y1, solve_y2=SOLVE_Y2)

        if SOLVE_Y1 or SOLVE_Y2:
            newton = p_opt.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
            newton.options['iprint'] = 0

        # NOTE: need to have this direct solver attached to the sellar comp until I define a solve_linear for it
        p_opt.model.linear_solver = om.DirectSolver(assemble_jac=True)

        p_opt.driver = om.ScipyOptimizeDriver()
        p_opt.driver.options['disp'] = False

        if not SOLVE_Y1:
            p_opt.model.add_design_var('y1', lower=-10, upper=10)
            p_opt.model.add_constraint('R_y1', equals=0)

        if not SOLVE_Y2:
            p_opt.model.add_design_var('y2', lower=-10, upper=10)
            p_opt.model.add_constraint('R_y2', equals=0)

        # this objective doesn't really matter... just need something there
        p_opt.model.add_objective('y2')

        p_opt.setup()

        # set
        p_opt['y2'] = 5
        p_opt['y1'] = 5

        p_opt.run_driver()

        np.testing.assert_almost_equal(p_opt['y1'][0], 2.109516506074582, decimal=5)
        np.testing.assert_almost_equal(p_opt['y2'][0], -0.5475825303740725, decimal=5)
        np.testing.assert_almost_equal(p_opt['x'][0], 2.0, decimal=5)
        np.testing.assert_almost_equal(p_opt['z'], np.array([-1., -1.]), decimal=5)


if __name__ == "__main__":
    unittest.main()
