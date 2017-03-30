"""Define the units/scaling tests."""
from __future__ import division, print_function

import unittest
from six import assertRaisesRegex

import numpy as np

from openmdao.api import Problem, Group, ExplicitComponent, ImplicitComponent, IndepVarComp
from openmdao.api import NewtonSolver, ScipyIterativeSolver, NonlinearBlockGS

from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayDense


class PassThroughLength(ExplicitComponent):
    """Units/scaling test component taking length in cm and passing it through in km."""

    def initialize_variables(self):
        self.add_input('old_length', val=1., units='cm')
        self.add_output('new_length', val=1., units='km', ref=0.1)

    def compute(self, inputs, outputs):
        length_cm = inputs['old_length']
        length_m = length_cm * 1e-2
        length_km = length_m * 1e-3
        outputs['new_length'] = length_km


class ScalingExample1(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., ref=1e2)
        self.add_output('y2', val=6000., ref=1e3)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1)/y1
        residuals['y2'] = 1e-5 * (x2 - y2)/y2


class ScalingExample2(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., res_ref=1e5)
        self.add_output('y2', val=6000., res_ref=1e-5)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1)/y1
        residuals['y2'] = 1e-5 * (x2 - y2)/y2

class ScalingExample3(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., ref=1e2, res_ref=1e5)
        self.add_output('y2', val=6000., ref=1e3, res_ref=1e-5)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1)/y1
        residuals['y2'] = 1e-5 * (x2 - y2)/y2

class ScalingExampleVector(ImplicitComponent):

    def initialize_variables(self):
        self.add_input('x', val=np.array([100., 5000.]))
        self.add_output('y', val=np.array([200., 6000.]),
                        ref=np.array([1e2, 1e3]),
                        res_ref=np.array([1e5, 1e-5]))

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']

        residuals['y'][0] = 1e5 * (x[0] - y[0])/y[0]
        residuals['y'][1] = 1e-5 * (x[1] - y[1])/y[1]


class SpeedComputationWithUnits(ExplicitComponent):
    """Simple speed computation from distance and time with unit conversations."""

    def initialize_variables(self):
        self.add_input('distance', 1.0, units='m')
        self.add_input('time', 1.0, units='s')
        self.add_output('speed', units='km/h')

    def compute(self, inputs, outputs):
        distance_m = inputs['distance']
        distance_km = distance_m * 1e-3

        time_s = inputs['time']
        time_h = time_s / 3600.

        speed_kph = distance_km / time_h
        outputs['speed'] = speed_kph


class ScalingTestComp(ImplicitComponent):
    """Explicit component used to test output and residual scaling.

    This component helps assemble a system of the following form with
    [ 10. r1 c1 ,  1. r1 c2 ] [u1] = [r1]
    [  1. r2 c1 , 10. r2 c2 ] [u2] = [r2]
    where r*, c* are parameters used to control where scaling is needed.

    This component computes one row of the above system.
    """

    def initialize_variables(self):
        self.metadata.declare('row', values=[1, 2])
        self.metadata.declare('coeffs')
        self.metadata.declare('use_scal', type_=bool)

        r1, r2, c1, c2 = self.metadata['coeffs']

        # We need to start at a different initial condition for different problems.
        init_state = 1.0

        # Scale the output based on the column coeff.
        if self.metadata['row'] == 1:
            ref = 1. / c1
            init_state = 1.0 / c1
        elif self.metadata['row'] == 2:
            ref = 1. / c2
            init_state = 1.0 / c2

        # Scale the output based on the column coeff.
        if self.metadata['row'] == 1:
            res_ref = r1
        elif self.metadata['row'] == 2:
            res_ref = r2

        # Overwrite to 1 if use_scal is False
        if not self.metadata['use_scal']:
            ref = 1.0
            res_ref = 1.0

        self.add_input('x')
        self.add_output('y', val = init_state, ref=ref, res_ref=res_ref)

    def apply_nonlinear(self, inputs, outputs, residuals):
        r1, r2, c1, c2 = self.metadata['coeffs']

        if self.metadata['row'] == 1:
            residuals['y'] = 10. * r1 * c1 * outputs['y'] + r1 * c2 * inputs['x'] - r1
        elif self.metadata['row'] == 2:
            residuals['y'] = 10. * r2 * c2 * outputs['y'] + r2 * c1 * inputs['x'] - r2

    def linearize(self, inputs, outputs, jacobian):
        r1, r2, c1, c2 = self.metadata['coeffs']

        if self.metadata['row'] == 1:
            jacobian['y', 'y'] = 10. * r1 * c1
            jacobian['y', 'x'] = r1 * c2
        if self.metadata['row'] == 2:
            jacobian['y', 'y'] = 10. * r2 * c2
            jacobian['y', 'x'] = r2 * c1


class TestScaling(unittest.TestCase):

    def test_error_messages(self):

        class EComp(ImplicitComponent):
            def initialize_variables(self):
                self.add_output('zz', val=np.ones((4, 2)), ref=np.ones((3, 5)))

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('comp', EComp())

        msg = "The ref argument has the wrong shape"
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

        class EComp(ImplicitComponent):
            def initialize_variables(self):
                self.add_output('zz', val=np.ones((4, 2)), ref0=np.ones((3, 5)))

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('comp', EComp())

        msg = "The ref0 argument has the wrong shape"
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

        class EComp(ImplicitComponent):
            def initialize_variables(self):
                self.add_output('zz', val=np.ones((4, 2)), res_ref=np.ones((3, 5)))

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('comp', EComp())

        msg = "The res_ref argument has the wrong shape"
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

        class EComp(ImplicitComponent):
            def initialize_variables(self):
                self.add_output('zz', val=np.ones((4, 2)), res_ref0=np.ones((3, 5)))

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('comp', EComp())

        msg = "The res_ref0 argument has the wrong shape"
        with assertRaisesRegex(self, ValueError, msg):
            prob.setup(check=False)

    def test_pass_through(self):
        group = Group()
        group.add_subsystem('sys1', IndepVarComp('old_length', 1.0,
                                                 units='mm', ref=1e5))
        group.add_subsystem('sys2', PassThroughLength())
        group.connect('sys1.old_length', 'sys2.old_length')

        prob = Problem(group)

        prob.setup(check=False)
        prob.model.suppress_solver_output = True

        prob['sys1.old_length'] = 3.e5
        assert_rel_error(self, prob['sys1.old_length'], 3.e5)
        assert_rel_error(self, prob.model._outputs['sys1.old_length'], 3.e5)
        prob.run_model()
        assert_rel_error(self, prob['sys2.new_length'], 3.e-1)
        assert_rel_error(self, prob.model._outputs['sys2.new_length'], 3.e-1)

    def test_speed(self):
        comp = IndepVarComp()
        comp.add_output('distance', 1., units='km')
        comp.add_output('time', 1., units='h')

        group = Group()
        group.add_subsystem('c1', comp)
        group.add_subsystem('c2', SpeedComputationWithUnits())
        group.connect('c1.distance', 'c2.distance')
        group.connect('c1.time', 'c2.time')

        prob = Problem(model=group)
        prob.setup(check=False)
        prob.model.suppress_solver_output = True

        prob.run_model()
        assert_rel_error(self, prob['c1.distance'], 1.0)  # units: km
        assert_rel_error(self, prob['c2.distance'], 1000.0)  # units: m
        assert_rel_error(self, prob['c1.time'], 1.0)  # units: h
        assert_rel_error(self, prob['c2.time'], 3600.0)  # units: s
        assert_rel_error(self, prob['c2.speed'], 1.0)  # units: km/h (i.e., kph)

    def test_scaling(self):
        """Test convergence in essentially one Newton iteration to atol=1e-5."""
        def runs_successfully(use_scal, coeffs):
            prob = Problem(model=Group())
            prob.model.add_subsystem('row1', ScalingTestComp(row=1, coeffs=coeffs,
                                                             use_scal=use_scal))
            prob.model.add_subsystem('row2', ScalingTestComp(row=2, coeffs=coeffs,
                                                             use_scal=use_scal))
            prob.model.connect('row1.y', 'row2.x')
            prob.model.connect('row2.y', 'row1.x')
            prob.model.nl_solver = NewtonSolver(maxiter=2, atol=1e-5, rtol=0)
            prob.model.nl_solver.ln_solver = ScipyIterativeSolver(maxiter=1)

            prob.model.suppress_solver_output = True

            prob.setup(check=False)
            result = prob.run_model()

            success = not result[0]
            return success

        # ---------------------------
        # coeffs: r1, r2, c1, c2
        coeffs = [1.e0, 1.e0, 1.e0, 1.e0]

        # Don't use scaling - but there's no need
        use_scal = False
        self.assertTrue(runs_successfully(use_scal, coeffs))
        # Use scaling - but there's no need
        use_scal = True
        self.assertTrue(runs_successfully(use_scal, coeffs))

        # ---------------------------
        # coeffs: r1, r2, c1, c2 - test output scaling:
        coeffs = [1.e0, 1.e0, 1.e10, 1.e0]

        # Don't use scaling - but output scaling needed
        use_scal = False
        self.assertTrue(not runs_successfully(use_scal, coeffs))
        # Use scaling - output scaling works successfully
        use_scal = True
        self.assertTrue(runs_successfully(use_scal, coeffs))

        # ---------------------------
        # coeffs: r1, r2, c1, c2 - test residual scaling:
        coeffs = [1.e10, 1.e0, 1.e10, 1.e0]

        # Don't use scaling - but residual scaling needed
        use_scal = False
        self.assertTrue(not runs_successfully(use_scal, coeffs))
        # Use scaling - residual scaling works successfully
        use_scal = True
        self.assertTrue(runs_successfully(use_scal, coeffs))

    def test_resid_scale_default(self):
        # This model checks the contents of the residual in both scaled and unscaled states.
        # The model is a cycle that iterates once, so the first component in the cycle carries
        # a residual.

        class Simple(ExplicitComponent):

            def __init__(self, ref=1.0, res_ref=None, ref0=0.0, res_ref0=None, **kwargs):

                kwargs['ref'] = ref
                kwargs['ref0'] = ref0
                kwargs['res_ref'] = res_ref
                kwargs['res_ref0'] = res_ref0

                super(Simple, self).__init__(**kwargs)

            def initialize_variables(self):

                ref = self.metadata['ref']
                ref0 = self.metadata['ref0']
                res_ref = self.metadata['res_ref']
                res_ref0 = self.metadata['res_ref0']

                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0, ref=ref, ref0=ref0, res_ref=res_ref, res_ref0=res_ref0)

            def compute(self, inputs, outputs):
                outputs['y'] = 2.0*(inputs['x'] + 1.0)

        # Baseline - all should be equal.

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', Simple())
        model.add_subsystem('p2', Simple())
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nl_solver = NonlinearBlockGS()
        model.nl_solver.options['maxiter'] = 1

        model._suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        res1 = model.get_subsystem('p1')._residuals.get_data()[0]
        out1 = model.get_subsystem('p1')._outputs.get_data()[0]
        out2 = model.get_subsystem('p2')._outputs.get_data()[0]

        self.assertEqual(res1, out1 - 2.0*(out2 + 1.0))
        with model._scaled_context():
            res1 = model.get_subsystem('p1')._residuals.get_data()[0]
            out1 = model.get_subsystem('p1')._outputs.get_data()[0]
            out2 = model.get_subsystem('p2')._outputs.get_data()[0]

            self.assertEqual(res1, out1 - 2.0*(out2 + 1.0))

        # Scale the outputs only.
        # Residual scaling uses output scaling by default.

        ref = 1.0
        ref0 = 1.5

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', Simple(ref=ref, ref0=ref0))
        model.add_subsystem('p2', Simple(ref=ref, ref0=ref0))
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nl_solver = NonlinearBlockGS()
        model.nl_solver.options['maxiter'] = 1

        model._suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        res1 = model.get_subsystem('p1')._residuals.get_data()[0]
        out1 = model.get_subsystem('p1')._outputs.get_data()[0]
        out2 = model.get_subsystem('p2')._outputs.get_data()[0]

        self.assertEqual(res1, (out1 - 2.0*(out2 + 1.0)))
        with model._scaled_context():
            res1a = model.get_subsystem('p1')._residuals.get_data()[0]

            self.assertEqual(res1a, (res1-ref0)/(ref-ref0))

        # Scale the residual

        res_ref = 4.0
        res_ref0 = 3.5

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', Simple(res_ref=res_ref, res_ref0=res_ref0))
        model.add_subsystem('p2', Simple(res_ref=res_ref, res_ref0=res_ref0))
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nl_solver = NonlinearBlockGS()
        model.nl_solver.options['maxiter'] = 1

        model._suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        res1 = model.get_subsystem('p1')._residuals.get_data()[0]
        out1 = model.get_subsystem('p1')._outputs.get_data()[0]
        out2 = model.get_subsystem('p2')._outputs.get_data()[0]

        self.assertEqual(res1, out1 - 2.0*(out2+1.0))
        with model._scaled_context():
            res1a = model.get_subsystem('p1')._residuals.get_data()[0]

            self.assertEqual(res1a, (res1-res_ref0)/(res_ref-res_ref0))

        # Simultaneously scale the residual and output with different values

        ref = 3.0
        ref0 = 2.75
        res_ref = 4.0
        res_ref0 = 3.5

        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('p1', Simple(ref=ref, ref0=ref0, res_ref=res_ref, res_ref0=res_ref0))
        model.add_subsystem('p2', Simple(ref=ref, ref0=ref0, res_ref=res_ref, res_ref0=res_ref0))
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nl_solver = NonlinearBlockGS()
        model.nl_solver.options['maxiter'] = 1

        model._suppress_solver_output = True

        prob.setup(check=False)
        prob.run_model()

        res1 = model.get_subsystem('p1')._residuals.get_data()[0]
        out1 = model.get_subsystem('p1')._outputs.get_data()[0]
        out2 = model.get_subsystem('p2')._outputs.get_data()[0]

        self.assertEqual(res1, out1 - 2.0*(out2+1.0))
        with model._scaled_context():
            res1a = model.get_subsystem('p1')._residuals.get_data()[0]

            self.assertEqual(res1a, (res1-res_ref0)/(res_ref-res_ref0))

    def test_scale_array_with_float(self):

        class ExpCompArrayScale(TestExplCompArrayDense):

            def initialize_variables(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((2, 2)))
                self.add_output('areas', val=np.ones((2, 2)), ref=2.0)
                self.add_output('stuff', val=np.ones((2, 2)), ref=3.0)
                self.add_output('total_volume', val=1.)

            def compute(self, inputs, outputs):
                super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] + inputs['lengths']

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('comp', ExpCompArrayScale())
        model.connect('p1.x', 'comp.lengths')

        prob.setup(check=False)
        prob['comp.widths'] = np.ones((2, 2))
        prob.run_model()

        assert_rel_error(self, prob['comp.total_volume'], 4.)

        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['areas']
            assert_rel_error(self, val[0, 0], 0.5)
            assert_rel_error(self, val[0, 1], 0.5)
            assert_rel_error(self, val[1, 0], 0.5)
            assert_rel_error(self, val[1, 1], 0.5)

            val = model.get_subsystem('comp')._outputs['stuff']
            assert_rel_error(self, val[0, 0], 2.0/3)
            assert_rel_error(self, val[0, 1], 2.0/3)
            assert_rel_error(self, val[1, 0], 2.0/3)
            assert_rel_error(self, val[1, 1], 2.0/3)

    def test_scale_array_with_array(self):

        class ExpCompArrayScale(TestExplCompArrayDense):

            def initialize_variables(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((2, 2)))
                self.add_output('areas', val=np.ones((2, 2)), ref=np.array([[2.0, 3.0], [5.0, 7.0]]))
                self.add_output('stuff', val=np.ones((2, 2)), ref=np.array([[11.0, 13.0], [17.0, 19.0]]))
                self.add_output('total_volume', val=1.)

            def compute(self, inputs, outputs):
                super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] + inputs['lengths']

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('comp', ExpCompArrayScale())
        model.connect('p1.x', 'comp.lengths')

        prob.setup(check=False)
        prob['comp.widths'] = np.ones((2, 2))
        prob.run_model()

        assert_rel_error(self, prob['comp.total_volume'], 4.)

        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['areas']
            assert_rel_error(self, val[0, 0], 1.0/2)
            assert_rel_error(self, val[0, 1], 1.0/3)
            assert_rel_error(self, val[1, 0], 1.0/5)
            assert_rel_error(self, val[1, 1], 1.0/7)

            val = model.get_subsystem('comp')._outputs['stuff']
            assert_rel_error(self, val[0, 0], 2.0/11)
            assert_rel_error(self, val[0, 1], 2.0/13)
            assert_rel_error(self, val[1, 0], 2.0/17)
            assert_rel_error(self, val[1, 1], 2.0/19)

    def test_scale_and_add_array_with_array(self):

        class ExpCompArrayScale(TestExplCompArrayDense):

            def initialize_variables(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((2, 2)))
                self.add_output('areas', val=np.ones((2, 2)), ref=np.array([[2.0, 3.0], [5.0, 7.0]]),
                                ref0=np.array([[0.1, 0.2], [0.3, 0.4]]), lower=-1000.0, upper=1000.0)
                self.add_output('stuff', val=np.ones((2, 2)), ref=np.array([[11.0, 13.0], [17.0, 19.0]]),
                                ref0=np.array([[0.6, 0.7], [0.8, 0.9]]),
                                lower=np.array([[-5000.0, -4000.0], [-3000.0, -2000.0]]),
                                upper=np.array([[5000.0, 4000.0], [3000.0, 2000.0]]))
                self.add_output('total_volume', val=1.)

            def compute(self, inputs, outputs):
                super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] + inputs['lengths']

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('comp', ExpCompArrayScale())
        model.connect('p1.x', 'comp.lengths')

        prob.setup(check=False)
        prob['comp.widths'] = np.ones((2, 2))
        prob.run_model()

        assert_rel_error(self, prob['comp.total_volume'], 4.)

        # (res1-res_ref0)/(res_ref-res_ref0))
        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['areas']
            assert_rel_error(self, val[0, 0], (1.0 - 0.1)/(2 - 0.1))
            assert_rel_error(self, val[0, 1], (1.0 - 0.2)/(3 - 0.2))
            assert_rel_error(self, val[1, 0], (1.0 - 0.3)/(5 - 0.3))
            assert_rel_error(self, val[1, 1], (1.0 - 0.4)/(7 - 0.4))

            val = model.get_subsystem('comp')._outputs['stuff']
            assert_rel_error(self, val[0, 0], (2.0 - 0.6)/(11 - 0.6))
            assert_rel_error(self, val[0, 1], (2.0 - 0.7)/(13 - 0.7))
            assert_rel_error(self, val[1, 0], (2.0 - 0.8)/(17 - 0.8))
            assert_rel_error(self, val[1, 1], (2.0 - 0.9)/(19 - 0.9))

            lb = model.get_subsystem('comp')._lower_bounds['areas']
            assert_rel_error(self, lb[0, 0], (-1000.0 - 0.1)/(2 - 0.1))
            assert_rel_error(self, lb[0, 1], (-1000.0 - 0.2)/(3 - 0.2))
            assert_rel_error(self, lb[1, 0], (-1000.0 - 0.3)/(5 - 0.3))
            assert_rel_error(self, lb[1, 1], (-1000.0 - 0.4)/(7 - 0.4))

            ub = model.get_subsystem('comp')._upper_bounds['areas']
            assert_rel_error(self, ub[0, 0], (1000.0 - 0.1)/(2 - 0.1))
            assert_rel_error(self, ub[0, 1], (1000.0 - 0.2)/(3 - 0.2))
            assert_rel_error(self, ub[1, 0], (1000.0 - 0.3)/(5 - 0.3))
            assert_rel_error(self, ub[1, 1], (1000.0 - 0.4)/(7 - 0.4))

            lb = model.get_subsystem('comp')._lower_bounds['stuff']
            assert_rel_error(self, lb[0, 0], (-5000.0 - 0.6)/(11 - 0.6))
            assert_rel_error(self, lb[0, 1], (-4000.0 - 0.7)/(13 - 0.7))
            assert_rel_error(self, lb[1, 0], (-3000.0 - 0.8)/(17 - 0.8))
            assert_rel_error(self, lb[1, 1], (-2000.0 - 0.9)/(19 - 0.9))

            ub = model.get_subsystem('comp')._upper_bounds['stuff']
            assert_rel_error(self, ub[0, 0], (5000.0 - 0.6)/(11 - 0.6))
            assert_rel_error(self, ub[0, 1], (4000.0 - 0.7)/(13 - 0.7))
            assert_rel_error(self, ub[1, 0], (3000.0 - 0.8)/(17 - 0.8))
            assert_rel_error(self, ub[1, 1], (2000.0 - 0.9)/(19 - 0.9))

    def test_implicit_scale(self):

        class ImpCompArrayScale(TestImplCompArrayDense):
            def initialize_variables(self):
                self.add_input('rhs', val=np.ones(2))
                self.add_output('x', val=np.zeros(2), ref=np.array([2.0, 3.0]),
                                ref0=np.array([4.0, 9.0]),
                                res_ref=np.array([7.0, 11.0]),
                                res_ref0=np.array([13.0, 18.0]))
                self.add_output('extra', val=np.zeros(2), ref=np.array([12.0, 13.0]),
                                ref0=np.array([14.0, 17.0]))

            def apply_nonlinear(self, inputs, outputs, residuals):
                super(ImpCompArrayScale, self).apply_nonlinear(inputs, outputs, residuals)
                residuals['extra'] = 2.0*self.metadata['mtx'].dot(outputs['x']) - 3.0*inputs['rhs']

            def linearize(self, inputs, outputs, jacobian):
                # These are incorrect derivatives, but we aren't doing any calculations, and it makes
                # it much easier to check that the scales are correct.
                jacobian['x', 'x'] = np.ones((2, 2))
                jacobian['x', 'extra'] = np.ones((2, 2))
                jacobian['extra', 'x'] = np.ones((2, 2))
                jacobian['x', 'rhs'] = -np.eye(2)


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImpCompArrayScale())
        model.connect('p1.x', 'comp.rhs')

        prob.setup(check=False)
        prob.run_model()

        base_x = model.get_subsystem('comp')._outputs['x'].copy()
        base_ex = model.get_subsystem('comp')._outputs['extra'].copy()
        base_res_x = model.get_subsystem('comp')._residuals['x'].copy()
        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['x']
            assert_rel_error(self, val[0], (base_x[0] - 4.0)/(2.0 - 4.0))
            assert_rel_error(self, val[1], (base_x[1] - 9.0)/(3.0 - 9.0))
            val = model.get_subsystem('comp')._outputs['extra']
            assert_rel_error(self, val[0], (base_ex[0] - 14.0)/(12.0 - 14.0))
            assert_rel_error(self, val[1], (base_ex[1] - 17.0)/(13.0 - 17.0))
            val = model.get_subsystem('comp')._residuals['x'].copy()
            assert_rel_error(self, val[0], (base_res_x[0] - 13.0)/(7.0 - 13.0))
            assert_rel_error(self, val[1], (base_res_x[1] - 18.0)/(11.0 - 18.0))

        model.run_linearize()

        with model._scaled_context():
            subjacs = comp.jacobian._subjacs

            assert_rel_error(self, subjacs['comp.x', 'comp.x'][0][0], (2.0 - 4.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.x'][1][0], (2.0 - 4.0)/(11.0 - 18.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.x'][0][1], (3.0 - 9.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.x'][1][1], (3.0 - 9.0)/(11.0 - 18.0))

            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][0][0], (12.0 - 14.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][1][0], (12.0 - 14.0)/(11.0 - 18.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][0][1], (13.0 - 17.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][1][1], (13.0 - 17.0)/(11.0 - 18.0))

            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][0][0], -1.0/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][1][0], 0.0)
            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][0][1], 0.0)
            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][1][1], -1.0/(11.0 - 18.0))

    def test_implicit_scale_with_scalar_jac(self):
        raise unittest.SkipTest('Cannot specify an n by m subjac with a scalar yet.')

        class ImpCompArrayScale(TestImplCompArrayDense):
            def initialize_variables(self):
                self.add_input('rhs', val=np.ones(2))
                self.add_output('x', val=np.zeros(2), ref=np.array([2.0, 3.0]),
                                ref0=np.array([4.0, 9.0]),
                                res_ref=np.array([7.0, 11.0]),
                                res_ref0=np.array([13.0, 18.0]))
                self.add_output('extra', val=np.zeros(2), ref=np.array([12.0, 13.0]),
                                ref0=np.array([14.0, 17.0]))

            def apply_nonlinear(self, inputs, outputs, residuals):
                super(ImpCompArrayScale, self).apply_nonlinear(inputs, outputs, residuals)
                residuals['extra'] = 2.0*self.metadata['mtx'].dot(outputs['x']) - 3.0*inputs['rhs']

            def linearize(self, inputs, outputs, jacobian):
                # These are incorrect derivatives, but we aren't doing any calculations, and it makes
                # it much easier to check that the scales are correct.
                jacobian['x', 'x'][:] = 1.0
                jacobian['x', 'extra'][:] = 1.0
                jacobian['extra', 'x'][:] = 1.0
                jacobian['x', 'rhs'] = -np.eye(2)


        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImpCompArrayScale())
        model.connect('p1.x', 'comp.rhs')

        prob.setup(check=False)
        prob.run_model()

        base_x = model.get_subsystem('comp')._outputs['x'].copy()
        base_ex = model.get_subsystem('comp')._outputs['extra'].copy()
        base_res_x = model.get_subsystem('comp')._residuals['x'].copy()
        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['x']
            assert_rel_error(self, val[0], (base_x[0] - 4.0)/(2.0 - 4.0))
            assert_rel_error(self, val[1], (base_x[1] - 9.0)/(3.0 - 9.0))
            val = model.get_subsystem('comp')._outputs['extra']
            assert_rel_error(self, val[0], (base_ex[0] - 14.0)/(12.0 - 14.0))
            assert_rel_error(self, val[1], (base_ex[1] - 17.0)/(13.0 - 17.0))
            val = model.get_subsystem('comp')._residuals['x'].copy()
            assert_rel_error(self, val[0], (base_res_x[0] - 13.0)/(7.0 - 13.0))
            assert_rel_error(self, val[1], (base_res_x[1] - 18.0)/(11.0 - 18.0))

        model.run_linearize()

        with model._scaled_context():
            subjacs = comp.jacobian._subjacs

            assert_rel_error(self, subjacs['comp.x', 'comp.x'][0][0], (2.0 - 4.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.x'][1][0], (2.0 - 4.0)/(11.0 - 18.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.x'][0][1], (3.0 - 9.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.x'][1][1], (3.0 - 9.0)/(11.0 - 18.0))

            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][0][0], (12.0 - 14.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][1][0], (12.0 - 14.0)/(11.0 - 18.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][0][1], (13.0 - 17.0)/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.extra'][1][1], (13.0 - 17.0)/(11.0 - 18.0))

            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][0][0], -1.0/(7.0 - 13.0))
            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][1][0], 0.0)
            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][0][1], 0.0)
            assert_rel_error(self, subjacs['comp.x', 'comp.rhs'][1][1], -1.0/(11.0 - 18.0))

    def test_feature1(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x1', 1.0))
        model.add_subsystem('p2', IndepVarComp('x2', 1.0))
        comp = model.add_subsystem('comp', ScalingExample1())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['y1']
            assert_rel_error(self, val, 2.0)
            val = model.get_subsystem('comp')._outputs['y2']
            assert_rel_error(self, val, 6.0)

    def test_feature2(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x1', 1.0))
        model.add_subsystem('p2', IndepVarComp('x2', 1.0))
        comp = model.add_subsystem('comp', ScalingExample2())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context():
            val = model.get_subsystem('comp')._outputs['y1']
            assert_rel_error(self, val, 200.0)
            val = model.get_subsystem('comp')._outputs['y2']
            assert_rel_error(self, val, 6000.0)

    def test_feature3(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p1', IndepVarComp('x1', 1.0))
        model.add_subsystem('p2', IndepVarComp('x2', 1.0))
        comp = model.add_subsystem('comp', ScalingExample3())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        prob.setup(check=False)
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context():
            val = model.get_subsystem('comp')._residuals['y1']
            assert_rel_error(self, val, -.995)
            val = model.get_subsystem('comp')._residuals['y2']
            assert_rel_error(self, val, (1-6000.)/6000.)

    def test_feature_vector(self):

        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('p', IndepVarComp('x', np.ones((2))))
        comp = model.add_subsystem('comp', ScalingExampleVector())
        model.connect('p.x', 'comp.x')

        prob.setup(check=False)
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context():
            val = model.get_subsystem('comp')._residuals['y']
            assert_rel_error(self, val[0], (1-200.)/200.)
            assert_rel_error(self, val[1], (1-6000.)/6000.)
            val = model.get_subsystem('comp')._outputs['y']
            assert_rel_error(self, val[0], 2.0)
            assert_rel_error(self, val[1], 6.0)

if __name__ == '__main__':
    unittest.main()
