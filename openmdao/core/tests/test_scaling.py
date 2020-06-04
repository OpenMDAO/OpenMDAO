"""Define the units/scaling tests."""
import unittest
from copy import deepcopy

import numpy as np

import openmdao.api as om
from openmdao.core.driver import Driver

from openmdao.test_suite.components.expl_comp_array import TestExplCompArrayDense
from openmdao.test_suite.components.impl_comp_array import TestImplCompArrayDense
from openmdao.utils.assert_utils import assert_near_equal


class PassThroughLength(om.ExplicitComponent):
    """Units/scaling test component taking length in cm and passing it through in km."""

    def setup(self):
        self.add_input('old_length', val=1., units='cm')
        self.add_output('new_length', val=1., units='km', ref=0.1)

    def compute(self, inputs, outputs):
        length_cm = inputs['old_length']
        length_m = length_cm * 1e-2
        length_km = length_m * 1e-3
        outputs['new_length'] = length_km


class ScalingExample1(om.ImplicitComponent):

    def setup(self):
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


class ScalingExample2(om.ImplicitComponent):

    def setup(self):
        self.add_input('x1', val=100.0)
        self.add_input('x2', val=5000.0)
        self.add_output('y1', val=200., ref=300.0, ref0=100.0)
        self.add_output('y2', val=6000., ref=11000.0, ref0=1000.0)

    def apply_nonlinear(self, inputs, outputs, residuals):
        x1 = inputs['x1']
        x2 = inputs['x2']
        y1 = outputs['y1']
        y2 = outputs['y2']

        residuals['y1'] = 1e5 * (x1 - y1)/y1
        residuals['y2'] = 1e-5 * (x2 - y2)/y2


class ScalingExample3(om.ImplicitComponent):

    def setup(self):
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


class ScalingExampleVector(om.ImplicitComponent):

    def setup(self):
        self.add_input('x', val=np.array([100., 5000.]))
        self.add_output('y', val=np.array([200., 6000.]),
                        ref=np.array([1e2, 1e3]),
                        res_ref=np.array([1e5, 1e-5]))

    def apply_nonlinear(self, inputs, outputs, residuals):
        x = inputs['x']
        y = outputs['y']

        residuals['y'][0] = 1e5 * (x[0] - y[0])/y[0]
        residuals['y'][1] = 1e-5 * (x[1] - y[1])/y[1]


class SpeedComputationWithUnits(om.ExplicitComponent):
    """Simple speed computation from distance and time with unit conversations."""

    def setup(self):
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


class ScalingTestComp(om.ImplicitComponent):
    """Explicit component used to test output and residual scaling.

    This component helps assemble a system of the following form with
    [ 10. r1 c1 ,  1. r1 c2 ] [u1] = [r1]
    [  1. r2 c1 , 10. r2 c2 ] [u2] = [r2]
    where r*, c* are parameters used to control where scaling is needed.

    This component computes one row of the above system.
    """

    def initialize(self):
        self.options.declare('row', values=[1, 2])
        self.options.declare('coeffs')
        self.options.declare('use_scal', types=bool)

    def setup(self):

        r1, r2, c1, c2 = self.options['coeffs']

        # We need to start at a different initial condition for different problems.
        init_state = 1.0

        # Scale the output based on the column coeff.
        if self.options['row'] == 1:
            ref = 1. / c1
            init_state = 1.0 / c1
        elif self.options['row'] == 2:
            ref = 1. / c2
            init_state = 1.0 / c2

        # Scale the output based on the column coeff.
        if self.options['row'] == 1:
            res_ref = r1
        elif self.options['row'] == 2:
            res_ref = r2

        # Overwrite to 1 if use_scal is False
        if not self.options['use_scal']:
            ref = 1.0
            res_ref = 1.0

        self.add_input('x')
        self.add_output('y', val=init_state, ref=ref, res_ref=res_ref)

        self.declare_partials('*', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        r1, r2, c1, c2 = self.options['coeffs']

        if self.options['row'] == 1:
            residuals['y'] = 10. * r1 * c1 * outputs['y'] + r1 * c2 * inputs['x'] - r1
        elif self.options['row'] == 2:
            residuals['y'] = 10. * r2 * c2 * outputs['y'] + r2 * c1 * inputs['x'] - r2

    def linearize(self, inputs, outputs, jacobian):
        r1, r2, c1, c2 = self.options['coeffs']

        if self.options['row'] == 1:
            jacobian['y', 'y'] = 10. * r1 * c1
            jacobian['y', 'x'] = r1 * c2
        if self.options['row'] == 2:
            jacobian['y', 'y'] = 10. * r2 * c2
            jacobian['y', 'x'] = r2 * c1


def _winfix(s):
    """clean up the string on Windows"""
    return s.replace('2L', '2').replace('3L', '3').replace('4L', '4').replace('5L', '5')


class TestScaling(unittest.TestCase):

    def test_error_messages(self):

        class EComp(om.ImplicitComponent):
            def setup(self):
                self.add_output('zz', val=np.ones((4, 2)), ref=np.ones((3, 5)))

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', EComp())

        msg = "EComp (comp): When adding output 'zz', expected shape (4, 2) but got shape (3, 5) for argument 'ref'."
        with self.assertRaises(ValueError) as context:
            prob.setup()
        self.assertEqual(_winfix(str(context.exception)), msg)

        class EComp(om.ImplicitComponent):
            def setup(self):
                self.add_output('zz', val=np.ones((4, 2)), ref0=np.ones((3, 5)))

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', EComp())

        msg = "EComp (comp): When adding output 'zz', expected shape (4, 2) but got shape (3, 5) for argument 'ref0'."
        with self.assertRaises(ValueError) as context:
            prob.setup()
        self.assertEqual(_winfix(str(context.exception)), msg)

        class EComp(om.ImplicitComponent):
            def setup(self):
                self.add_output('zz', val=np.ones((4, 2)), res_ref=np.ones((3, 5)))

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('comp', EComp())

        msg = "EComp (comp): When adding output 'zz', expected shape (4, 2) but got shape (3, 5) for argument 'res_ref'."
        with self.assertRaises(ValueError) as context:
            prob.setup()
        self.assertEqual(_winfix(str(context.exception)), msg)

    def test_pass_through(self):
        group = om.Group()
        group.add_subsystem('sys1', om.IndepVarComp('old_length', 1.0,
                                                    units='mm', ref=1e5))
        group.add_subsystem('sys2', PassThroughLength())
        group.connect('sys1.old_length', 'sys2.old_length')

        prob = om.Problem(group)

        prob.setup()
        prob.set_solver_print(level=0)

        prob['sys1.old_length'] = 3.e5
        prob.final_setup()

        assert_near_equal(prob['sys1.old_length'], 3.e5)
        assert_near_equal(prob.model._outputs['sys1.old_length'], 3.e5)
        prob.run_model()
        assert_near_equal(prob['sys2.new_length'], 3.e-1)
        assert_near_equal(prob.model._outputs['sys2.new_length'], 3.e-1)

    def test_speed(self):
        comp = om.IndepVarComp()
        comp.add_output('distance', 1., units='km')
        comp.add_output('time', 1., units='h')

        group = om.Group()
        group.add_subsystem('c1', comp)
        group.add_subsystem('c2', SpeedComputationWithUnits())
        group.connect('c1.distance', 'c2.distance')
        group.connect('c1.time', 'c2.time')

        prob = om.Problem(model=group)
        prob.setup()
        prob.set_solver_print(level=0)

        prob.run_model()
        assert_near_equal(prob['c1.distance'], 1.0)  # units: km
        assert_near_equal(prob['c2.distance'], 1000.0)  # units: m
        assert_near_equal(prob['c1.time'], 1.0)  # units: h
        assert_near_equal(prob['c2.time'], 3600.0)  # units: s
        assert_near_equal(prob['c2.speed'], 1.0)  # units: km/h (i.e., kph)

    def test_scaling(self):
        """Test convergence in essentially one Newton iteration to atol=1e-5."""
        def runs_successfully(use_scal, coeffs):
            prob = om.Problem()
            prob.model.add_subsystem('row1', ScalingTestComp(row=1, coeffs=coeffs,
                                                             use_scal=use_scal))
            prob.model.add_subsystem('row2', ScalingTestComp(row=2, coeffs=coeffs,
                                                             use_scal=use_scal))
            prob.model.connect('row1.y', 'row2.x')
            prob.model.connect('row2.y', 'row1.x')
            prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, maxiter=2, atol=1e-5, rtol=0)
            prob.model.nonlinear_solver.linear_solver = om.ScipyKrylov(maxiter=1)

            prob.set_solver_print(level=0)

            prob.setup()
            prob.run_model()

            return np.linalg.norm(prob.model._residuals._data) < 1e-5

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

        class Simple(om.ExplicitComponent):

            def initialize(self):
                self.options.declare('ref', default=1.0)
                self.options.declare('ref0', default=0.0)
                self.options.declare('res_ref', default=None)
                self.options.declare('res_ref0', default=None)

            def setup(self):

                ref = self.options['ref']
                ref0 = self.options['ref0']
                res_ref = self.options['res_ref']

                self.add_input('x', val=1.0)
                self.add_output('y', val=1.0, ref=ref, ref0=ref0, res_ref=res_ref)

                self.declare_partials('*', '*')

            def compute(self, inputs, outputs):
                outputs['y'] = 2.0*(inputs['x'] + 1.0)

            def compute_partials(self, inputs, partials):
                """
                Jacobian for Sellar discipline 1.
                """
                partials['y', 'x'] = 2.0


        # Baseline - all should be equal.

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', Simple())
        model.add_subsystem('p2', Simple())
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.nonlinear_solver.options['maxiter'] = 1
        model.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        res1 = -model.p1._residuals._data[0]
        out1 = model.p1._outputs._data[0]
        out2 = model.p2._outputs._data[0]

        self.assertEqual(res1, out1 - 2.0*(out2 + 1.0))
        with model._scaled_context_all():
            res1 = -model.p1._residuals._data[0]
            out1 = model.p1._outputs._data[0]
            out2 = model.p2._outputs._data[0]

            self.assertEqual(res1, out1 - 2.0*(out2 + 1.0))

        # Jacobian is unscaled
        prob.model.run_linearize()
        deriv = model.p1._jacobian
        assert_near_equal(deriv['p1.y', 'p1.x'], [[2.0]])

        # Scale the outputs only.
        # Residual scaling uses output scaling by default.

        ref = 1.0
        ref0 = 1.5

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', Simple(ref=ref, ref0=ref0))
        model.add_subsystem('p2', Simple(ref=ref, ref0=ref0))
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.nonlinear_solver.options['maxiter'] = 1
        model.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        res1 = -model.p1._residuals._data[0]
        out1 = model.p1._outputs._data[0]
        out2 = model.p2._outputs._data[0]

        self.assertEqual(res1, (out1 - 2.0*(out2 + 1.0)))
        with model._scaled_context_all():
            res1a = -model.p1._residuals._data[0]

            self.assertEqual(res1a, (res1)/(ref))

        # Jacobian is unscaled
        prob.model.run_linearize()
        deriv = model.p1._jacobian
        assert_near_equal(deriv['p1.y', 'p1.x'], [[2.0]])

        # Scale the residual

        res_ref = 4.0

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', Simple(res_ref=res_ref))
        model.add_subsystem('p2', Simple(res_ref=res_ref))
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.nonlinear_solver.options['maxiter'] = 1
        model.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        res1 = -model.p1._residuals._data[0]
        out1 = model.p1._outputs._data[0]
        out2 = model.p2._outputs._data[0]

        self.assertEqual(res1, out1 - 2.0*(out2+1.0))
        with model._scaled_context_all():
            res1a = -model.p1._residuals._data[0]

            self.assertEqual(res1a, res1/res_ref)

        # Jacobian is unscaled
        prob.model.run_linearize()
        deriv = model.p1._jacobian
        assert_near_equal(deriv['p1.y', 'p1.x'], [[2.0]])

        # Simultaneously scale the residual and output with different values

        ref = 3.0
        ref0 = 2.75
        res_ref = 4.0

        prob = om.Problem()
        model = prob.model
        model.add_subsystem('p1', Simple(ref=ref, ref0=ref0, res_ref=res_ref))
        model.add_subsystem('p2', Simple(ref=ref, ref0=ref0, res_ref=res_ref))
        model.connect('p1.y', 'p2.x')
        model.connect('p2.y', 'p1.x')

        model.nonlinear_solver = om.NonlinearBlockGS()
        model.nonlinear_solver.options['maxiter'] = 1
        model.nonlinear_solver.options['use_apply_nonlinear'] = True

        prob.set_solver_print(level=0)

        prob.setup()
        prob.run_model()

        res1 = -model.p1._residuals._data[0]
        out1 = model.p1._outputs._data[0]
        out2 = model.p2._outputs._data[0]

        self.assertEqual(res1, out1 - 2.0*(out2+1.0))
        with model._scaled_context_all():
            res1a = -model.p1._residuals._data[0]

            self.assertEqual(res1a, (res1)/(res_ref))

        # Jacobian is unscaled
        prob.model.run_linearize()
        deriv = model.p1._jacobian
        assert_near_equal(deriv['p1.y', 'p1.x'], [[2.0]])

    def test_scale_array_with_float(self):

        class ExpCompArrayScale(TestExplCompArrayDense):

            def setup(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((2, 2)))
                self.add_output('areas', val=np.ones((2, 2)), ref=2.0)
                self.add_output('stuff', val=np.ones((2, 2)), ref=3.0)
                self.add_output('total_volume', val=1.)

            def compute(self, inputs, outputs):
                super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] + inputs['lengths']

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('comp', ExpCompArrayScale())
        model.connect('p1.x', 'comp.lengths')

        prob.setup()
        prob['comp.widths'] = np.ones((2, 2))

        prob.run_model()

        assert_near_equal(prob['comp.total_volume'], 4.)

        with model._scaled_context_all():
            val = model.comp._outputs['areas']
            assert_near_equal(val[0, 0], 0.5)
            assert_near_equal(val[0, 1], 0.5)
            assert_near_equal(val[1, 0], 0.5)
            assert_near_equal(val[1, 1], 0.5)

            val = model.comp._outputs['stuff']
            assert_near_equal(val[0, 0], 2.0/3)
            assert_near_equal(val[0, 1], 2.0/3)
            assert_near_equal(val[1, 0], 2.0/3)
            assert_near_equal(val[1, 1], 2.0/3)

    def test_scale_array_with_array(self):

        class ExpCompArrayScale(TestExplCompArrayDense):

            def setup(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((2, 2)))
                self.add_output('areas', val=np.ones((2, 2)), ref=np.array([[2.0, 3.0], [5.0, 7.0]]))
                self.add_output('stuff', val=np.ones((2, 2)), ref=np.array([[11.0, 13.0], [17.0, 19.0]]))
                self.add_output('total_volume', val=1.)

            def compute(self, inputs, outputs):
                super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] + inputs['lengths']

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('comp', ExpCompArrayScale())
        model.connect('p1.x', 'comp.lengths')

        prob.setup()
        prob['comp.widths'] = np.ones((2, 2))
        prob.run_model()

        assert_near_equal(prob['comp.total_volume'], 4.)

        with model._scaled_context_all():
            val = model.comp._outputs['areas']
            assert_near_equal(val[0, 0], 1.0/2)
            assert_near_equal(val[0, 1], 1.0/3)
            assert_near_equal(val[1, 0], 1.0/5)
            assert_near_equal(val[1, 1], 1.0/7)

            val = model.comp._outputs['stuff']
            assert_near_equal(val[0, 0], 2.0/11)
            assert_near_equal(val[0, 1], 2.0/13)
            assert_near_equal(val[1, 0], 2.0/17)
            assert_near_equal(val[1, 1], 2.0/19)

    def test_scale_and_add_array_with_array(self):

        class ExpCompArrayScale(TestExplCompArrayDense):

            def setup(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((2, 2)))
                self.add_output('areas', val=np.ones((2, 2)), ref=np.array([[2.0, 3.0], [5.0, 7.0]]),
                                ref0=np.array([[0.1, 0.2], [0.3, 0.4]]), lower=-1000.0, upper=1000.0)
                self.add_output('stuff', val=np.ones((2, 2)), ref=np.array([[11.0, 13.0], [17.0, 19.0]]),
                                ref0=np.array([[0.6, 0.7], [0.8, 0.9]]),
                                lower=np.array([[-5000.0, -4000.0], [-3000.0, -2000.0]]),
                                upper=np.array([[5000.0, 4000.0], [3000.0, 2000.0]]))
                self.add_output('total_volume', val=1.)
                self.declare_partials(['*'], ['*'], method='cs')

            def compute(self, inputs, outputs):
                super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] + inputs['lengths']

        prob = om.Problem()
        model = prob.model
        
        # bounds arrays don't exist any more unless there's a linesearch that uses them,
        # so use Newton here even though we don't need to.
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)

        model.add_subsystem('p1', om.IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('comp', ExpCompArrayScale())
        model.connect('p1.x', 'comp.lengths')

        prob.setup()
        prob['comp.widths'] = np.ones((2, 2))
        prob.run_model()

        assert_near_equal(prob['comp.total_volume'], 4.)

        slices = model._outputs.get_slice_dict()

        with model._scaled_context_all():
            val = model.comp._outputs['areas']
            assert_near_equal(val[0, 0], (1.0 - 0.1)/(2 - 0.1), tolerance=1e-11)
            assert_near_equal(val[0, 1], (1.0 - 0.2)/(3 - 0.2), tolerance=1e-11)
            assert_near_equal(val[1, 0], (1.0 - 0.3)/(5 - 0.3), tolerance=1e-11)
            assert_near_equal(val[1, 1], (1.0 - 0.4)/(7 - 0.4), tolerance=1e-11)

            val = model.comp._outputs['stuff']
            assert_near_equal(val[0, 0], (2.0 - 0.6)/(11 - 0.6), tolerance=1e-11)
            assert_near_equal(val[0, 1], (2.0 - 0.7)/(13 - 0.7), tolerance=1e-11)
            assert_near_equal(val[1, 0], (2.0 - 0.8)/(17 - 0.8), tolerance=1e-11)
            assert_near_equal(val[1, 1], (2.0 - 0.9)/(19 - 0.9), tolerance=1e-11)

            slc = slices['comp.areas']
            lb = model.nonlinear_solver.linesearch._lower_bounds[slc]

            assert_near_equal(lb[0], (-1000.0 - 0.1)/(2 - 0.1))
            assert_near_equal(lb[1], (-1000.0 - 0.2)/(3 - 0.2))
            assert_near_equal(lb[2], (-1000.0 - 0.3)/(5 - 0.3))
            assert_near_equal(lb[3], (-1000.0 - 0.4)/(7 - 0.4))

            ub = model.nonlinear_solver.linesearch._upper_bounds[slc]
            assert_near_equal(ub[0], (1000.0 - 0.1)/(2 - 0.1))
            assert_near_equal(ub[1], (1000.0 - 0.2)/(3 - 0.2))
            assert_near_equal(ub[2], (1000.0 - 0.3)/(5 - 0.3))
            assert_near_equal(ub[3], (1000.0 - 0.4)/(7 - 0.4))

            slc = slices['comp.stuff']
            lb = model.nonlinear_solver.linesearch._lower_bounds[slc]
            assert_near_equal(lb[0], (-5000.0 - 0.6)/(11 - 0.6))
            assert_near_equal(lb[1], (-4000.0 - 0.7)/(13 - 0.7))
            assert_near_equal(lb[2], (-3000.0 - 0.8)/(17 - 0.8))
            assert_near_equal(lb[3], (-2000.0 - 0.9)/(19 - 0.9))

            ub = model.nonlinear_solver.linesearch._upper_bounds[slc]
            assert_near_equal(ub[0], (5000.0 - 0.6)/(11 - 0.6))
            assert_near_equal(ub[1], (4000.0 - 0.7)/(13 - 0.7))
            assert_near_equal(ub[2], (3000.0 - 0.8)/(17 - 0.8))
            assert_near_equal(ub[3], (2000.0 - 0.9)/(19 - 0.9))

    def test_implicit_scale(self):

        class ImpCompArrayScale(TestImplCompArrayDense):
            def setup(self):
                self.add_input('rhs', val=np.ones(2))
                self.add_output('x', val=np.zeros(2), ref=np.array([2.0, 3.0]),
                                ref0=np.array([4.0, 9.0]),
                                res_ref=np.array([7.0, 11.0]))
                self.add_output('extra', val=np.zeros(2), ref=np.array([12.0, 13.0]),
                                ref0=np.array([14.0, 17.0]))

                self.declare_partials('*', '*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                super(ImpCompArrayScale, self).apply_nonlinear(inputs, outputs, residuals)
                residuals['extra'] = 2.0*self.mtx.dot(outputs['x']) - 3.0*inputs['rhs']

            def linearize(self, inputs, outputs, jacobian):
                # These are incorrect derivatives, but we aren't doing any calculations, and it makes
                # it much easier to check that the scales are correct.
                jacobian['x', 'x'] = np.ones((2, 2))
                jacobian['x', 'extra'] = np.ones((2, 2))
                jacobian['extra', 'x'] = np.ones((2, 2))
                jacobian['x', 'rhs'] = -np.eye(2)


        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImpCompArrayScale())
        model.connect('p1.x', 'comp.rhs')

        prob.setup()
        prob.run_model()

        base_x = model.comp._outputs['x'].copy()
        base_ex = model.comp._outputs['extra'].copy()
        base_res_x = model.comp._residuals['x'].copy()

        with model._scaled_context_all():
            val = model.comp._outputs['x']
            assert_near_equal(val[0], (base_x[0] - 4.0)/(2.0 - 4.0))
            assert_near_equal(val[1], (base_x[1] - 9.0)/(3.0 - 9.0))
            val = model.comp._outputs['extra']
            assert_near_equal(val[0], (base_ex[0] - 14.0)/(12.0 - 14.0))
            assert_near_equal(val[1], (base_ex[1] - 17.0)/(13.0 - 17.0))
            val = model.comp._residuals['x'].copy()
            assert_near_equal(val[0], (base_res_x[0])/(7.0))
            assert_near_equal(val[1], (base_res_x[1])/(11.0))

        model.run_linearize()

        with model._scaled_context_all():
            subjacs = comp._jacobian

            assert_near_equal(subjacs['comp.x', 'comp.x'], np.ones((2, 2)))
            assert_near_equal(subjacs['comp.x', 'comp.extra'], np.ones((2, 2)))
            assert_near_equal(subjacs['comp.x', 'comp.rhs'], -np.eye(2))

    def test_implicit_scale_with_scalar_jac(self):
        raise unittest.SkipTest('Cannot specify an n by m subjac with a scalar yet.')

        class ImpCompArrayScale(TestImplCompArrayDense):
            def setup(self):
                self.add_input('rhs', val=np.ones(2))
                self.add_output('x', val=np.zeros(2), ref=np.array([2.0, 3.0]),
                                ref0=np.array([4.0, 9.0]),
                                res_ref=np.array([7.0, 11.0]))
                self.add_output('extra', val=np.zeros(2), ref=np.array([12.0, 13.0]),
                                ref0=np.array([14.0, 17.0]))

            def apply_nonlinear(self, inputs, outputs, residuals):
                super(ImpCompArrayScale, self).apply_nonlinear(inputs, outputs, residuals)
                residuals['extra'] = 2.0*self.mtx.dot(outputs['x']) - 3.0*inputs['rhs']

            def linearize(self, inputs, outputs, jacobian):
                # These are incorrect derivatives, but we aren't doing any calculations, and it makes
                # it much easier to check that the scales are correct.
                jacobian['x', 'x'][:] = 1.0
                jacobian['x', 'extra'][:] = 1.0
                jacobian['extra', 'x'][:] = 1.0
                jacobian['x', 'rhs'] = -np.eye(2)


        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.ones(2)))
        comp = model.add_subsystem('comp', ImpCompArrayScale())
        model.connect('p1.x', 'comp.rhs')

        prob.setup()
        prob.run_model()

        base_x = model.comp._outputs['x'].copy()
        base_ex = model.comp._outputs['extra'].copy()
        base_res_x = model.comp._residuals['x'].copy()
        with model._scaled_context_all():
            val = model.comp._outputs['x']
            assert_near_equal(val[0], (base_x[0] - 4.0)/(2.0 - 4.0))
            assert_near_equal(val[1], (base_x[1] - 9.0)/(3.0 - 9.0))
            val = model.comp._outputs['extra']
            assert_near_equal(val[0], (base_ex[0] - 14.0)/(12.0 - 14.0))
            assert_near_equal(val[1], (base_ex[1] - 17.0)/(13.0 - 17.0))
            val = model.comp._residuals['x'].copy()
            assert_near_equal(val[0], (base_res_x[0])/(7.0))
            assert_near_equal(val[1], (base_res_x[1])/(11.0))

        model.run_linearize()

        with model._scaled_context_all():
            subjacs = comp._jacobian

            assert_near_equal(subjacs['comp.x', 'comp.x'][0][0], (2.0 - 4.0)/(7.0 - 13.0))
            assert_near_equal(subjacs['comp.x', 'comp.x'][1][0], (2.0 - 4.0)/(11.0 - 18.0))
            assert_near_equal(subjacs['comp.x', 'comp.x'][0][1], (3.0 - 9.0)/(7.0 - 13.0))
            assert_near_equal(subjacs['comp.x', 'comp.x'][1][1], (3.0 - 9.0)/(11.0 - 18.0))

            assert_near_equal(subjacs['comp.x', 'comp.extra'][0][0], (12.0 - 14.0)/(7.0 - 13.0))
            assert_near_equal(subjacs['comp.x', 'comp.extra'][1][0], (12.0 - 14.0)/(11.0 - 18.0))
            assert_near_equal(subjacs['comp.x', 'comp.extra'][0][1], (13.0 - 17.0)/(7.0 - 13.0))
            assert_near_equal(subjacs['comp.x', 'comp.extra'][1][1], (13.0 - 17.0)/(11.0 - 18.0))

            assert_near_equal(subjacs['comp.x', 'comp.rhs'][0][0], -1.0/(7.0 - 13.0))
            assert_near_equal(subjacs['comp.x', 'comp.rhs'][1][0], 0.0)
            assert_near_equal(subjacs['comp.x', 'comp.rhs'][0][1], 0.0)
            assert_near_equal(subjacs['comp.x', 'comp.rhs'][1][1], -1.0/(11.0 - 18.0))

    def test_scale_array_bug1(self):
        # Tests a bug when you have two connections with different sizes (code was using a
        # stale value for the size).

        class ExpCompArrayScale(TestExplCompArrayDense):

            def setup(self):
                self.add_input('lengths', val=np.ones((2, 2)))
                self.add_input('widths', val=np.ones((1, 3)))
                self.add_output('areas', val=np.ones((2, 2)), ref=np.array([[2.0, 3.0], [5.0, 7.0]]),
                                ref0=np.array([[1.1, 1.2], [1.3, 1.4]]))
                self.add_output('stuff', val=np.ones((1, 3)), ref=np.array([[11.0, 13.0, 19.0]]),
                                ref0=np.array([[1.1, 1.2, 1.4]]))
                self.add_output('total_volume', val=1.)

            def compute(self, inputs, outputs):
                """ Don't need to do much."""
                #super(ExpCompArrayScale, self).compute(inputs, outputs)
                outputs['stuff'] = inputs['widths'] * 2
                outputs['areas'] = inputs['lengths'] * 2

                outputs['total_volume'] = np.sum(outputs['areas']) + np.sum(outputs['stuff'])

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', np.ones((2, 2))))
        model.add_subsystem('p2', om.IndepVarComp('x', np.ones((1, 3))))
        model.add_subsystem('comp1', ExpCompArrayScale())
        model.add_subsystem('comp2', ExpCompArrayScale())
        model.connect('p1.x', 'comp1.lengths')
        model.connect('p2.x', 'comp1.widths')
        model.connect('comp1.areas', 'comp2.lengths')
        model.connect('comp1.stuff', 'comp2.widths')

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp1.total_volume'], 14.)
        assert_near_equal(prob['comp2.total_volume'], 28.)

    def test_newton_resid_scaling(self):

        class SimpleComp(om.ImplicitComponent):

            def setup(self):
                self.add_input('x', val=6.0)
                self.add_output('y', val=1.0, ref=100.0, res_ref=10.1)

                self.declare_partials('*', '*')

            def apply_nonlinear(self, inputs, outputs, residuals):
                residuals['y'] = 3.0*outputs['y'] - inputs['x']

            def linearize(self, inputs, outputs, jacobian):

                jacobian['y', 'x'] = -1.0
                jacobian['y', 'y'] = 3.0

        prob = om.Problem()
        model = prob.model = om.Group(assembled_jac_type='dense')

        model.add_subsystem('p1', om.IndepVarComp('x', 6.0))
        model.add_subsystem('comp', SimpleComp())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.DirectSolver()

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.y'], 2.0)

        # Now, let's try with an AssembledJacobian.

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x', 6.0))
        model.add_subsystem('comp', SimpleComp())

        model.connect('p1.x', 'comp.x')

        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.linear_solver = om.DirectSolver(assemble_jac=True)

        prob.setup()
        prob.run_model()

        assert_near_equal(prob['comp.y'], 2.0)

    def test_feature1(self):
        import openmdao.api as om
        from openmdao.core.tests.test_scaling import ScalingExample1

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x1', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x2', 1.0))
        comp = model.add_subsystem('comp', ScalingExample1())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        prob.setup()
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.comp._outputs['y1']
            assert_near_equal(val, 2.0)
            val = model.comp._outputs['y2']
            assert_near_equal(val, 6.0)

    def test_feature2(self):
        import openmdao.api as om
        from openmdao.core.tests.test_scaling import ScalingExample2

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x1', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x2', 1.0))
        comp = model.add_subsystem('comp', ScalingExample2())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        prob.setup()
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.comp._outputs['y1']
            assert_near_equal(val, 0.5)
            val = model.comp._outputs['y2']
            assert_near_equal(val, 0.5)

    def test_feature3(self):
        import openmdao.api as om
        from openmdao.core.tests.test_scaling import ScalingExample3

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p1', om.IndepVarComp('x1', 1.0))
        model.add_subsystem('p2', om.IndepVarComp('x2', 1.0))
        comp = model.add_subsystem('comp', ScalingExample3())
        model.connect('p1.x1', 'comp.x1')
        model.connect('p2.x2', 'comp.x2')

        prob.setup()
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.comp._residuals['y1']
            assert_near_equal(val, -.995)
            val = model.comp._residuals['y2']
            assert_near_equal(val, (1-6000.)/6000.)

    def test_feature_vector(self):
        import openmdao.api as om
        from openmdao.core.tests.test_scaling import ScalingExampleVector

        prob = om.Problem()
        model = prob.model

        model.add_subsystem('p', om.IndepVarComp('x', np.ones((2))))
        comp = model.add_subsystem('comp', ScalingExampleVector())
        model.connect('p.x', 'comp.x')

        prob.setup()
        prob.run_model()

        model.run_apply_nonlinear()

        with model._scaled_context_all():
            val = model.comp._residuals['y']
            assert_near_equal(val[0], (1-200.)/200.)
            assert_near_equal(val[1], (1-6000.)/6000.)
            val = model.comp._outputs['y']
            assert_near_equal(val[0], 2.0)
            assert_near_equal(val[1], 6.0)


class MyComp(om.ExplicitComponent):

    def setup(self):

        self.add_input('x2_u_u')
        self.add_input('x2_u_s')
        self.add_input('x2_s_u')
        self.add_input('x2_s_s')

        self.add_output('x3_u_u', val=1.0)
        self.add_output('x3_u_s', val=1.0)
        self.add_output('x3_s_u', val=1.0, ref=5.0)
        self.add_output('x3_s_s', val=1.0, ref=5.0)

        self.J = np.array([[2.0, 3.0, -5.0, 1.5],
                           [1.0, 6.0, -2.3, 1.0],
                           [7.0, 5.0, 1.1, 2.2],
                           [-3.0, 2.0, 6.8, -1.5]
                           ])
        rows = np.repeat(np.arange(4), 4)
        cols = np.tile(np.arange(4), 4)

        self.declare_partials(of='x3_u_u', wrt='x2_u_u', val=self.J[0, 0])
        self.declare_partials(of='x3_u_u', wrt='x2_u_s', val=self.J[0, 1])
        self.declare_partials(of='x3_u_u', wrt='x2_s_u', val=self.J[0, 2])
        self.declare_partials(of='x3_u_u', wrt='x2_s_s', val=self.J[0, 3])

        self.declare_partials(of='x3_u_s', wrt='x2_u_u', val=self.J[1, 0])
        self.declare_partials(of='x3_u_s', wrt='x2_u_s', val=self.J[1, 1])
        self.declare_partials(of='x3_u_s', wrt='x2_s_u', val=self.J[1, 2])
        self.declare_partials(of='x3_u_s', wrt='x2_s_s', val=self.J[1, 3])

        self.declare_partials(of='x3_s_u', wrt='x2_u_u', val=self.J[2, 0])
        self.declare_partials(of='x3_s_u', wrt='x2_u_s', val=self.J[2, 1])
        self.declare_partials(of='x3_s_u', wrt='x2_s_u', val=self.J[2, 2])
        self.declare_partials(of='x3_s_u', wrt='x2_s_s', val=self.J[2, 3])

        self.declare_partials(of='x3_s_s', wrt='x2_u_u', val=self.J[3, 0])
        self.declare_partials(of='x3_s_s', wrt='x2_u_s', val=self.J[3, 1])
        self.declare_partials(of='x3_s_s', wrt='x2_s_u', val=self.J[3, 2])
        self.declare_partials(of='x3_s_s', wrt='x2_s_s', val=self.J[3, 3])

    def compute(self, inputs, outputs, discrete_inputs=None,
                discrete_outputs=None):

        outputs['x3_u_u'] = self.J[0, 0] * inputs['x2_u_u'] + self.J[0, 1] * inputs['x2_u_s'] + self.J[0, 2] * inputs['x2_s_u'] + self.J[0, 3] * inputs['x2_s_s']
        outputs['x3_u_s'] = self.J[1, 0] * inputs['x2_u_u'] + self.J[1, 1] * inputs['x2_u_s'] + self.J[1, 2] * inputs['x2_s_u'] + self.J[1, 3] * inputs['x2_s_s']
        outputs['x3_s_u'] = self.J[2, 0] * inputs['x2_u_u'] + self.J[2, 1] * inputs['x2_u_s'] + self.J[2, 2] * inputs['x2_s_u'] + self.J[2, 3] * inputs['x2_s_s']
        outputs['x3_s_s'] = self.J[3, 0] * inputs['x2_u_u'] + self.J[3, 1] * inputs['x2_u_s'] + self.J[3, 2] * inputs['x2_s_u'] + self.J[3, 3] * inputs['x2_s_s']


class MyImplicitComp(om.ImplicitComponent):

    def setup(self):

        self.add_input('x2_u')

        self.add_output('x3_u', val=1.0)
        self.add_output('x3_s', val=1.0, ref=5.0)

        self.declare_partials('*', '*')

        self.J = np.array([[.3, -.7, .5], [1.1, 1.3, -1.7]])

    def apply_nonlinear(self, inputs, outputs, residuals):
        residuals['x3_u'] = self.J[0, 0]*inputs['x2_u']**2 + self.J[0, 1]*outputs['x3_u']**2 + self.J[0, 2]*outputs['x3_s']**2
        residuals['x3_s'] = self.J[1, 0]*inputs['x2_u']**2 + self.J[1, 1]*outputs['x3_u'] **2+ self.J[1, 2]*outputs['x3_s']**2

    def linearize(self, inputs, outputs, jacobian):
        jacobian['x3_u', 'x2_u'] = 2.0 * self.J[0, 0] * inputs['x2_u']
        jacobian['x3_u', 'x3_u'] = 2.0 * self.J[0, 1] * outputs['x3_u']
        jacobian['x3_u', 'x3_s'] = 2.0 * self.J[0, 2] * outputs['x3_s']
        jacobian['x3_s', 'x2_u'] = 2.0 * self.J[1, 0] * inputs['x2_u']
        jacobian['x3_s', 'x3_u'] = 2.0 * self.J[1, 1] * outputs['x3_u']
        jacobian['x3_s', 'x3_s'] = 2.0 * self.J[1, 2] * outputs['x3_s']


class MyDriver(Driver):

    def run(self):

        self.param_meta = deepcopy(self._designvars)
        self.param_vals = self.get_design_var_values()
        self.con_meta = deepcopy(self._cons)

        # Run model
        model = self._problem().model
        model.run_solve_nonlinear()

        # Con vals and derivs
        self.con_vals = deepcopy(self.get_constraint_values())
        self.sens_dict = self._compute_totals(of=list(self.con_meta.keys()),
                                              wrt=list(self.param_meta.keys()),
                                              return_format='dict')

        # Obj vals
        self.obj_vals = deepcopy(self.get_objective_values())


class TestScalingOverhaul(unittest.TestCase):

    def test_in_driver(self):
        # This test assures that the driver is correctly seeing unscaled (physical) data.

        prob = om.Problem()
        model = prob.model

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('x1_u_u', val=1.0)
        inputs_comp.add_output('x1_u_s', val=1.0)
        inputs_comp.add_output('x1_s_u', val=1.0, ref=3.0)
        inputs_comp.add_output('x1_s_s', val=1.0, ref=3.0)
        inputs_comp.add_output('ox1_u_u', val=1.0)
        inputs_comp.add_output('ox1_u_s', val=1.0)
        inputs_comp.add_output('ox1_s_u', val=1.0, ref=3.0)
        inputs_comp.add_output('ox1_s_s', val=1.0, ref=3.0)

        model.add_subsystem('p', inputs_comp)
        mycomp = model.add_subsystem('comp', MyComp())

        model.connect('p.x1_u_u', 'comp.x2_u_u')
        model.connect('p.x1_u_s', 'comp.x2_u_s')
        model.connect('p.x1_s_u', 'comp.x2_s_u')
        model.connect('p.x1_s_s', 'comp.x2_s_s')

        driver = prob.driver = MyDriver()

        model.add_design_var('p.x1_u_u', lower=-11, upper=11)
        model.add_design_var('p.x1_u_s', ref=7.0, lower=-11, upper=11)
        model.add_design_var('p.x1_s_u', lower=-11, upper=11)
        model.add_design_var('p.x1_s_s', ref=7.0, lower=-11, upper=11)

        # easy constraints for basic check
        model.add_constraint('p.x1_u_u', upper=3.3)
        model.add_constraint('p.x1_u_s', upper=3.3, ref=13.0)
        model.add_constraint('p.x1_s_u', upper=3.3)
        model.add_constraint('p.x1_s_s', upper=3.3, ref=13.0)

        # harder to calculate constraints
        model.add_constraint('comp.x3_u_u', upper=3.3)
        model.add_constraint('comp.x3_u_s', upper=3.3, ref=17.0)
        model.add_constraint('comp.x3_s_u', upper=3.3)
        model.add_constraint('comp.x3_s_s', upper=3.3, ref=17.0)

        model.add_objective('p.ox1_u_u')
        model.add_objective('p.ox1_u_s', ref=15.0)
        model.add_objective('p.ox1_s_u')
        model.add_objective('p.ox1_s_s', ref=15.0)

        prob.setup()

        prob.run_driver()

        # Parameter values
        assert_near_equal(driver.param_vals['p.x1_u_u'], 1.0)
        assert_near_equal(driver.param_vals['p.x1_u_s'], 1.0/7.0)
        assert_near_equal(driver.param_vals['p.x1_s_u'], 1.0)
        assert_near_equal(driver.param_vals['p.x1_s_s'], 1.0/7.0)

        assert_near_equal(driver.param_meta['p.x1_u_u']['upper'], 11.0)
        assert_near_equal(driver.param_meta['p.x1_u_s']['upper'], 11.0/7.0)
        assert_near_equal(driver.param_meta['p.x1_s_u']['upper'], 11.0)
        assert_near_equal(driver.param_meta['p.x1_s_s']['upper'], 11.0/7.0)

        assert_near_equal(driver.con_meta['p.x1_u_u']['upper'], 3.3)
        assert_near_equal(driver.con_meta['p.x1_u_s']['upper'], 3.3/13.0)
        assert_near_equal(driver.con_meta['p.x1_s_u']['upper'], 3.3)
        assert_near_equal(driver.con_meta['p.x1_s_s']['upper'], 3.3/13.0)

        assert_near_equal(driver.con_vals['p.x1_u_u'], 1.0)
        assert_near_equal(driver.con_vals['p.x1_u_s'], 1.0/13.0)
        assert_near_equal(driver.con_vals['p.x1_s_u'], 1.0)
        assert_near_equal(driver.con_vals['p.x1_s_s'], 1.0/13.0)

        assert_near_equal(driver.obj_vals['p.ox1_u_u'], 1.0)
        assert_near_equal(driver.obj_vals['p.ox1_u_s'], 1.0/15.0)
        assert_near_equal(driver.obj_vals['p.ox1_s_u'], 1.0)
        assert_near_equal(driver.obj_vals['p.ox1_s_s'], 1.0/15.0)

        J = model.comp.J

        assert_near_equal(driver.sens_dict['comp.x3_u_u']['p.x1_u_u'][0][0], J[0, 0])
        assert_near_equal(driver.sens_dict['comp.x3_u_s']['p.x1_u_u'][0][0], J[1, 0] / 17.0)
        assert_near_equal(driver.sens_dict['comp.x3_s_u']['p.x1_u_u'][0][0], J[2, 0])
        assert_near_equal(driver.sens_dict['comp.x3_s_s']['p.x1_u_u'][0][0], J[3, 0] / 17.0)

        assert_near_equal(driver.sens_dict['comp.x3_u_u']['p.x1_u_s'][0][0], J[0, 1] * 7.0)
        assert_near_equal(driver.sens_dict['comp.x3_u_s']['p.x1_u_s'][0][0], J[1, 1] / 17.0 * 7.0)
        assert_near_equal(driver.sens_dict['comp.x3_s_u']['p.x1_u_s'][0][0], J[2, 1] * 7.0)
        assert_near_equal(driver.sens_dict['comp.x3_s_s']['p.x1_u_s'][0][0], J[3, 1] / 17.0 * 7.0)

        assert_near_equal(driver.sens_dict['comp.x3_u_u']['p.x1_s_u'][0][0], J[0, 2])
        assert_near_equal(driver.sens_dict['comp.x3_u_s']['p.x1_s_u'][0][0], J[1, 2] / 17.0)
        assert_near_equal(driver.sens_dict['comp.x3_s_u']['p.x1_s_u'][0][0], J[2, 2])
        assert_near_equal(driver.sens_dict['comp.x3_s_s']['p.x1_s_u'][0][0], J[3, 2] / 17.0)

        assert_near_equal(driver.sens_dict['comp.x3_u_u']['p.x1_s_s'][0][0], J[0, 3] * 7.0)
        assert_near_equal(driver.sens_dict['comp.x3_u_s']['p.x1_s_s'][0][0], J[1, 3] / 17.0* 7.0)
        assert_near_equal(driver.sens_dict['comp.x3_s_u']['p.x1_s_s'][0][0], J[2, 3] * 7.0)
        assert_near_equal(driver.sens_dict['comp.x3_s_s']['p.x1_s_s'][0][0], J[3, 3] / 17.0 * 7.0)

        totals = prob.check_totals(compact_print=True, out_stream=None)

        for (of, wrt) in totals:
            assert_near_equal(totals[of, wrt]['abs error'][0], 0.0, 1e-7)

    def test_iimplicit(self):
        # Testing that our scale/unscale contexts leave the output vector in the correct state when
        # linearize is called on implicit components.
        prob = om.Problem()
        model = prob.model

        inputs_comp = om.IndepVarComp()
        inputs_comp.add_output('x1_u', val=1.0)

        model.add_subsystem('p', inputs_comp)
        mycomp = model.add_subsystem('comp', MyImplicitComp())

        model.connect('p.x1_u', 'comp.x2_u')

        model.linear_solver = om.DirectSolver()
        model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
        model.nonlinear_solver.options['atol'] = 1e-12
        model.nonlinear_solver.options['rtol'] = 1e-12

        model.add_design_var('p.x1_u', lower=-11, upper=11)
        model.add_constraint('p.x1_u', upper=3.3)
        model.add_objective('comp.x3_u')
        model.add_objective('comp.x3_s')

        prob.setup()
        prob.run_model()

        totals = prob.check_totals(compact_print=True, out_stream=None)

        for (of, wrt) in totals:
            assert_near_equal(totals[of, wrt]['abs error'][0], 0.0, 1e-7)


if __name__ == '__main__':
    unittest.main()
