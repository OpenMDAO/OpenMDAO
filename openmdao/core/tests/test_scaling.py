"""Define the units/scaling tests."""
from __future__ import division, print_function

import numpy as np
import unittest

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


class PassThroughLengths(ExplicitComponent):
    """
    Units/scaling test component taking an array of lengths in cm and
    passing them through in km.
    Currently raises an error for passing array value as 'ref'.
    """

    def initialize_variables(self):
        self.add_input('old_lengths', val=np.ones(4), units='cm')
        self.add_output('new_lengths', val=np.ones(4), units='km', ref=0.1*np.ones(4))

    def compute(self, inputs, outputs):
        lengths_cm = inputs['old_lengths']
        lengths_m = lengths_cm * 1e-2
        lengths_km = lengths_m * 1e-3
        outputs['new_lengths'] = lengths_km


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

        # Scale the output based on the column coeff.
        if self.metadata['row'] == 1:
            ref = 1. / c1
        elif self.metadata['row'] == 2:
            ref = 1. / c2

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
        self.add_output('y', ref=ref, res_ref=res_ref)

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

    def test_pass_through_array(self):
        group = Group()
        group.add_subsystem('sys1', IndepVarComp('old_lengths', np.ones(4),
                                                 units='mm',
                                                 ref=1e5*np.ones(4)))
        group.add_subsystem('sys2', PassThroughLengths())
        group.connect('sys1.old_lengths', 'sys2.old_lengths')

        prob = Problem(group)

        with self.assertRaises(Exception) as cm:
            prob.setup(check=False)
        self.assertEqual(str(cm.exception),
                         "The ref argument should be a float")

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
           
if __name__ == '__main__':
    unittest.main()
