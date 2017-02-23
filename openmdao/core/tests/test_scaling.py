"""Define the units/scaling tests."""
from __future__ import division, print_function

import numpy
import scipy.sparse
import unittest

from six import iteritems
from six.moves import range

from openmdao.api import Problem, Group, ExplicitComponent, ImplicitComponent, IndepVarComp
from openmdao.api import NewtonSolver, ScipyIterativeSolver

from openmdao.devtools.testutil import assert_rel_error


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


if __name__ == '__main__':
    unittest.main()
