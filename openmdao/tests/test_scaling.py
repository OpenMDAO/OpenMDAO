"""Define the units/scaling tests."""
from __future__ import division, print_function

import numpy
import scipy.sparse
import unittest

from six import iteritems
from six.moves import range

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp

from openmdao.devtools.testutil import assert_rel_error


class PassThroughLength(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('old_length', val=1., units='cm')
        self.add_output('new_length', val=1., units='km', ref=0.1)

    def compute(self, inputs, outputs):
        length_cm = inputs['old_length']
        length_m = length_cm * 1e-2
        length_km = length_m * 1e-3
        outputs['new_length'] = length_km


class SpeedComputationWithUnits(ExplicitComponent):

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


class TestScaling(unittest.TestCase):

    def test_scaling(self):
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
        assert_rel_error(self, prob.model._outputs['sys1.old_length'], 3.)
        prob.run_model()
        assert_rel_error(self, prob['sys2.new_length'], 3.e-1)
        assert_rel_error(self, prob.model._outputs['sys2.new_length'], 3.)

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
        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['c1.distance'], 1.0)  # units: km
        assert_rel_error(self, prob['c2.distance'], 1000.0)  # units: m
        assert_rel_error(self, prob['c1.time'], 1.0)  # units: h
        assert_rel_error(self, prob['c2.time'], 3600.0)  # units: s
        assert_rel_error(self, prob['c2.speed'], 1.0)  # units: km/h (i.e., kph)


if __name__ == '__main__':
    unittest.main()
