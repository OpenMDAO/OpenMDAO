"""Define the units/scaling tests."""
from __future__ import division, print_function

import numpy
import scipy.sparse
import unittest

from six import iteritems
from six.moves import range

from openmdao.api import Problem, Group, ExplicitComponent, IndepVarComp


class PassThroughLength(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('old_length', val=1., units='cm')
        self.add_output('new_length', val=1., units='km', ref=0.1)

    def compute(self, inputs, outputs):
        length_cm = inputs['old_length']
        length_m = length_cm * 1e-2
        length_km = length_m * 1e-3
        outputs['new_length'] = length_km


class TestScaling(unittest.TestCase):

    def test_scaling(self):
        group = Group()
        group.add_subsystem('sys1', IndepVarComp('old_length', 1.0,
                                                 units='mm', ref=1e5))
        group.add_subsystem('sys2', PassThroughLength())
        group.connect('sys1.old_length', 'sys2.old_length')

        prob = Problem(group)

        prob.setup(check=False)
        prob.root.suppress_solver_output = True

        prob['sys1.old_length'] = 3.e5
        self.assertAlmostEqual(prob['sys1.old_length'], 3.e5)
        self.assertAlmostEqual(prob.root._outputs['sys1.old_length'], 3.)
        prob.run()
        self.assertAlmostEqual(prob['sys2.new_length'], 3.e-1)
        self.assertAlmostEqual(prob.root._outputs['sys2.new_length'], 3.)


if __name__ == '__main__':
    unittest.main()
