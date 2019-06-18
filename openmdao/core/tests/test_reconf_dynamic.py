from __future__ import division
import numpy as np
import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, ExecComp
from openmdao.utils.assert_utils import assert_rel_error


class ReconfComp(ExplicitComponent):

    def initialize(self):
        self.size = 1
        self.counter = 0

    def reconfigure(self):
        self.counter += 1

        if self.counter % 2 == 0:
            self.size += 1
            return True
        else:
            return False

    def setup(self):
        self.add_input('x', val=1.0)
        self.add_output('y', val=np.zeros(self.size))
        # All derivatives are defined.
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        outputs['y'] = 2 * inputs['x']

    def compute_partials(self, inputs, jacobian):
        jacobian['y', 'x'] = 2 * np.ones((self.size, 1))


class ReconfGroup(Group):

    def reconfigure(self):
        return True


class Comp(ExplicitComponent):

    def setup(self):
        self.add_input('x', val=1.0)
        self.add_output('z', val=1.0)

    def compute(self, inputs, outputs):
        outputs['z'] = 3 * inputs['x']

    def compute_partials(self, inputs, jacobian):
        jacobian['z', 'x'] = 3.0


class Test(unittest.TestCase):

    def ttest_reconf_comp(self):
        p = Problem()

        p.model.add_subsystem('c1', IndepVarComp('x', 1.0), promotes_outputs=['x'])
        p.model.add_subsystem('c2', ReconfComp(), promotes_inputs=['x'], promotes_outputs=['y'])
        p.model.add_subsystem('c3', Comp(), promotes_inputs=['x'], promotes_outputs=['z'])

        p.setup()
        p['x'] = 3.

        # First run the model once; counter = 1, size of y = 1
        p.run_model()
        totals = p.compute_totals(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0)
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], 2.0)
        print(p['x'], p['y'], p['z'], totals['y', 'x'].flatten())

        # Run the model again, which will trigger reconfiguration; counter = 2, size of y = 2
        p.run_model()
        totals = p.compute_totals(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0 * np.ones(2))
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], 2.0 * np.ones(2, 1))

    def test_reconf_group(self):
        p = Problem()

        p.model.add_subsystem('s1', IndepVarComp('x', 1.0), promotes_outputs=['x'])
        s2 = p.model.add_subsystem('s2', ReconfGroup(), promotes=['*'])
        p.model.add_subsystem('s3', ExecComp('z=3*x'), promotes=['*'])
        s2.add_subsystem('comp', ReconfComp(), promotes=['*'])

        p.setup()
        p['x'] = 3.

        p.run_model()
        totals = p.compute_totals(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0)
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], [[2.0]])

        # This tests for a bug in which inputs from sources outside the reconfiguring system
        # are zero-ed out during the execution following the reconfiguration (prior to updates)
        # because the scaling vectors for those external-source inputs are all zero.
        # The solution is to initialize the multiplier in the scaling vector to 1.
        assert_rel_error(self, s2._inputs['x'], 3.0)


if __name__ == '__main__':
    unittest.main()
