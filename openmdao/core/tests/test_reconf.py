from __future__ import division
import numpy as np
import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExplicitComponent, DefaultVector
from openmdao.devtools.testutil import assert_rel_error

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class ReconfComp(ExplicitComponent):

    def __init__(self, size=0):
        super(ReconfComp, self).__init__()

        self.size = size

    def initialize_variables(self):
        self.size += 1
        self.add_input('x', val=1.0)
        self.add_output('y', val=np.zeros(self.size))

    def compute(self, inputs, outputs):
        outputs['y'] = 2 * inputs['x']

    def compute_partial_derivs(self, inputs, outputs, jacobian):
        jacobian['y', 'x'] = 2 * np.ones((self.size, 1))


class Comp(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('x', val=1.0)
        self.add_output('z', val=1.0)

    def compute(self, inputs, outputs):
        outputs['z'] = 3 * inputs['x']

    def compute_partials_derivs(self, inputs, outputs, jacobian):
        jacobian['z', 'x'] = 3.0


class Test(unittest.TestCase):

    def test(self):
        p = Problem()

        p.model = Group()
        p.model.add_subsystem('c1', IndepVarComp('x', 1.0), promotes_outputs=['x'])
        p.model.add_subsystem('c2', ReconfComp(), promotes_inputs=['x'], promotes_outputs=['y'])
        p.model.add_subsystem('c3', Comp(), promotes_inputs=['x'], promotes_outputs=['z'])

        # First run the usual setup method on Problem; size of y = 1
        p.setup()
        p['x'] = 2
        p.run_model()
        totals = p.compute_total_derivs(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 2.0)
        assert_rel_error(self, p['y'], 4.0)
        assert_rel_error(self, p['z'], 6.0)
        assert_rel_error(self, totals['y', 'x'], 2.0)
        print(p['x'], p['y'], p['z'], totals['y', 'x'].flatten())

        # Now run the setup method on the root system; size of y = 2
        p.model.setup()
        p['x'] = 3
        p.run_model()
        totals = p.compute_total_derivs(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0 * np.ones(2))
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], 2.0 * np.ones(2))
        print(p['x'], p['y'], p['z'], totals['y', 'x'].flatten())

        # Now reconfigure from c2 and update in root; size of y = 3; the value of x is preserved
        p.model.get_subsystem('c2').setup('reconf')
        p.model.setup('update')
        p.run_model()
        totals = p.compute_total_derivs(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0 * np.ones(3))
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], 2.0 * np.ones(3))
        print(p['x'], p['y'], p['z'], totals['y', 'x'].flatten())

        # Now reconfigure from c3 and update in root; size of y = 3; the value of x is preserved
        p.model.get_subsystem('c3').setup('reconf')
        p.model.setup('update')
        p.run_model()
        totals = p.compute_total_derivs(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0 * np.ones(3))
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], 2.0 * np.ones(3))
        print(p['x'], p['y'], p['z'], totals['y', 'x'].flatten())

        # Finally, setup reconf from root; size of y = 4
        # Since we are at the root, calling setup('full') and setup('reconf') have the same effect.
        # In both cases, variable values are lost so we have to set x=3 again.
        p.model.setup('reconf')
        p['x'] = 3
        p.run_model()
        totals = p.compute_total_derivs(wrt=['x'], of=['y'])
        assert_rel_error(self, p['x'], 3.0)
        assert_rel_error(self, p['y'], 6.0 * np.ones(4))
        assert_rel_error(self, p['z'], 9.0)
        assert_rel_error(self, totals['y', 'x'], 2.0 * np.ones(4))
        print(p['x'], p['y'], p['z'], totals['y', 'x'].flatten())


if __name__ == '__main__':
    unittest.main()
