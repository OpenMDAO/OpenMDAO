from __future__ import division
import unittest

from openmdao.api import Problem, Group, IndepVarComp, ExecComp
from openmdao.utils.assert_utils import assert_rel_error

try:
    from openmdao.parallel_api import PETScVector
except ImportError:
    PETScVector = None


class ReconfGroup(Group):

    def __init__(self, size=0):
        super(ReconfGroup, self).__init__()

        self.size = size

    def setup(self):
        self.size += 1

        for ind in range(self.size):
            self.add_subsystem(
                'C%i' % ind, ExecComp('y%i = %i * x + 1.' % (ind, ind)), promotes=['*'])


class Test(unittest.TestCase):

    def test(self):
        prob = Problem(model=Group())
        prob.model.add_subsystem('Cx', IndepVarComp('x', 1.0), promotes=['x'])
        prob.model.add_subsystem('g', ReconfGroup(), promotes=['*'])
        prob.setup(check=False)

        # First run with the initial setup.
        prob['x'] = 2.0
        prob.run_model()
        assert_rel_error(self, prob['y0'], 1.0)
        print(prob['y0'])

        # Now reconfigure ReconfGroup and re-run, ensuring the value of x is preserved.
        prob.model.g.resetup('reconf')
        prob.model.resetup('update')
        prob.run_model()
        assert_rel_error(self, prob['y0'], 1.0)
        assert_rel_error(self, prob['y1'], 3.0)
        print(prob['y0'], prob['y1'])

        # Running reconf setup from root is equivalent to running full setup, but is faster
        prob.model.resetup('reconf')
        prob['x'] = 2.0
        prob.run_model()
        assert_rel_error(self, prob['y0'], 1.0)
        assert_rel_error(self, prob['y1'], 3.0)
        assert_rel_error(self, prob['y2'], 5.0)
        print(prob['y0'], prob['y1'], prob['y2'])


if __name__ == '__main__':
    unittest.main()
