""" Unit tests for the problem interface."""

import unittest

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.test_suite.components.paraboloid import Paraboloid


class TestProblem(unittest.TestCase):

    def test_compute_total_derivs_basic(self):
        # Basic test for the method using default solvers on simple model.


        top = Problem()
        root = top.root = Group()
        root.add('xx', IndepVarComp('p1', name='x'), promotes=['x'])
        root.add('xx', IndepVarComp('p2', name='y'), promotes=['y'])
        root.add('comp', promotes=['x', 'y', 'f_xy'])

        top.setup(mode='fwd')


if __name__ == "__main__":
    unittest.main()
