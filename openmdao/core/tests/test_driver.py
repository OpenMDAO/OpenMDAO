""" Unit tests for the Driver base class."""

from __future__ import print_function

import unittest

from openmdao.api import Problem
from openmdao.test_suite.components.sellar import SellarDerivatives

class TestDriver(unittest.TestCase):

    def test_basic_get(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        model.suppress_solver_output = True

        prob.setup(check=False)
        prob.run_driver()

        designvars = prob.driver.get_design_var_values()
        self.assertEqual(designvars['x'], 2.0 )

if __name__ == "__main__":
    unittest.main()
