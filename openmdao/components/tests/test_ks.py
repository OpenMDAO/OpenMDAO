""" Test the ExternalCode. """
from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp, Group, ExecComp, ScipyOptimizeDriver, \
     ExplicitComponent
from openmdao.components.ks import KSComponent
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.test_examples.beam_optimization.multipoint_beam_stress import MultipointBeamGroup
from openmdao.utils.assert_utils import assert_rel_error


class TestKSFunction(unittest.TestCase):

    def test_basic_ks(self):
        prob = Problem()
        prob.model = model = Group()

        model.add_subsystem('px', IndepVarComp(name="x", val=np.ones((2, ))))
        model.add_subsystem('comp', DoubleArrayComp())
        model.add_subsystem('ks', KSComponent(width=2))
        model.connect('px.x', 'comp.x1')
        model.connect('comp.y2', 'ks.g')

        model.add_design_var('px.x')
        model.add_objective('comp.y1')
        model.add_constraint('ks.KS', upper=0.0)

        prob.setup(check=False)
        prob.run_driver()

        assert_rel_error(self, max(prob['comp.y2']), prob['ks.KS'])

    def test_beam_stress(self):
        E = 1.
        L = 1.
        b = 0.1
        volume = 0.01
        max_bending = 100.0

        num_cp = 5
        num_elements = 25
        num_load_cases = 2

        prob = Problem(model=MultipointBeamGroup(E=E, L=L, b=b, volume=volume, max_bending = max_bending,
                                                 num_elements=num_elements, num_cp=num_cp,
                                                 num_load_cases=num_load_cases))

        prob.driver = ScipyOptimizeDriver()
        prob.driver.options['optimizer'] = 'SLSQP'
        prob.driver.options['tol'] = 1e-9
        prob.driver.options['disp'] = False

        prob.setup(mode='rev')

        prob.run_driver()

        stress0 = prob['parallel.sub_0.max_stress_0.g_con']
        stress1 = prob['parallel.sub_0.max_stress_1.g_con']

        # Test that the the maximum constraint prior to aggregation is close to "active".
        assert_rel_error(self, max(stress0), 0.0, tolerance=5e-2)
        assert_rel_error(self, max(stress1), 0.0, tolerance=5e-2)

        # Test that no original constraint is violated.
        self.assertTrue(np.all(stress0 < 100.0))
        self.assertTrue(np.all(stress1 < 100.0))

if __name__ == "__main__":
    unittest.main()
