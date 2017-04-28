""" Unit tests for the Driver base class."""

from __future__ import print_function

import unittest

import numpy as np

from openmdao.api import Problem
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.sellar import SellarDerivatives


class TestDriver(unittest.TestCase):

    def test_basic_get(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_driver()

        designvars = prob.driver.get_design_var_values()
        self.assertEqual(designvars['pz.z'][0], 5.0 )

        designvars = prob.driver.get_objective_values()
        self.assertEqual(designvars['obj_cmp.obj'], prob['obj'] )

        designvars = prob.driver.get_constraint_values()
        self.assertEqual(designvars['con_cmp1.con1'], prob['con1']  )

    def test_scaled_design_vars(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', ref=5.0, ref0=3.0)
        model.add_objective('obj')
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)

        dv = prob.driver.get_design_var_values()
        self.assertEqual(dv['pz.z'][0], 1.0)
        self.assertEqual(dv['pz.z'][1], -0.5)

        prob.driver.set_design_var('pz.z', np.array((2.0, -2.0)))
        self.assertEqual(prob['z'][0], 7.0)
        self.assertEqual(prob['z'][1], -1.0)

    def test_scaled_constraints(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj')
        model.add_constraint('con1', lower=0, ref=2.0, ref0=3.0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        cv = prob.driver.get_constraint_values()['con_cmp1.con1'][0]
        base = prob['con1']
        self.assertEqual((base-3.0)/(2.0-3.0), cv)

    def test_scaled_objectves(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z')
        model.add_objective('obj', ref=2.0, ref0=3.0)
        model.add_constraint('con1', lower=0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        cv = prob.driver.get_objective_values()['obj_cmp.obj'][0]
        base = prob['obj']
        self.assertEqual((base-3.0)/(2.0-3.0), cv)

    def test_scaled_derivs(self):

        prob = Problem()
        prob.model = model = SellarDerivatives()

        model.add_design_var('z', ref=2.0, ref0=0.0)
        model.add_objective('obj', ref=1.0, ref0=0.0)
        model.add_constraint('con1', lower=0, ref=2.0, ref0=0.0)
        prob.set_solver_print(level=0)

        prob.setup(check=False)
        prob.run_model()

        base = prob.compute_total_derivs(of=['obj', 'con1'], wrt=['z'])
        derivs = prob.driver._compute_total_derivs(of=['obj_cmp.obj', 'con_cmp1.con1'], wrt=['pz.z'],
                                                   return_format='dict')
        assert_rel_error(self, base[('con1', 'z')][0], derivs['con_cmp1.con1']['pz.z'][0], 1e-5)
        assert_rel_error(self, base[('obj', 'z')][0]*2.0, derivs['obj_cmp.obj']['pz.z'][0], 1e-5)

if __name__ == "__main__":
    unittest.main()
