""" Unit tests for the design_variable and response interface to system."""
from __future__ import print_function
import unittest

import numpy as np

from openmdao.api import Problem, NonlinearBlockGS
from openmdao.devtools.testutil import assert_rel_error

from openmdao.test_suite.components.sellar import SellarDerivatives, SellarDerivativesConnected

class TestDesVarsResponses(unittest.TestCase):

    def test_api_backwards_compatible(self):
        raise unittest.SkipTest("api not implemented yet")

        prob = Problem()
        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.driver = ScipyOpt()
        prob.driver.options['method'] = 'slsqp'
        prob.driver.add_design_var('x', lower=-100, upper=100)
        prob.driver.add_design_var('z', lower=-100, upper=100)
        prob.driver.add_objective('obj')
        prob.driver.add_constraint('con1')
        prob.driver.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_des_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertItemsEqual(des_vars.keys(), ('x', 'z'))
        self.assertItemsEqual(obj.keys(), ('obj',))
        self.assertItemsEqual(constraints.keys(), ('con1', 'con2'))

    def test_api_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj'})
        self.assertEqual(set(constraints.keys()), {'con1', 'con2'})

    def test_api_response_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=-100, upper=100)
        prob.model.add_response('obj', type="obj")
        prob.model.add_response('con1', type="con")
        prob.model.add_response('con2', type="con")

        prob.setup()

        des_vars = prob.model.get_design_vars()
        responses = prob.model.get_responses()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(responses.keys()), {'obj', 'con1', 'con2'})
        self.assertEqual(set(obj.keys()), {'obj'})
        self.assertEqual(set(constraints.keys()), {'con1', 'con2'})

    def test_api_list_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z', lower=[-100, -20], upper=[100, 20])
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj',})
        self.assertEqual(set(constraints.keys()), {'con1', 'con2'})

    def test_api_array_on_model(self):

        prob = Problem()

        prob.model = SellarDerivatives()
        prob.model.nl_solver = NonlinearBlockGS()

        prob.model.add_design_var('x', lower=-100, upper=100)
        prob.model.add_design_var('z',
                                  lower=np.array([-100, -20]),
                                  upper=np.array([100, 20]))
        prob.model.add_objective('obj')
        prob.model.add_constraint('con1')
        prob.model.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_des_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'x', 'z'})
        self.assertEqual(set(obj.keys()), {'obj',})
        self.assertEqual(set(constraints.keys()), {'con1', 'con2'})

    def test_api_on_subsystems(self):

        prob = Problem()

        prob.model = SellarDerivativesConnected()
        prob.model.nl_solver = NonlinearBlockGS()

        px = prob.model.get_subsystem('px')
        px.add_design_var('x', lower=-100, upper=100)

        pz = prob.model.get_subsystem('pz')
        pz.add_design_var('z', lower=-100, upper=100)

        obj = prob.model.get_subsystem('obj_cmp')
        obj.add_objective('obj')

        con_comp1 = prob.model.get_subsystem('con_cmp1')
        con_comp1.add_constraint('con1')

        con_comp2 = prob.model.get_subsystem('con_cmp2')
        con_comp2.add_constraint('con2')

        prob.setup()

        des_vars = prob.model.get_design_vars()
        obj = prob.model.get_objectives()
        constraints = prob.model.get_constraints()

        self.assertEqual(set(des_vars.keys()), {'px.x', 'pz.z'})
        self.assertEqual(set(obj.keys()), {'obj_cmp.obj',})
        self.assertEqual(set(constraints.keys()), {'con_cmp1.con1', 'con_cmp2.con2'})