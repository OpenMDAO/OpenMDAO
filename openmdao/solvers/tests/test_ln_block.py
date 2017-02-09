"""Test the BlockLinearSolver classes."""

from __future__ import division, print_function
from six import iteritems

import unittest

import numpy as np

from openmdao.api import Group, IndepVarComp, Problem, LinearBlockGS, LinearBlockJac, GlobalJacobian
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense

class TestLinearBlockGSSolver(unittest.TestCase):

    def test_globaljac_err(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = LinearBlockGS()
        model.suppress_solver_output = True

        prob.model.jacobian = GlobalJacobian()
        prob.setup(check=False, mode='fwd')
        
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        with self.assertRaises(RuntimeError) as context:
            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')

        self.assertEqual(str(context.exception),
                         "A block linear solver 'LN: LNBGS' is being used with a GlobalJacobian in system ''")
        


class TestLinearBlockJacSolver(unittest.TestCase):

    def test_globaljac_err(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = LinearBlockJac()
        model.suppress_solver_output = True

        prob.model.jacobian = GlobalJacobian()
        prob.setup(check=False, mode='fwd')
        
        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        with self.assertRaises(RuntimeError) as context:
            J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')

            self.assertEqual(str(context.exception),
                             "A block linear solver 'LN: LNBJ' is being used with a GlobalJacobian in system ''")
        


if __name__ == "__main__":
    unittest.main()
