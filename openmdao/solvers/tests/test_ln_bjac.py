"""Test the LinearBlockJac class."""

from __future__ import division, print_function
from six import iteritems

import unittest

import numpy as np

from openmdao.api import Group, IndepVarComp, Problem, LinearBlockJac, AssembledJacobian
from openmdao.devtools.testutil import assert_rel_error
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleJacVec
from openmdao.test_suite.components.sellar import SellarDerivativesGrouped, \
     SellarStateConnection, SellarDerivatives
from openmdao.test_suite.components.expl_comp_simple import TestExplCompSimpleDense
from openmdao.test_suite.components.simple_comps import DoubleArrayComp
from openmdao.test_suite.groups.parallel_groups import FanIn, FanInGrouped, \
     FanOut, FanOutGrouped, ConvergeDivergeFlat, \
     ConvergeDivergeGroups, Diamond, DiamondFlat
from openmdao.solvers.tests.linear_test_base import LinearSolverTests

class TestLinearBlockJacSolver(LinearSolverTests.LinearSolverTestCase):

    ln_solver_class = LinearBlockJac

    def test_globaljac_err(self):
        prob = Problem()
        model = prob.model = Group()
        model.add_subsystem('x_param', IndepVarComp('length', 3.0),
                            promotes=['length'])
        model.add_subsystem('mycomp', TestExplCompSimpleDense(),
                            promotes=['length', 'width', 'area'])

        model.ln_solver = LinearBlockJac()
        prob.set_solver_print(level=0)

        prob.model.jacobian = AssembledJacobian()
        prob.setup(check=False, mode='fwd')

        prob['width'] = 2.0
        prob.run_model()

        of = ['area']
        wrt = ['length']

        with self.assertRaises(RuntimeError) as context:
            prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')

            self.assertEqual(str(context.exception),
                             "A block linear solver 'LN: LNBJ' is being used with"
                             " an AssembledJacobian in system ''")


class TestBJacSolverFeature(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockJac()

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61001056, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78448534, .00001)

    def test_feature_maxiter(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockJac()
        model.ln_solver.options['maxiter'] = 5

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.60230118004, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78022500547, .00001)

    def test_feature_atol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockJac()
        model.ln_solver.options['atol'] = 1.0e-3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78456955704, .00001)

    def test_feature_rtol(self):
        prob = Problem()
        model = prob.model = SellarDerivatives()

        model.ln_solver = LinearBlockJac()
        model.ln_solver.options['rtol'] = 1.0e-3

        prob.setup()
        prob.run_model()

        wrt = ['z']
        of = ['obj']

        J = prob.compute_total_derivs(of=of, wrt=wrt, return_format='flat_dict')
        assert_rel_error(self, J['obj', 'z'][0][0], 9.61016296175, .00001)
        assert_rel_error(self, J['obj', 'z'][0][1], 1.78456955704, .00001)

if __name__ == "__main__":
    unittest.main()
