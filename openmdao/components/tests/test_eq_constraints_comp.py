from __future__ import print_function, division, absolute_import

import os
import unittest

import numpy as np
from numpy.testing import assert_almost_equal
from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyOptimizeDriver
from openmdao.components.eq_constraints_comp import EqualityConstraintsComp
from openmdao.utils.general_utils import printoptions

from openmdao.utils.general_utils import set_pyoptsparse_opt, run_driver

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

# pyOptSparseDriver = None


class TestEqualityConstraintsComp(unittest.TestCase):

    def test_two_parabolas(self):
        """
        two parabolas intersecting at:
            (1.3979185,  1.84999737)
            (3.6609049, -1.34480698)
        """
        prob = Problem()
        model = prob.model

        indep = IndepVarComp()
        indep.add_output('x', val=4.)

        equal = EqualityConstraintsComp()
        equal.add_eq_output('x')

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=1.5*x**2-9*x+11.5', x=0.))
        model.add_subsystem('g', ExecComp('y=-0.2*x**2-0.4*x+2.8', x=0.))
        model.add_subsystem('equal', equal)

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:x')
        model.connect('g.y', 'equal.rhs:x')

        model.add_design_var('indep.x', lower=0., upper=5.)
        model.add_constraint('equal.x', equals=0.)  # TODO: auto-add constraint
        model.add_objective('f.y')

        if pyOptSparseDriver:
            prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=True)
        else:
            prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-11, disp=True)

        # prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs', 'totals']

        prob.setup(mode='fwd')

        print('\nOPT:')
        failed = prob.run_driver()
        print('success?', not failed)
        print('x:', prob['indep.x'], 'y:', prob['f.y'], prob['g.y'], 'eq_con:', prob['equal.x'])

        self.assertFalse(failed, 'Optimization failed.')

        assert_almost_equal(prob['equal.x'], 0.0)
        assert_almost_equal(prob['indep.x'], 3.6609049)

    def test_line_parabola(self):
        """
        line and parabola intersecting at:
            (-0.92,  3.62)
            (-4.33, -1.49)
        """
        prob = Problem()
        model = prob.model

        indep = IndepVarComp()
        indep.add_output('x', val=0.)

        equal = EqualityConstraintsComp()
        equal.add_eq_output('x')

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=1.5*x+5.', x=0.))
        model.add_subsystem('g', ExecComp('y=2*x**2+12*x+13', x=0.))
        model.add_subsystem('equal', equal)

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:x')
        model.connect('g.y', 'equal.rhs:x')

        model.add_design_var('indep.x', lower=-5., upper=0.)
        model.add_constraint('equal.x', equals=0.)  # TODO: auto-add constraint
        model.add_objective('f.y')

        if pyOptSparseDriver:
            prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=True)
            if OPTIMIZER == 'SLSQP':
                prob.driver.opt_settings['ACC'] = 1e-9
            if OPTIMIZER == 'SNOPT':
                prob.driver.opt_settings['Major optimality tolerance'] = 1e-9

        else:
            prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=True)

        # prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs', 'totals']

        prob.setup(mode='fwd')

        print('\nOPT:')
        failed = prob.run_driver()
        print('success?', not failed)
        print('x:', prob['indep.x'], 'y:', prob['f.y'], prob['g.y'], 'eq_con:', prob['equal.x'])

        self.assertFalse(failed, 'Optimization failed.')

        assert_almost_equal(prob['equal.x'], 0.0)
        # assert_almost_equal(prob['indep.x'], 3.6609049)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
