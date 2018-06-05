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

pyOptSparseDriver = None


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
        model.add_objective('f.y')

        if pyOptSparseDriver:
            prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=True)
            prob.driver.opt_settings['Major optimality tolerance'] = 1e-9
        else:
            prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-11, disp=True)

        # prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs', 'totals']

        # make sure constraint is not added by default
        prob.setup(mode='fwd')
        self.assertFalse('equal.x' in model.get_constraints())

        model.add_constraint('equal.x', equals=0.)
        prob.setup(mode='fwd')

        print('\nOPT:')
        failed = prob.run_driver()

        print('x:', prob['indep.x'], 'y:', prob['f.y'], prob['g.y'], 'eq_con:', prob['equal.x'])

        if pyOptSparseDriver:
            info = prob.driver.pyopt_solution.optInform
            self.assertFalse(failed, "Optimization failed, info = " +
                                     str(info['value'])+": "+info['text'])
        else:
            self.assertFalse(failed, "Optimization failed, result =\n" +
                                     str(prob.driver.result))

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
        equal.add_eq_output('x', add_constraint=True)

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=1.5*x+5.', x=0.))
        model.add_subsystem('g', ExecComp('y=2*x**2+12*x+13', x=0.))
        model.add_subsystem('equal', equal)

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:x')
        model.connect('g.y', 'equal.rhs:x')

        model.add_design_var('indep.x', lower=-5., upper=0.)
        model.add_objective('f.y')

        if pyOptSparseDriver:
            prob.driver = pyOptSparseDriver(optimizer=OPTIMIZER, print_results=True)
            if OPTIMIZER == 'SLSQP':
                prob.driver.opt_settings['ACC'] = 1e-9
            if OPTIMIZER == 'SNOPT':
                prob.driver.opt_settings['Major optimality tolerance'] = 1e-9

        else:
            prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=True)

        prob.setup(mode='fwd')

        # make sure constraint is added as requested
        self.assertTrue('equal.x' in model.get_constraints())

        prob.run_model()

        cpd = prob.check_partials()

        for (of, wrt) in cpd['equal']:
            assert_almost_equal(cpd['equal'][of, wrt]['abs error'], 0.0, decimal=5)

        # prob.driver.options['debug_print'] = ['desvars', 'nl_cons', 'objs', 'totals']

        print('\nOPT:')
        failed = prob.run_driver()

        print('x:', prob['indep.x'], 'y:', prob['f.y'], prob['g.y'], 'eq_con:', prob['equal.x'])

        if pyOptSparseDriver:
            info = prob.driver.pyopt_solution.optInform
            self.assertFalse(failed, "Optimization failed, info = " +
                                     str(info['value'])+": "+info['text'])
        else:
            self.assertFalse(failed, "Optimization failed, result =\n" +
                                     str(prob.driver.result))


        assert_almost_equal(prob['equal.x'], 0.0)
        # assert_almost_equal(prob['indep.x'], 3.6609049)

    def test_two_lines(self):
        prob = Problem()
        model = prob.model

        indep = IndepVarComp()
        indep.add_output('x', val=0.)

        equal = EqualityConstraintsComp()
        equal.add_eq_output('x', add_constraint=True)

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=3*x-3', x=0.))
        model.add_subsystem('g', ExecComp('y=2.3*x+4', x=0.))
        model.add_subsystem('equal', equal)

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        model.add_design_var('indep.x', lower=0., upper=20.)
        model.add_objective('f.y')

        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=True)

        prob.setup(mode='fwd')

        # check that constraint has been added as requested
        self.assertTrue('equal.y' in model.get_constraints())

        # do one test in an unconverged state, to capture accuracy of partials
        # set rhs and lhs to very different values. Trying to capture some derivatives wrt
        prob['f.y'] = 100000
        prob['g.y'] = .001

        prob.run_model()

        cpd = prob.check_partials()

        for (of, wrt) in cpd['equal']:
            assert_almost_equal(cpd['equal'][of, wrt]['abs error'], 0.0, decimal=5)

        prob.run_driver()

        assert_almost_equal(prob['equal.y'], 0.)
        assert_almost_equal(prob['indep.x'], 10.)
        assert_almost_equal(prob['f.y'], 27.)
        assert_almost_equal(prob['g.y'], 27.)

    def test_feature_two_lines(self):
        prob = Problem()
        model = prob.model

        # create model with two intersecting lines, f and g
        indep = IndepVarComp()
        indep.add_output('x', val=0.)

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=3*x-3', x=0.))
        model.add_subsystem('g', ExecComp('y=2.3*x+4', x=0.))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        # constrain f.y and g.y to be equal
        equal = EqualityConstraintsComp()
        equal.add_eq_output('y', add_constraint=True)

        model.add_subsystem('equals', equals)
        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        # find minimum y for x between 0 and 20
        model.add_design_var('indep.x', lower=0., upper=20.)
        model.add_objective('f.y')

        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=True)

        prob.setup(mode='fwd')
        prob.run_driver()

        # the
        assert_almost_equal(prob['equal.y'], 0.)

        assert_almost_equal(prob['indep.x'], 10.)
        assert_almost_equal(prob['f.y'], 27.)
        assert_almost_equal(prob['g.y'], 27.)

    def plot_feature_line_parabola():
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.linspace(-5, 5, 100)
        f = [1.5*x+5 for x in x]
        g = [2*x**2+12*x+13 for x in x]

        fig, ax = plt.subplots()
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        plt.xlim(-5, 1)
        plt.ylim(-5, 5)
        plt.xticks(range(-5, 1, 1))
        plt.yticks(range(-5, 5, 1))

        plt.plot(x, f)
        plt.plot(x, g)
        plt.grid(True)
        plt.show()

    def test_feature_line_parabola(self):
        prob = Problem()
        model = prob.model

        indep = IndepVarComp()
        indep.add_output('x', val=-5.)

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=1.5*x+5.', x=0.))
        model.add_subsystem('g', ExecComp('y=2*x**2+12*x+13', x=0.))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        # constrain f.y and g.y to be equal
        equal = EqualityConstraintsComp()
        equal.add_eq_output('y', add_constraint=True)

        model.add_subsystem('equal', equal)
        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        # find minimum y where x is between -5 and 0
        model.add_design_var('indep.x', lower=-5., upper=0.)
        model.add_objective('f.y')

        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=True)
        prob.setup(mode='fwd')

        prob.run_driver()

        print('x:', prob['indep.x'], 'y:', prob['f.y'], prob['g.y'], 'eq_con:', prob['equal.y'])

        # assert_almost_equal(prob['equal.y'], 0.0)
        # assert_almost_equal(prob['indep.x'], 3.6609049)

    def plot_feature_two_parabolas():
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.linspace(-5, 5, 100)
        f = [1.5*x**2-9*x+11.5 for x in x]
        g = [-0.2*x**2-0.4*x+2.8 for x in x]

        fig, ax = plt.subplots()
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        plt.xlim(-2, 5)
        plt.ylim(-5, 5)
        plt.xticks(range(-5, 5, 1))
        plt.yticks(range(-5, 5, 1))

        plt.plot(x, f)
        plt.plot(x, g)

        offset = 1.05

        plt.plot(1.39,  1.85, 'ko')
        plt.text(1.39*offset,  1.85*offset, '(1.39, 1.85)')

        plt.plot(3.66, -1.34, 'ko')
        plt.text(3.66*offset, -1.34*offset, '(3.66, -1.34)')

        plt.grid(True)
        plt.show()

    def test_feature_two_parabolas(self):
        prob = Problem()
        model = prob.model

        # define our two parabolas as functions of x
        indep = IndepVarComp()
        indep.add_output('x', val=5.)

        model.add_subsystem('indep', indep)
        model.add_subsystem('f', ExecComp('y=1.5*x**2-9*x+11.5', x=0.))
        model.add_subsystem('g', ExecComp('y=-0.2*x**2-0.4*x+2.8', x=0.))

        model.connect('indep.x', 'f.x')
        model.connect('indep.x', 'g.x')

        # constrain f.y and g.y to be equal
        equal = EqualityConstraintsComp()
        equal.add_eq_output('y', add_constraint=True)

        model.add_subsystem('equal', equal)
        model.connect('f.y', 'equal.lhs:y')
        model.connect('g.y', 'equal.rhs:y')

        # find minimum y where x is between 0 and 5
        model.add_design_var('indep.x', lower=0., upper=5.)
        model.add_objective('f.y')

        prob.driver = ScipyOptimizeDriver(optimizer='SLSQP', tol=1e-9, disp=True)

        prob.setup(mode='fwd')
        prob.run_driver()

        assert_almost_equal(prob['indep.x'], 3.6609049)
        assert_almost_equal(prob['f.y'], -1.344807)
        assert_almost_equal(prob['g.y'], -1.344807)


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
