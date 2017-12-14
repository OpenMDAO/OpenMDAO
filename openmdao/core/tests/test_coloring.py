from __future__ import print_function

import unittest
import numpy as np
import math

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from openmdao.api import Problem, IndepVarComp, ExecComp, DenseJacobian, DirectSolver,\
    ExplicitComponent, LinearRunOnce
from openmdao.devtools.testutil import assert_rel_error

from openmdao.utils.general_utils import set_pyoptsparse_opt

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class RunOnceCounter(LinearRunOnce):
    def __init__(self, *args, **kwargs):
        self._solve_count = 0
        super(RunOnceCounter, self).__init__(*args, **kwargs)

    def _iter_execute(self):
        super(RunOnceCounter, self)._iter_execute()
        self._solve_count += 1

def run_opt(SIZE, color_info=None):

    p = Problem()

    p.model.linear_solver = RunOnceCounter()

    indeps = p.model.add_subsystem('indeps', IndepVarComp())
    indeps.add_output('x', np.ones(SIZE))
    indeps.add_output('y', np.ones(SIZE))
    indeps.add_output('r', 1.)

    p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

    # nonlinear constraints
    p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r',
                                            g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)
    p.model.add_subsystem('theta_con', ExecComp('g=arctan(y/x) - theta',
                                                g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE),
                                                theta=thetas))
    p.model.add_subsystem('delta_theta_con', ExecComp('g = arctan(y/x)[::2]-arctan(y/x)[1::2]',
                                                      g=np.ones(SIZE//2), x=np.ones(SIZE),
                                                      y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)


    # linear constraint
    p.model.add_subsystem('l_conx', ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))

    p.model.connect('indeps.r', ('circle.r', 'r_con.r'))
    p.model.connect('indeps.x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])

    p.model.connect('indeps.x', 'l_conx.x')

    p.model.connect('indeps.y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = OPTIMIZER
    p.driver.options['print_results'] = False

    p.model.add_design_var('indeps.x')
    p.model.add_design_var('indeps.y')
    p.model.add_design_var('indeps.r', lower=.5, upper=10)
    p.model.add_constraint('r_con.g', equals=0)

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[0::2]
    p.model.add_constraint('theta_con.g', equals=0, indices=ODD_IND)
    p.model.add_constraint('delta_theta_con.g', equals=0)

    #TODO: setting this one to true breaks the optimization
    # p.model.add_constraint('l_conx.g', equals=0, linear=False)
    p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])
    p.model.add_constraint('indeps.y', equals=0, indices=[0,], linear=True)

    p.model.add_objective('circle.area', ref=-1)

    # # setup coloring
    if color_info is not None:
        p.driver.set_simul_coloring(color_info)

    p.setup(mode='fwd')
    p.run_driver()

    return p

class SimulColoringTestCase(unittest.TestCase):

    def test_simul_coloring(self):
        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest("This test needs SNOPT.")

        #note: size must always be an even number!!

        # first, run w/o coloring
        p = run_opt(10)

        color_info = (
        {
            'indeps.y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'indeps.x': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        },
        {
            'delta_theta_con.g': {
                'indeps.y': {
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),
                    1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])},
                'indeps.x': {
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),
                    1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])}},
            'r_con.g': {
                'indeps.y': {
                    0: ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8]),
                    1: ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])},
                'indeps.x': {
                    0: ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8]),
                    1: ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])}},
            'l_conx.g': {
                'indeps.x': {
                    0: ([0], [0])}},
            'theta_con.g': {
                'indeps.y': {
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])},
                'indeps.x': {
                    0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])
                }
            }
        })

        p_color = run_opt(10, color_info)

        assert_almost_equal(p['circle.area'], p_color['circle.area'], decimal=7)

        # 6 calls to _gradfunc, coloring saves 16 solves per driver iter  (5 vs 21)
        # 16 * 6 = 96
        self.assertEqual(p.model.linear_solver._solve_count - 96,
                         p_color.model.linear_solver._solve_count)

if __name__ == '__main__':
    #unittest.main()
    run_opt(10)
