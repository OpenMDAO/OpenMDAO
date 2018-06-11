from __future__ import print_function

import os
import shutil
import tempfile

import unittest
import numpy as np
import math

from numpy.testing import assert_array_almost_equal, assert_almost_equal

from openmdao.api import Problem, IndepVarComp, ExecComp, DenseJacobian, DirectSolver,\
    ExplicitComponent, LinearRunOnce, ScipyOptimizeDriver
from openmdao.utils.assert_utils import assert_rel_error

from openmdao.utils.general_utils import set_pyoptsparse_opt

# check that pyoptsparse is installed
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


def run_opt(driver_class, mode, color_info=None, sparsity=None, **options):

    # note: size must be an even number
    SIZE = 10
    p = Problem()

    p.model.linear_solver = RunOnceCounter()
    indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

    # the following were randomly generated using np.random.random(10)*2-1 to randomly
    # disperse them within a unit circle centered at the origin.
    indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                      0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
    indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                     -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
    indeps.add_output('r', .7)

    p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

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

    p.model.add_subsystem('l_conx', ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))

    p.model.connect('r', ('circle.r', 'r_con.r'))
    p.model.connect('x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])

    p.model.connect('x', 'l_conx.x')

    p.model.connect('y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

    p.driver = driver_class()
    p.driver.options.update(options)

    p.model.add_design_var('x')
    p.model.add_design_var('y')
    p.model.add_design_var('r', lower=.5, upper=10)

    # nonlinear constraints
    p.model.add_constraint('r_con.g', equals=0)

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[0::2]  # all odd indices
    p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=ODD_IND)
    p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

    # this constrains x[0] to be 1 (see definition of l_conx)
    p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

    # linear constraint
    p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

    p.model.add_objective('circle.area', ref=-1)

    # # setup coloring
    if color_info is not None:
        p.driver.set_simul_deriv_color(color_info)
    elif sparsity is not None:
        p.driver.set_total_jac_sparsity(sparsity)

    p.setup(mode=mode)
    p.run_driver()

    return p


class SimulColoringTestCase(unittest.TestCase):

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)

        color_info = [[
           [20],   # uncolored columns
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19]   # color 4
        ],
        [
           [1, 11, 12, 17],   # column 0
           [2, 17],   # column 1
           [3, 13, 18],   # column 2
           [4, 18],   # column 3
           [5, 14, 19],   # column 4
           [6, 19],   # column 5
           [7, 15, 20],   # column 6
           [8, 20],   # column 7
           [9, 16, 21],   # column 8
           [10, 21],   # column 9
           [1, 12, 17],   # column 10
           [2, 17],   # column 11
           [3, 13, 18],   # column 12
           [4, 18],   # column 13
           [5, 14, 19],   # column 14
           [6, 19],   # column 15
           [7, 15, 20],   # column 16
           [8, 20],   # column 17
           [9, 16, 21],   # column 18
           [10, 21],   # column 19
           None   # column 20
        ],
        {
        "circle.area": {
           "indeps.x": [[], [], [1, 10]],
           "indeps.y": [[], [], [1, 10]],
           "indeps.r": [[0], [0], [1, 1]]
        },
        "r_con.g": {
           "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
           "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
           "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
        },
        "l_conx.g": {
           "indeps.x": [[0], [0], [1, 10]],
           "indeps.y": [[], [], [1, 10]],
           "indeps.r": [[], [], [1, 1]]
        },
        "theta_con.g": {
           "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
           "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
           "indeps.r": [[], [], [5, 1]]
        },
        "delta_theta_con.g": {
           "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
           "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
           "indeps.r": [[], [], [5, 1]]
        }
        }]
        p_color = run_opt(pyOptSparseDriver, 'fwd', color_info, optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_dynamic_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)
        p_color = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False,
                          dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21 * 4) / 5)

    def test_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        color_info = [[
           [20],   # uncolored columns
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19]   # color 4
        ],
        [
           [1, 11, 12, 17],   # column 0
           [2, 17],   # column 1
           [3, 13, 18],   # column 2
           [4, 18],   # column 3
           [5, 14, 19],   # column 4
           [6, 19],   # column 5
           [7, 15, 20],   # column 6
           [8, 20],   # column 7
           [9, 16, 21],   # column 8
           [10, 21],   # column 9
           [1, 12, 17],   # column 10
           [2, 17],   # column 11
           [3, 13, 18],   # column 12
           [4, 18],   # column 13
           [5, 14, 19],   # column 14
           [6, 19],   # column 15
           [7, 15, 20],   # column 16
           [8, 20],   # column 17
           [9, 16, 21],   # column 18
           [10, 21],   # column 19
           None   # column 20
        ],
        {
        "circle.area": {
           "indeps.x": [[], [], [1, 10]],
           "indeps.y": [[], [], [1, 10]],
           "indeps.r": [[0], [0], [1, 1]]
        },
        "r_con.g": {
           "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
           "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
           "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
        },
        "l_conx.g": {
           "indeps.x": [[0], [0], [1, 10]],
           "indeps.y": [[], [], [1, 10]],
           "indeps.r": [[], [], [1, 1]]
        },
        "theta_con.g": {
           "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
           "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
           "indeps.r": [[], [], [5, 1]]
        },
        "delta_theta_con.g": {
           "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
           "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
           "indeps.r": [[], [], [5, 1]]
        }
        }]

        p_color = run_opt(pyOptSparseDriver, 'fwd', color_info, optimizer='SLSQP', print_results=False)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

    def test_dynamic_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        p_color = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False,
                          dynamic_simul_derivs=True)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)
        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21 * 4) / 5)


class SimulColoringScipyTestCase(unittest.TestCase):

    def test_simul_coloring(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, 'fwd', optimizer='SLSQP', disp=False)

        color_info = [[
               [20],   # uncolored columns
               [0, 2, 4, 6, 8],   # color 1
               [1, 3, 5, 7, 9],   # color 2
               [10, 12, 14, 16, 18],   # color 3
               [11, 13, 15, 17, 19]   # color 4
            ],
            [
               [1, 11, 16, 21],   # column 0
               [2, 16],   # column 1
               [3, 12, 17],   # column 2
               [4, 17],   # column 3
               [5, 13, 18],   # column 4
               [6, 18],   # column 5
               [7, 14, 19],   # column 6
               [8, 19],   # column 7
               [9, 15, 20],   # column 8
               [10, 20],   # column 9
               [1, 11, 16],   # column 10
               [2, 16],   # column 11
               [3, 12, 17],   # column 12
               [4, 17],   # column 13
               [5, 13, 18],   # column 14
               [6, 18],   # column 15
               [7, 14, 19],   # column 16
               [8, 19],   # column 17
               [9, 15, 20],   # column 18
               [10, 20],   # column 19
               None   # column 20
            ], None]

        p_color = run_opt(ScipyOptimizeDriver, 'fwd', color_info, optimizer='SLSQP', disp=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

    def test_dynamic_simul_coloring(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, 'fwd', optimizer='SLSQP', disp=False)
        p_color = run_opt(ScipyOptimizeDriver, 'fwd', optimizer='SLSQP', disp=False, dynamic_simul_derivs=True)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - dynamic case does 3 full compute_totals to compute coloring, which adds 21 * 3 solves
        # - (total_solves - N) / (solves_per_iter) should be equal between the two cases,
        # - where N is 21 for the uncolored case and 21 * 4 for the dynamic colored case.
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21 * 4) / 5)

    def test_simul_coloring_example(self):

        from openmdao.api import Problem, IndepVarComp, ExecComp, ScipyOptimizeDriver
        import numpy as np

        # note: size must be an even number
        SIZE = 10
        p = Problem()

        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes_outputs=['*'])

        # the following were randomly generated using np.random.random(10)*2-1 to randomly
        # disperse them within a unit circle centered at the origin.
        indeps.add_output('x', np.array([ 0.55994437, -0.95923447,  0.21798656, -0.02158783,  0.62183717,
                                          0.04007379,  0.46044942, -0.10129622,  0.27720413, -0.37107886]))
        indeps.add_output('y', np.array([ 0.52577864,  0.30894559,  0.8420792 ,  0.35039912, -0.67290778,
                                          -0.86236787, -0.97500023,  0.47739414,  0.51174103,  0.10052582]))
        indeps.add_output('r', .7)

        p.model.add_subsystem('circle', ExecComp('area=pi*r**2'))

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

        p.model.add_subsystem('l_conx', ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))

        p.model.connect('r', ('circle.r', 'r_con.r'))
        p.model.connect('x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])

        p.model.connect('x', 'l_conx.x')

        p.model.connect('y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

        p.driver = ScipyOptimizeDriver()
        p.driver.options['optimizer'] = 'SLSQP'
        p.driver.options['disp'] = False

        p.model.add_design_var('x')
        p.model.add_design_var('y')
        p.model.add_design_var('r', lower=.5, upper=10)

        # nonlinear constraints
        p.model.add_constraint('r_con.g', equals=0)

        IND = np.arange(SIZE, dtype=int)
        ODD_IND = IND[0::2]  # all odd indices
        p.model.add_constraint('theta_con.g', lower=-1e-5, upper=1e-5, indices=ODD_IND)
        p.model.add_constraint('delta_theta_con.g', lower=-1e-5, upper=1e-5)

        # this constrains x[0] to be 1 (see definition of l_conx)
        p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])

        # linear constraint
        p.model.add_constraint('y', equals=0, indices=[0,], linear=True)

        p.model.add_objective('circle.area', ref=-1)

        # setup coloring
        color_info = ([
           [20],   # uncolored column list
           [0, 2, 4, 6, 8],   # color 1
           [1, 3, 5, 7, 9],   # color 2
           [10, 12, 14, 16, 18],   # color 3
           [11, 13, 15, 17, 19],   # color 4
        ],
        [
           [1, 11, 16, 21],   # column 0
           [2, 16],   # column 1
           [3, 12, 17],   # column 2
           [4, 17],   # column 3
           [5, 13, 18],   # column 4
           [6, 18],   # column 5
           [7, 14, 19],   # column 6
           [8, 19],   # column 7
           [9, 15, 20],   # column 8
           [10, 20],   # column 9
           [1, 11, 16],   # column 10
           [2, 16],   # column 11
           [3, 12, 17],   # column 12
           [4, 17],   # column 13
           [5, 13, 18],   # column 14
           [6, 18],   # column 15
           [7, 14, 19],   # column 16
           [8, 19],   # column 17
           [9, 15, 20],   # column 18
           [10, 20],   # column 19
           None,   # column 20
        ], None)

        p.driver.set_simul_deriv_color(color_info)

        p.setup(mode='fwd')
        p.run_driver()

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)


class SparsityTestCase(unittest.TestCase):

    def setUp(self):
        self.startdir = os.getcwd()
        self.tempdir = tempfile.mkdtemp(prefix='SparsityTestCase-')
        os.chdir(self.tempdir)

        self.sparsity = {
            "circle.area": {
               "indeps.x": [[], [], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[0], [0], [1, 1]]
            },
            "r_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.y": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 10]],
               "indeps.r": [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [10, 1]]
            },
            "theta_con.g": {
               "indeps.x": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.y": [[0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "delta_theta_con.g": {
               "indeps.x": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.y": [[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [5, 10]],
               "indeps.r": [[], [], [5, 1]]
            },
            "l_conx.g": {
               "indeps.x": [[0], [0], [1, 10]],
               "indeps.y": [[], [], [1, 10]],
               "indeps.r": [[], [], [1, 1]]
            }
        }

    def tearDown(self):
        os.chdir(self.startdir)
        try:
            shutil.rmtree(self.tempdir)
        except OSError:
            pass

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_sparsity_snopt(self):
        # first, run without sparsity
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SNOPT', print_results=False)

        # run with dynamic sparsity
        p_dynamic = run_opt(pyOptSparseDriver, 'fwd', dynamic_derivs_sparsity=True,
                            optimizer='SNOPT', print_results=False)

        # run with provided sparsity
        p_sparsity = run_opt(pyOptSparseDriver, 'fwd', sparsity=self.sparsity,
                             optimizer='SNOPT', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_dynamic['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_sparsity['circle.area'], np.pi, decimal=7)

    def test_sparsity_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        # first, run without sparsity
        p = run_opt(pyOptSparseDriver, 'fwd', optimizer='SLSQP', print_results=False)

        # run with dynamic sparsity
        p_dynamic = run_opt(pyOptSparseDriver, 'fwd', dynamic_derivs_sparsity=True,
                            optimizer='SLSQP', print_results=False)

        # run with provided sparsity
        p_sparsity = run_opt(pyOptSparseDriver, 'fwd', sparsity=self.sparsity,
                             optimizer='SLSQP', print_results=False)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_dynamic['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_sparsity['circle.area'], np.pi, decimal=7)


if __name__ == '__main__':
    unittest.main()
