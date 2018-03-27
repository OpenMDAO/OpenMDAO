from __future__ import print_function

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

def run_opt(driver_class, options, color_info=None):

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

    p.setup(mode='fwd')
    p.run_driver()

    return p


class SimulColoringTestCase(unittest.TestCase):

    @unittest.skipUnless(OPTIMIZER == 'SNOPT', "This test requires SNOPT.")
    def test_simul_coloring_snopt(self):
        # first, run w/o coloring
        p = run_opt(pyOptSparseDriver,
                    {'optimizer': 'SNOPT', 'print_results': False})

        color_info = ([
           [20],
           [0, 2, 4, 6, 8],
           [1, 3, 5, 7, 9],
           [10, 12, 14, 16, 18],
           [11, 13, 15, 17, 19],
        ],
        {
           0: [1, 11, 12, 17],
           1: [2, 17],
           2: [3, 13, 18],
           3: [4, 18],
           4: [5, 14, 19],
           5: [6, 19],
           6: [7, 15, 20],
           7: [8, 20],
           8: [9, 16, 21],
           9: [10, 21],
           10: [1, 12, 17],
           11: [2, 17],
           12: [3, 13, 18],
           13: [4, 18],
           14: [5, 14, 19],
           15: [6, 19],
           16: [7, 15, 20],
           17: [8, 20],
           18: [9, 16, 21],
           19: [10, 21],
        },
        {'delta_theta_con.g': {'indeps.x': ([0, 1, 2, 2, 3, 3, 4, 4],
                                            [6, 8, 0, 1, 2, 3, 4, 5],
                                            (5, 10)),
                               'indeps.y': ([0, 1, 2, 2, 3, 3, 4, 4],
                                            [6, 8, 0, 1, 2, 3, 4, 5],
                                            (5, 10))},
         'l_conx.g': {'indeps.x': ([0, 0], [6, 7], (1, 10)),
                      'indeps.y': ([0, 0], [6, 7], (1, 10))},
         'r_con.g': {'indeps.r': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  (10, 1)),
                     'indeps.x': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  (10, 10)),
                     'indeps.y': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  (10, 10))},
         'theta_con.g': {'indeps.r': ([0], [0], (5, 1)),
                         'indeps.x': ([0, 1, 2, 3, 4], [9, 0, 0, 2, 4], (5, 10)),
                         'indeps.y': ([0, 2, 3, 4], [9, 0, 2, 4], (5, 10))}}
        )

        p_color = run_opt(pyOptSparseDriver, {'optimizer': 'SNOPT', 'print_results': False},
                          color_info)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

    def test_simul_coloring_pyoptsparse_slsqp(self):
        try:
            from pyoptsparse import OPT
        except ImportError:
            raise unittest.SkipTest("This test requires pyoptsparse.")

        try:
            OPT('SLSQP')
        except:
            raise unittest.SkipTest("This test requires pyoptsparse SLSQP.")

        color_info = ([
           [20],
           [0, 2, 4, 6, 8],
           [1, 3, 5, 7, 9],
           [10, 12, 14, 16, 18],
           [11, 13, 15, 17, 19],
        ],
        {
           0: [1, 11, 12, 17],
           1: [2, 17],
           2: [3, 13, 18],
           3: [4, 18],
           4: [5, 14, 19],
           5: [6, 19],
           6: [7, 15, 20],
           7: [8, 20],
           8: [9, 16, 21],
           9: [10, 21],
           10: [1, 12, 17],
           11: [2, 17],
           12: [3, 13, 18],
           13: [4, 18],
           14: [5, 14, 19],
           15: [6, 19],
           16: [7, 15, 20],
           17: [8, 20],
           18: [9, 16, 21],
           19: [10, 21],
        },
        {'delta_theta_con.g': {'indeps.x': ([0, 1, 2, 2, 3, 3, 4, 4],
                                            [6, 8, 0, 1, 2, 3, 4, 5],
                                            (5, 10)),
                               'indeps.y': ([0, 1, 2, 2, 3, 3, 4, 4],
                                            [6, 8, 0, 1, 2, 3, 4, 5],
                                            (5, 10))},
         'l_conx.g': {'indeps.x': ([0, 0], [6, 7], (1, 10)),
                      'indeps.y': ([0, 0], [6, 7], (1, 10))},
         'r_con.g': {'indeps.r': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  (10, 1)),
                     'indeps.x': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  (10, 10)),
                     'indeps.y': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                  (10, 10))},
         'theta_con.g': {'indeps.r': ([0], [0], (5, 1)),
                         'indeps.x': ([0, 1, 2, 3, 4], [9, 0, 0, 2, 4], (5, 10)),
                         'indeps.y': ([0, 2, 3, 4], [9, 0, 2, 4], (5, 10))}}
        )

        p_color = run_opt(pyOptSparseDriver, {'optimizer': 'SLSQP', 'print_results': False},
                          color_info)

        # run w/o coloring
        p = run_opt(pyOptSparseDriver, {'optimizer': 'SLSQP', 'print_results': False})

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)


class SimulColoringScipyTestCase(unittest.TestCase):

    def test_simul_coloring(self):

        # first, run w/o coloring
        p = run_opt(ScipyOptimizeDriver, {'optimizer': 'SLSQP', 'disp': False})

        color_info = ([
            [20],
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
            [10, 12, 14, 16, 18],
            [11, 13, 15, 17, 19],
         ],
         {
            0: [1, 11, 16, 21],
            1: [2, 16],
            2: [3, 12, 17],
            3: [4, 17],
            4: [5, 13, 18],
            5: [6, 18],
            6: [7, 14, 19],
            7: [8, 19],
            8: [9, 15, 20],
            9: [10, 20],
            10: [1, 11, 16],
            11: [2, 16],
            12: [3, 12, 17],
            13: [4, 17],
            14: [5, 13, 18],
            15: [6, 18],
            16: [7, 14, 19],
            17: [8, 19],
            18: [9, 15, 20],
            19: [10, 20],
         },
         {'delta_theta_con.g': {'indeps.x': ([0, 1, 1, 2, 2, 3, 3, 4, 4],
                                             [8, 0, 1, 2, 3, 4, 5, 6, 7],
                                             (5, 10)),
                                'indeps.y': ([0, 1, 1, 2, 2, 3, 3, 4, 4],
                                             [8, 0, 1, 2, 3, 4, 5, 6, 7],
                                             (5, 10))},
          'l_conx.g': {'indeps.x': ([0, 0], [8, 9], (1, 10)),
                       'indeps.y': ([0, 0], [8, 9], (1, 10))},
          'r_con.g': {'indeps.r': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   (10, 1)),
                      'indeps.x': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   (10, 10)),
                      'indeps.y': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   (10, 10))},
          'theta_con.g': {'indeps.r': ([0], [0], (5, 1)),
                          'indeps.x': ([0, 1, 2, 3, 4], [9, 0, 2, 4, 6], (5, 10)),
                          'indeps.y': ([0, 1, 2, 3, 4], [9, 0, 2, 4, 6], (5, 10))}}
         )
        p_color = run_opt(ScipyOptimizeDriver, {'optimizer': 'SLSQP', 'disp': False},
                          color_info)

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)
        assert_almost_equal(p_color['circle.area'], np.pi, decimal=7)

        # - coloring saves 16 solves per driver iter  (5 vs 21)
        # - initial solve for linear constraints takes 21 in both cases (only done once)
        # - (total_solves - 21) / (solves_per_iter) should be equal between the two cases
        self.assertEqual((p.model.linear_solver._solve_count - 21) / 21,
                         (p_color.model.linear_solver._solve_count - 21) / 5)

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
            [20],
            [0, 2, 4, 6, 8],
            [1, 3, 5, 7, 9],
            [10, 12, 14, 16, 18],
            [11, 13, 15, 17, 19],
         ],
         {
            0: [1, 11, 16, 21],
            1: [2, 16],
            2: [3, 12, 17],
            3: [4, 17],
            4: [5, 13, 18],
            5: [6, 18],
            6: [7, 14, 19],
            7: [8, 19],
            8: [9, 15, 20],
            9: [10, 20],
            10: [1, 11, 16],
            11: [2, 16],
            12: [3, 12, 17],
            13: [4, 17],
            14: [5, 13, 18],
            15: [6, 18],
            16: [7, 14, 19],
            17: [8, 19],
            18: [9, 15, 20],
            19: [10, 20],
         },
         {'delta_theta_con.g': {'indeps.x': ([0, 1, 1, 2, 2, 3, 3, 4, 4],
                                             [8, 0, 1, 2, 3, 4, 5, 6, 7],
                                             (5, 10)),
                                'indeps.y': ([0, 1, 1, 2, 2, 3, 3, 4, 4],
                                             [8, 0, 1, 2, 3, 4, 5, 6, 7],
                                             (5, 10))},
          'l_conx.g': {'indeps.x': ([0, 0], [8, 9], (1, 10)),
                       'indeps.y': ([0, 0], [8, 9], (1, 10))},
          'r_con.g': {'indeps.r': ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   (10, 1)),
                      'indeps.x': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   (10, 10)),
                      'indeps.y': ([1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 1, 2, 3, 4, 5, 6, 7, 8],
                                   (10, 10))},
          'theta_con.g': {'indeps.r': ([0], [0], (5, 1)),
                          'indeps.x': ([0, 1, 2, 3, 4], [9, 0, 2, 4, 6], (5, 10)),
                          'indeps.y': ([0, 1, 2, 3, 4], [9, 0, 2, 4, 6], (5, 10))}}
         )

        p.driver.set_simul_deriv_color(color_info)

        p.setup(mode='fwd')
        p.run_driver()

        assert_almost_equal(p['circle.area'], np.pi, decimal=7)

if __name__ == '__main__':
    unittest.main()
