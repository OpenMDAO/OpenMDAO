from __future__ import print_function

import unittest
import numpy as np
import math

from openmdao.api import Problem, IndepVarComp, ExecComp, DenseJacobian, DirectSolver, ExplicitComponent
from openmdao.devtools.testutil import assert_rel_error

from openmdao.utils.general_utils import set_pyoptsparse_opt

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

class LConX(ExplicitComponent):

    def setup(self):
        self.add_input('x', np.zeros(4))
        self.add_output('con', np.zeros(4))

        self.declare_partials('con', 'x')

    def compute(self, inputs, outputs):
        outputs['con'] = inputs['x'] + np.array([0, 1, 0, -1])

    def compute_partials(self, inputs, partials):
        partials['con', 'x'] = np.eye(4)


def run_opt():
    #note: must always be an even number!!
    SIZE = 10

    p = Problem()

    indeps = p.model.add_subsystem('indeps', IndepVarComp())
    indeps.add_output('x', np.ones(SIZE))
    indeps.add_output('y', np.ones(SIZE))
    indeps.add_output('r', 1.)

    p.model.add_subsystem('area', ExecComp('area=pi*r**2'))

    # nonlinear constraints
    p.model.add_subsystem('r_con', ExecComp('g=x**2 + y**2 - r', g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)
    p.model.add_subsystem('theta_con', ExecComp('g=arctan(y/x) - theta',  g=np.ones(SIZE), x=np.ones(SIZE), y=np.ones(SIZE), theta=thetas))
    p.model.add_subsystem('delta_theta_con', ExecComp('g = arctan(y/x)[::2]-arctan(y/x)[1::2]',  g=np.ones(SIZE//2), x=np.ones(SIZE), y=np.ones(SIZE)))

    thetas = np.linspace(0, np.pi/4, SIZE)


    # linear constraint
    # p.model.add_subsystem('l_conx', ExecComp('g=x-1'))
    p.model.add_subsystem('l_conx', ExecComp('g=x-1', g=np.ones(SIZE), x=np.ones(SIZE)))
    # p.model.add_subsystem('l_conx', ExecComp('g=y')) # not needed, can just set directly

    p.model.connect('indeps.r', ('area.r', 'r_con.r'))
    p.model.connect('indeps.x', ['r_con.x', 'theta_con.x', 'delta_theta_con.x'])

    # p.model.connect('indeps.x', 'l_conx.x', src_indices=[0])
    p.model.connect('indeps.x', 'l_conx.x')

    p.model.connect('indeps.y', ['r_con.y', 'theta_con.y', 'delta_theta_con.y'])

    p.driver = pyOptSparseDriver()
    p.driver.options['optimizer'] = "SNOPT"

    p.model.add_design_var('indeps.x')
    p.model.add_design_var('indeps.y')
    p.model.add_design_var('indeps.r', lower=.5, upper=10)
    p.model.add_constraint('r_con.g', equals=0)

    IND = np.arange(SIZE, dtype=int)
    ODD_IND = IND[0::2]
    p.model.add_constraint('theta_con.g', equals=0, indices=ODD_IND)
    p.model.add_constraint('delta_theta_con.g', equals=0)

    # p.model.add_constraint('l_conx.g', equals=0, linear=False) #TODO: setting this one to true breaks the optimization
    p.model.add_constraint('l_conx.g', equals=0, linear=False, indices=[0,])
    p.model.add_constraint('indeps.y', equals=0, indices=[0,], linear=True)

    p.model.add_objective('area.area', ref=-1)

    # # setup coloring
    # color_info = (
    # {
    #     'indeps.y': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #     'indeps.x': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    # },
    # {
    #     'delta_theta_con.g': {
    #         'indeps.y': {
    #             0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),
    #             1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])},
    #         'indeps.x': {
    #             0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8]),
    #             1: ([0, 1, 2, 3, 4], [1, 3, 5, 7, 9])}},
    #     'r_con.g': {
    #         'indeps.y': {
    #             0: ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8]),
    #             1: ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])},
    #         'indeps.x': {
    #             0: ([0, 2, 4, 6, 8], [0, 2, 4, 6, 8]),
    #             1: ([1, 3, 5, 7, 9], [1, 3, 5, 7, 9])}},
    #     'l_conx.g': {
    #         'indeps.x': {
    #             0: ([0], [0])}},
    #     'theta_con.g': {
    #         'indeps.y': {
    #             0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])},
    #         'indeps.x': {
    #             0: ([0, 1, 2, 3, 4], [0, 2, 4, 6, 8])
    #         }
    #     }
    # })
    # p.driver.set_simul_coloring(color_info)

    p.setup(mode='fwd')
    p.run_driver()

    J = p.driver._compute_totals(return_format='array')
    from openmdao.utils.array_utils import array_viz
    array_viz(J)


class SimulColoringTestCase(unittest.TestCase):

    def test_simul_coloring(self):
        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest()

        run_opt()


if __name__ == '__main__':
    #unittest.main()
    run_opt()
