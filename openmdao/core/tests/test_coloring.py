from __future__ import print_function

import unittest
import numpy as np
import math

from openmdao.api import Problem, IndepVarComp, ExecComp
from openmdao.devtools.testutil import assert_rel_error

from openmdao.utils.general_utils import set_pyoptsparse_opt

# check that pyoptsparse is installed
# if it is, try to use SNOPT but fall back to SLSQP
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')

if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class SimulColoringTestCase(unittest.TestCase):

    def test_simul_coloring(self):
        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest()

        size = 10
        assert(size >= 4)
        p = Problem()
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        model = p.model
        indep = model.add_subsystem('indep', IndepVarComp('x', val=np.ones(size)))
        indep.add_output('y', val=np.ones(size))
        thetas = np.linspace(0., math.pi/4., num=size-4)
        #thetas[:4] = np.array([math.pi/2., math.pi, math.pi*3./2., 0.])

        indep.add_output('theta', val=thetas)
        indep.add_output('r', val=1.0)

        circle = model.add_subsystem('circle', ExecComp('area=pi*r*r'))

        con1 = model.add_subsystem('con1', ExecComp('con=x*x + y*y - r*r',
                                                    x=np.zeros(size), y=np.zeros(size),
                                                    con=np.zeros(size)))
        con2 = model.add_subsystem('con2', ExecComp('con=tan(y/x) - theta',
                                                    x=np.zeros(size-4), y=np.zeros(size-4),
                                                    theta=np.zeros(size-4),
                                                    con=np.zeros(size-4)))

        lconx0 = model.add_subsystem('lconx0', ExecComp('con=x'))
        lconx1 = model.add_subsystem('lconx1', ExecComp('con=x+1'))
        lconx2 = model.add_subsystem('lconx2', ExecComp('con=x'))
        lconx3 = model.add_subsystem('lconx3', ExecComp('con=x-1'))

        lcony0 = model.add_subsystem('lcony0', ExecComp('con=y-1'))
        lcony1 = model.add_subsystem('lcony1', ExecComp('con=y'))
        lcony2 = model.add_subsystem('lcony2', ExecComp('con=y+1'))
        lcony3 = model.add_subsystem('lcony3', ExecComp('con=y'))

        ind = list(range(4, size))

        model.connect('indep.x', 'con1.x')#, src_indices=ind)
        model.connect('indep.x', 'con2.x', src_indices=ind)
        model.connect('indep.y', 'con1.y')#, src_indices=ind)
        model.connect('indep.y', 'con2.y', src_indices=ind)
        model.connect('indep.theta', 'con2.theta')
        model.connect('indep.r', ('circle.r', 'con1.r')) #, 'lconx1.r', 'lconx3.r', 'lcony0.r', 'lcony2.r'))

        model.connect('indep.x', 'lconx0.x', src_indices=[0])
        model.connect('indep.x', 'lconx1.x', src_indices=[1])
        model.connect('indep.x', 'lconx2.x', src_indices=[2])
        model.connect('indep.x', 'lconx3.x', src_indices=[3])

        model.connect('indep.y', 'lcony0.y', src_indices=[0])
        model.connect('indep.y', 'lcony1.y', src_indices=[1])
        model.connect('indep.y', 'lcony2.y', src_indices=[2])
        model.connect('indep.y', 'lcony3.y', src_indices=[3])

        model.add_design_var('indep.x')
        model.add_design_var('indep.y')
        model.add_design_var('indep.r', upper=10.)

        model.add_objective('circle.area', scaler=-1.)

        lin=True
        model.add_constraint('lconx0.con', equals=0., linear=lin)
        model.add_constraint('lconx1.con', equals=0., linear=lin)
        model.add_constraint('lconx2.con', equals=0., linear=lin)
        model.add_constraint('lconx3.con', equals=0., linear=lin)

        model.add_constraint('con1.con', equals=0.)
        model.add_constraint('con2.con', equals=0.)

        model.add_constraint('lcony0.con', equals=0., linear=lin)
        model.add_constraint('lcony1.con', equals=0., linear=lin)
        model.add_constraint('lcony2.con', equals=0., linear=lin)
        model.add_constraint('lcony3.con', equals=0., linear=lin)

        p.setup(mode='fwd')
        p.run_driver()

        print('con1.con', p['con1.con'])
        print('x', p['indep.x'])
        print('y', p['indep.y'])
        J = p.driver._compute_totals(return_format='array')
        from openmdao.utils.array_utils import array_viz
        array_viz(J)
