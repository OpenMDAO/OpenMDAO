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

class SimulColoringTestCase(unittest.TestCase):

    def test_simul_coloring(self):
        if OPTIMIZER is not 'SNOPT':
            raise unittest.SkipTest()

        size = 5
        p = Problem()
        p.driver = pyOptSparseDriver()
        p.driver.options['optimizer'] = OPTIMIZER

        model = p.model
        indep = model.add_subsystem('indep', IndepVarComp())
        indep.add_output('x1', val=np.ones(4))
        indep.add_output('y1', val=np.ones(4))

        indep.add_output('x2', val=np.ones(size))
        indep.add_output('y2', val=np.ones(size))

        thetas = np.linspace(0., math.pi/4., num=size)
        #thetas[:4] = np.array([math.pi/2., math.pi, math.pi*3./2., 0.])

        indep.add_output('theta', val=thetas)
        indep.add_output('r', val=1.0)

        circle = model.add_subsystem('circle', ExecComp('area=pi*r*r'))

        con_r1 = model.add_subsystem('con_r1', ExecComp('con=x*x + y*y - r*r',
                                                    x=np.zeros(4), y=np.zeros(4),
                                                    con=np.zeros(4)))

        con_r2 = model.add_subsystem('con_r2', ExecComp('con=x*x + y*y - r*r',
                                                        x=np.zeros(size), y=np.zeros(size),
                                                        con=np.zeros(size)))
        con2 = model.add_subsystem('con_theta2', ExecComp('con=arctan(y/x) - theta',
                                                 x=np.zeros(size), y=np.zeros(size),
                                                 theta=np.zeros(size),
                                                 con=np.zeros(size)))

        #lconx0 = model.add_subsystem('lconx0', ExecComp('con=x'))
        #lconx1 = model.add_subsystem('lconx1', ExecComp('con=x+1'))
        #lconx2 = model.add_subsystem('lconx2', ExecComp('con=x'))
        #lconx3 = model.add_subsystem('lconx3', ExecComp('con=x-1'))

        #lcony0 = model.add_subsystem('lcony0', ExecComp('con=y-1'))
        #lcony1 = model.add_subsystem('lcony1', ExecComp('con=y'))
        #lcony2 = model.add_subsystem('lcony2', ExecComp('con=y+1'))
        #lcony3 = model.add_subsystem('lcony3', ExecComp('con=y'))

        # lconx = model.add_subsystem('lconx', ExecComp('con=x+c', x=np.zeros(4), c=np.array([0, 1, 0, -1]), con=np.zeros(4)))
        lconx = model.add_subsystem('lconx', LConX())
        lcony = model.add_subsystem('lcony', ExecComp('con=y+c', y=np.zeros(4), c=np.array([-1, 0, 1, 0]), con=np.zeros(4)))

        ind = list(range(4, size))

        #model.connect('indep.x', 'con1.x')#, src_indices=ind)
        #model.connect('indep.x', 'con2.x', src_indices=ind)
        #model.connect('indep.y', 'con1.y')#, src_indices=ind)
        #model.connect('indep.y', 'con2.y', src_indices=ind)
        model.connect('indep.theta', 'con_theta2.theta')
        model.connect('indep.r', ('circle.r', 'con_r1.r', 'con_r2.r')) #, 'lconx1.r', 'lconx3.r', 'lcony0.r', 'lcony2.r'))

        #model.connect('indep.x', 'lconx0.x', src_indices=[0])
        #model.connect('indep.x', 'lconx1.x', src_indices=[1])
        #model.connect('indep.x', 'lconx2.x', src_indices=[2])
        #model.connect('indep.x', 'lconx3.x', src_indices=[3])

        #model.connect('indep.y', 'lcony0.y', src_indices=[0])
        #model.connect('indep.y', 'lcony1.y', src_indices=[1])
        #model.connect('indep.y', 'lcony2.y', src_indices=[2])
        #model.connect('indep.y', 'lcony3.y', src_indices=[3])

        model.connect('indep.x1', 'lconx.x')
        model.connect('indep.y1', 'lcony.y')

        model.connect('indep.x1', 'con_r1.x')
        model.connect('indep.y1', 'con_r1.y')

        model.connect('indep.x2', ['con_r2.x', 'con_theta2.x'])
        model.connect('indep.y2', ['con_r2.y', 'con_theta2.y'])


        model.add_design_var('indep.x1')
        model.add_design_var('indep.x2')
        model.add_design_var('indep.y1')
        model.add_design_var('indep.y2')
        model.add_design_var('indep.r', upper=10.)

        model.add_objective('circle.area', scaler=-1.)

        lin=True
        model.add_constraint('lconx.con', equals=0., linear=True)
        model.add_constraint('lcony.con', equals=0., linear=False)

        model.add_constraint('con_r1.con', equals=0.)
        model.add_constraint('con_r2.con', equals=0.)

        p.model.jacobian = DenseJacobian()
        p.model.linear_solver = DirectSolver()

        p.setup(mode='fwd')
        p.run_driver()

        #print('con1.con', p['con1.con'])
        print('x', p['indep.x1'], p['indep.x2'])
        print('y', p['indep.y1'], p['indep.y2'])
        J = p.driver._compute_totals(return_format='array')
        from openmdao.utils.array_utils import array_viz
        array_viz(J)


if __name__ == '__main__':
    unittest.main()
