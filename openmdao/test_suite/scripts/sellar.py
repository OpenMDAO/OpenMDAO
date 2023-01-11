import numpy as np

import openmdao.api as om
from openmdao.test_suite.components.sellar import SellarDis1, SellarDis2


class SellarMDAConnect(om.Group):

    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes_inputs=['x', 'z'])
        cycle.add_subsystem('d1', SellarDis1(), promotes_inputs=['x', 'z'])
        cycle.add_subsystem('d2', SellarDis2(), promotes_inputs=['z'])
        cycle.connect('d1.y1', 'd2.y1')

        ######################################
        # This is a "forgotten" connection!!
        ######################################
        #cycle.connect('d2.y2', 'd1.y2')

        cycle.set_input_defaults('x', 1.0)
        cycle.set_input_defaults('z', np.array([5.0, 2.0]))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        self.add_subsystem('obj_cmp', om.ExecComp('obj = x**2 + z[1] + y1 + exp(-y2)',
                                                  z=np.array([0.0, 0.0]), x=0.0),
                           promotes_inputs=['x', 'z'])

        self.add_subsystem('con_cmp1', om.ExecComp('con1 = 3.16 - y1'))
        self.add_subsystem('con_cmp2', om.ExecComp('con2 = y2 - 24.0'))

        self.connect('cycle.d1.y1', ['obj_cmp.y1', 'con_cmp1.y1'])
        self.connect('cycle.d2.y2', ['obj_cmp.y2', 'con_cmp2.y2'])


prob = om.Problem()

prob.model = SellarMDAConnect()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-8

prob.set_solver_print(level=0)

prob.model.add_design_var('x', lower=0, upper=10)
prob.model.add_design_var('z', lower=0, upper=10)
prob.model.add_objective('obj_cmp.obj')
prob.model.add_constraint('con_cmp1.con1', upper=0)
prob.model.add_constraint('con_cmp2.con2', upper=0)

prob.setup()

prob.set_val('x', 2.0)
prob.set_val('z', [0., 0.])

prob.run_driver()
print('minimum found at')
print(prob.get_val('x')[0])
print(prob.get_val('z'))
print('minumum objective')
print(prob.get_val('obj_cmp.obj')[0])
