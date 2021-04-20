import unittest
import numpy as np
import openmdao.api as om 
import gc

class TestComponentIOIndependenceFromProb(unittest.TestCase):

    class Parabola(om.ExplicitComponent):

        def __init__(self):
            super(TestComponentIOIndependenceFromProb.Parabola, self).__init__()
            self.x = 0
            self.y = 999
            self.a = 999

        def set_x(self, x):
            ''' Set x in mm'''
            self.x = x

        def setup(self):
            self.add_input('x', units='mm')
            self.add_input('a', units='mm')
            self.add_output('y', units='mm')

        def compute(self, inputs, outputs):
            x = inputs['x']
            a = inputs['a']
            y = (x-a)**2 + 4
            outputs['y'] = y

        def u_run_opt(self):
            prob = om.Problem()
            model = prob.model

            idv = model.add_subsystem('idv', om.IndepVarComp(), promotes=['*'])
            idv.add_output('x', val=self.x, units='mm')
            idv.add_output('a', val=1, units='mm')
            model.add_design_var('a')

            model.add_subsystem('parabola', self, promotes=['*'])

            model.add_objective('y')

            model.approx_totals()

            prob.driver = om.ScipyOptimizeDriver()

            prob.setup()
            prob.run_driver()


    class ParabolaSystem(om.Group):

        def __init__(self):
            super(TestComponentIOIndependenceFromProb.ParabolaSystem, 
                self).__init__()
            self.p = TestComponentIOIndependenceFromProb.Parabola()
            self.x0 = 3
            self.y2 = 999

        def set_x0(self, x0):
            ''' Set x in mm'''
            self.x0=x0

        def setup(self):
            self.add_subsystem('xx0', om.ExecComp('x = x0+7'), promotes=['*'])
            self.p = self.add_subsystem('parabola', self.p, promotes=['*'])
            self.add_subsystem('yy2', om.ExecComp('y2 = y+2'), promotes=['*'])

        def u_run_opt(self):
            prob = om.Problem()
            model = prob.model

            idv = model.add_subsystem('idv', om.IndepVarComp(), promotes=['*'])
            idv.add_output('x0', val=self.x0, units='mm')
            idv.add_output('a', val=1, units='mm')
            model.add_design_var('a')

            model.add_subsystem('parabolasys', self, promotes=['*'])
            model.add_objective('y2')
            model.approx_totals()
            prob.driver = om.ScipyOptimizeDriver()

            prob.setup()
            prob.run_driver()

    def test_parabola(self):
        # Runs the Optimization
        p = TestComponentIOIndependenceFromProb.Parabola()
        p.set_x(3)
        p.u_run_opt()

        # Garbage Collection
        gc.collect()

        # Ensures Variables can be returns once the prob is removed from memory
        a = p.get_val('a', units='mm', from_src=False)[0]
        self.assertAlmostEqual(a, 3)

    def test_parabola_system(self):
        # Runs System the Optimization
        ps = TestComponentIOIndependenceFromProb.ParabolaSystem()
        ps.set_x0(3)
        ps.u_run_opt()

        # Garbage Collection
        gc.collect()

        # Ensures Variables can be returns once the prob is removed from memory
        #   at both the system and componen level
        a = ps.get_val('a', units='mm', from_src=False)[0]
        self.assertAlmostEqual(a, 10)

        y = ps.p.get_val('y', units='mm', from_src=False)[0]
        self.assertAlmostEqual(y, 4)

if __name__ == '__main__':

    unittest.main()
