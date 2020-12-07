import openmdao.api as om



class ParaboloidA(om.ExplicitComponent):
    def setup(self):
        self.add_input('x', val=0.0)
        self.add_input('y', val=0.0)

        self.add_output('f_xy', val=0.0)
        self.add_output('g_xy', val=0.0)

        # makes extra calls to the model with no actual steps
        self.declare_partials(of='*', wrt='*', method='fd', form='forward', step=1e-6)

        self.count = 0

    def compute(self, inputs, outputs):
        x = inputs['x']
        y = inputs['y']

        outputs['f_xy'] = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        g_xy = (x-3.0)**2 + x*y + (y+4.0)**2 - 3.0
        outputs['g_xy'] = g_xy * 3

        self.count += 1

if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model
    model.add_subsystem('px', om.IndepVarComp('x', val=3.0))
    model.add_subsystem('py', om.IndepVarComp('y', val=5.0))
    model.add_subsystem('parab', ParaboloidA())

    model.connect('px.x', 'parab.x')
    model.connect('py.y', 'parab.y')

    model.add_design_var('px.x', lower=-50, upper=50)
    model.add_design_var('py.y', lower=-50, upper=50)
    model.add_objective('parab.f_xy')

    prob.setup()
    prob.run_model()

    om.n2(prob, show_browser=True)