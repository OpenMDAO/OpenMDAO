
import openmdao.api as om
import numpy as np

class BadConnection(om.Group):

    def setup(self):
        indeps = self.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('x', 10*np.ones(4))

        # c2.y is implicitly connected to c1.y
        self.add_subsystem('c1', om.ExecComp('y = 2*x', x=np.ones(4), y=2*np.ones(4)),
                           promotes=['y'])
        self.add_subsystem('c2', om.ExecComp('z = 2*y', y=np.ones(4), z=2*np.ones(4)),
                           promotes=['y'])

        # make a second, explicit, connection to y (which is c2.y promoted)
        self.connect('indeps.x', 'y')

if __name__ == '__main__':
    p = om.Problem(model=BadConnection())
    p.setup()
    p.run_model()
