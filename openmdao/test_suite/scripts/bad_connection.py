
import openmdao.api as om
import numpy as np

class BadConnectionModel(om.Group):

    def setup(self):
        sub = self.add_subsystem('sub', om.Group())

        idv = sub.add_subsystem('src', om.IndepVarComp())
        idv.add_output('x', np.arange(15).reshape((5, 3)))  # array
        idv.add_output('s', 3.)                             # scalar

        sub.add_subsystem('tgt', om.ExecComp('y = x'))
        sub.add_subsystem('cmp', om.ExecComp('z = x'))
        sub.add_subsystem('arr', om.ExecComp('a = x', x=np.zeros(2)))

        self.sub.connect('tgt.x', 'cmp.x')

if __name__ == '__main__':
    p = om.Problem(model=BadConnectionModel())
    p.setup()
    p.run_model()
