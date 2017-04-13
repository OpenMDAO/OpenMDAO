from openmdao.core.group import Group
from openmdao.solvers.nl_newton import NewtonSolver
from openmdao.solvers.ln_direct import DirectSolver
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives

class SubSellar(Group):

    def __init__(self, **kwargs):
        super(SubSellar, self).__init__(**kwargs)

        self.add_subsystem('d1', SellarDis1withDerivatives(),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(),
                           promotes=['z', 'y1', 'y2'])


class DoubleSellar(Group):

    def __init__(self, **kwargs):
        super(DoubleSellar, self).__init__(**kwargs)

        self.add_subsystem('g1', SubSellar())
        self.add_subsystem('g2', SubSellar())

        self.connect('g1.y2', 'g2.x')
        self.connect('g2.y2', 'g1.x')

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        self.nl_solver = NewtonSolver()
        self.ln_solver = DirectSolver()
