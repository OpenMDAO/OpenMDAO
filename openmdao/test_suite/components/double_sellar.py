from openmdao.core.group import Group
from openmdao.solvers.nonlinear.newton import NewtonSolver
from openmdao.solvers.linear.direct import DirectSolver
from openmdao.test_suite.components.sellar import SellarDis1withDerivatives, SellarDis2withDerivatives
from openmdao.test_suite.components.sellar import SellarImplicitDis1

# TODO -- Need to convert these over to use `setup` after we have two setup
# phases split up and have the ability to change driver settings


class SubSellar(Group):

    def __init__(self, units=None, scaling=None, **kwargs):
        super(SubSellar, self).__init__(**kwargs)

        self.add_subsystem('d1', SellarDis1withDerivatives(units=units, scaling=scaling),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarDis2withDerivatives(units=units, scaling=scaling),
                           promotes=['z', 'y1', 'y2'])


class DoubleSellar(Group):

    def __init__(self, units=None, scaling=None, **kwargs):
        super(DoubleSellar, self).__init__(**kwargs)

        self.add_subsystem('g1', SubSellar(units=units, scaling=scaling))
        self.add_subsystem('g2', SubSellar(units=units, scaling=scaling))

        self.connect('g1.y2', 'g2.x')
        self.connect('g2.y2', 'g1.x')

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        self.nonlinear_solver = NewtonSolver()
        self.linear_solver = DirectSolver()


class SubSellarImplicit(Group):

    def __init__(self, units=None, scaling=None, **kwargs):
        super(SubSellarImplicit, self).__init__(**kwargs)

        self.add_subsystem('d1', SellarImplicitDis1(units=units, scaling=scaling),
                           promotes=['x', 'z', 'y1', 'y2'])
        self.add_subsystem('d2', SellarImplicitDis1(units=units, scaling=scaling),
                           promotes=['z', 'y1', 'y2'])


class DoubleSellarImplicit(Group):

    def __init__(self, units=None, scaling=None, **kwargs):
        super(DoubleSellarImplicit, self).__init__(**kwargs)

        self.add_subsystem('g1', SubSellar(units=units, scaling=scaling))
        self.add_subsystem('g2', SubSellar(units=units, scaling=scaling))

        self.connect('g1.y2', 'g2.x')
        self.connect('g2.y2', 'g1.x')

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        self.nonlinear_solver = NewtonSolver()
        self.linear_solver = DirectSolver()
