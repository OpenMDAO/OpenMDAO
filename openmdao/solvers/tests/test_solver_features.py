"""Test the code we put in out main dolver featuer document."""

import unittest

import numpy

from openmdao.api import ExplicitComponent, Problem, Group, NonlinearBlockGS, \
     ScipyIterativeSolver, IndepVarComp, NewtonSolver
from openmdao.devtools.testutil import assert_rel_error


# Note, we are including "clean" versions of the Sellar disciplines for
# showcasing in the feature doc.

class SellarDis1(ExplicitComponent):

    def __init__(self):
        super(SellarDis1, self).__init__()

        # Global Design Variable
        self.add_input('z', val=numpy.zeros(2))

        # Local Design Variable
        self.add_input('x', val=0.)

        # Coupling parameter
        self.add_input('y2', val=1.0)

        # Coupling output
        self.add_output('y1', val=1.0)

    def compute(self, params, unknowns):
        z1 = params['z'][0]
        z2 = params['z'][1]
        x1 = params['x']
        y2 = params['y2']

        unknowns['y1'] = z1**2 + z2 + x1 - 0.2*y2

    def compute_jacobian(self, params, unknowns, J):
        J['y1','y2'] = -0.2
        J['y1','z'] = numpy.array([[2.0*params['z'][0], 1.0]])
        J['y1','x'] = 1.0


class SellarDis2(ExplicitComponent):

    def __init__(self):
        super(SellarDis2, self).__init__()

        # Global Design Variable
        self.add_input('z', val=numpy.zeros(2))

        # Coupling parameter
        self.add_input('y1', val=1.0)

        # Coupling output
        self.add_output('y2', val=1.0)

        self.execution_count = 0

    def compute(self, params, unknowns):

        z1 = params['z'][0]
        z2 = params['z'][1]
        y1 = params['y1']

        # Note: this may cause some issues. However, y1 is constrained to be
        # above 3.16, so lets just let it converge, and the optimizer will
        # throw it out
        if y1.real < 0.0:
            y1 *= -1

        unknowns['y2'] = y1**.5 + z1 + z2

    def compute_jacobian(self, params, unknowns, J):
        J['y2', 'y1'] = .5*params['y1']**-.5
        J['y2', 'z'] = numpy.array([[1.0, 1.0]])


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):
        prob = Problem()
        model = prob.model = Group()

        model.add_subsystem('d1', SellarDis1(), promotes=['x', 'z', 'y1', 'y2'])
        model.add_subsystem('d2', SellarDis2(), promotes=['z', 'y1', 'y2'])

        model.add_subsystem('px', IndepVarComp('x', val=1.0), promotes=['x'])
        model.add_subsystem('pz', IndepVarComp('z', val=numpy.array([5.0, 2.0])),
                            promotes=['z'])

        # Specify Newton's method for the nonlinear solver.
        model.nl_solver = NewtonSolver()

        # Specify scipy GMRES for the linear solver.
        model.ln_solver = ScipyIterativeSolver()

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['y1'], 25.58830273, .00001)
        assert_rel_error(self, prob['y2'], 12.05848819, .00001)

    def test_specify_subgroup_solvers(self):
        class SellarSubGroup(Group):

            def __init__(self):
                super(SellarSubGroup, self).__init__()

                self.add_subsystem('d1', SellarDis1(),
                                   promotes=['x', 'z', 'y1', 'y2'])
                self.add_subsystem('d2', SellarDis2(),
                                   promotes=['z', 'y1', 'y2'])

                # Each Sellar group uses Newton's method
                self.nl_solver = NewtonSolver()
                self.ln_solver = ScipyIterativeSolver()

        prob = Problem()
        root = prob.model = Group()

        root.add_subsystem('g1', SellarSubGroup())
        root.add_subsystem('g2', SellarSubGroup())

        root.connect('g1.y2', 'g2.x')
        root.connect('g2.y2', 'g1.x')

        # Converge the outer loop with Gauss Seidel, with a looser tolerance.
        root.nl_solver = NonlinearBlockGS()
        root.nl_solver.options['rtol'] = 1.0e-5

        prob.setup()
        prob.run_model()

        assert_rel_error(self, prob['g1.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g1.y2'], 0.80, .00001)
        assert_rel_error(self, prob['g2.y1'], 0.64, .00001)
        assert_rel_error(self, prob['g2.y2'], 0.80, .00001)

if __name__ == "__main__":
    unittest.main()