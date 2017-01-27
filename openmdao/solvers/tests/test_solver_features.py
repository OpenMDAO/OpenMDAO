"""Test the code we put in out main dolver featuer document."""

import unittest

from openmdao.core.explicitcomponent import ExplicitComponent


# Note, we are inclduing "clean" versions of the Sellar disciplines for
# showcasing in the feature doc.

class SellarDis1(ExplicitComponent):
    """Component containing Discipline 1."""

    def __init__(self):
        super(SellarDis1, self).__init__()

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

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
        J['y1','z'] = np.array([[2.0*params['z'][0], 1.0]])
        J['y1','x'] = 1.0


class SellarDis2(ExplicitComponent):
    """Component containing Discipline 2."""

    def __init__(self):
        super(SellarDis2, self).__init__()

        # Global Design Variable
        self.add_input('z', val=np.zeros(2))

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
        J['y2', 'z'] = np.array([[1.0, 1.0]])


class TestSolverFeatures(unittest.TestCase):

    def test_specify_solver(self):
        pass

if __name__ == "__main__":
    unittest.main()