""" Defines a few simple comps that are used in tests."""

import numpy as np

import openmdao.api as om


class DoubleArrayComp(om.ExplicitComponent):
    """
    A fairly simple array component.
    """

    def __init__(self):
        super().__init__()

        self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                            [6.0, 2.5, 2.0, 4.0],
                            [-1.0, 0.0, 8.0, 1.0],
                            [1.0, 4.0, -5.0, 6.0]])

    def setup(self):
        # Params
        self.add_input('x1', np.zeros([2]))
        self.add_input('x2', np.zeros([2]))

        # Unknowns
        self.add_output('y1', np.zeros([2]))
        self.add_output('y2', np.zeros([2]))

    def setup_partials(self):
        # Derivs
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        """
        Execution.
        """
        outputs['y1'] = self.JJ[0:2, 0:2].dot(inputs['x1']) + \
                        self.JJ[0:2, 2:4].dot(inputs['x2'])
        outputs['y2'] = self.JJ[2:4, 0:2].dot(inputs['x1']) + \
                        self.JJ[2:4, 2:4].dot(inputs['x2'])

    def compute_partials(self, inputs, partials):
        """
        Analytical derivatives.
        """
        partials[('y1', 'x1')] = self.JJ[0:2, 0:2]
        partials[('y1', 'x2')] = self.JJ[0:2, 2:4]
        partials[('y2', 'x1')] = self.JJ[2:4, 0:2]
        partials[('y2', 'x2')] = self.JJ[2:4, 2:4]


class NonSquareArrayComp(om.ExplicitComponent):
    """
    A fairly simple array component.
    """

    def __init__(self):
        super().__init__()

        self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                            [6.0, 2.5, 2.0, 4.0],
                            [-1.0, 0.0, 8.0, 1.0],
                            [1.0, 4.0, -5.0, 6.0]])

    def setup(self):
        # Params
        self.add_input('x1', np.zeros([2]))
        self.add_input('x2', np.zeros([2]))

        # Unknowns
        self.add_output('y1', np.zeros([3]))
        self.add_output('y2', np.zeros([1]))

    def setup_partials(self):
        # Derivs
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        """
        Execution.
        """
        outputs['y1'] = self.JJ[0:3, 0:2].dot(inputs['x1']) + \
                         self.JJ[0:3, 2:4].dot(inputs['x2'])
        outputs['y2'] = self.JJ[3:4, 0:2].dot(inputs['x1']) + \
                         self.JJ[3:4, 2:4].dot(inputs['x2'])

    def compute_partials(self, inputs, partials):
        """
        Analytical derivatives.
        """
        partials[('y1', 'x1')] = self.JJ[0:3, 0:2]
        partials[('y1', 'x2')] = self.JJ[0:3, 2:4]
        partials[('y2', 'x1')] = self.JJ[3:4, 0:2]
        partials[('y2', 'x2')] = self.JJ[3:4, 2:4]
