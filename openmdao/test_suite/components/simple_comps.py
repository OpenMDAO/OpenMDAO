""" Defines a few simple comps that are used in tests."""

import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent


class DoubleArrayComp(ExplicitComponent):
    """A fairly simple array component."""

    def __init__(self):
        super(DoubleArrayComp, self).__init__()

        # Params
        self.add_input('x1', np.zeros([2]))
        self.add_input('x2', np.zeros([2]))

        # Unknowns
        self.add_output('y1', np.zeros([2]))
        self.add_output('y2', np.zeros([2]))

        self.JJ = np.array([[1.0, 3.0, -2.0, 7.0],
                            [6.0, 2.5, 2.0, 4.0],
                            [-1.0, 0.0, 8.0, 1.0],
                            [1.0, 4.0, -5.0, 6.0]])

    def compute(self, inputs, outputs):
        """Execution."""
        outputs['y1'] = self.JJ[0:2, 0:2].dot(inputs['x1']) + \
                         self.JJ[0:2, 2:4].dot(inputs['x2'])
        outputs['y2'] = self.JJ[2:4, 0:2].dot(inputs['x1']) + \
                         self.JJ[2:4, 2:4].dot(inputs['x2'])

    def compute_jacobian(self, inputs, outputs, jacobian):
        """Analytical derivatives."""
        jacobian[('y1', 'x1')] = self.JJ[0:2, 0:2]
        jacobian[('y1', 'x2')] = self.JJ[0:2, 2:4]
        jacobian[('y2', 'x1')] = self.JJ[2:4, 0:2]
        jacobian[('y2', 'x2')] = self.JJ[2:4, 2:4]
