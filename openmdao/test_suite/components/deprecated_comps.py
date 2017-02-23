""" Defines a few simple comps that are used to test deprecations."""

import numpy as np

from openmdao.api import ExplicitComponent


class DeprecatedComp(ExplicitComponent):
    """
    A simple component that adds variables in the __init__ function.
    """

    def __init__(self):
        super(DeprecatedComp, self).__init__()

        self.add_input('x1', np.zeros([2]))
        self.add_output('y1', np.zeros([2]))

    def compute(self, inputs, outputs):
        """
        Execution.
        """
        outputs['y1'] = 2.*inputs['x1']
