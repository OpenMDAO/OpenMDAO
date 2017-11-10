"""
Component for a metadata feature test.
"""
import numpy as np
from types import FunctionType

from openmdao.api import ExplicitComponent


class UnitaryFunctionComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('func', values=('exp', 'cos', 'sin'), types=FunctionType)

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('y', 'x', method='fd')

    def compute(self, inputs, outputs):
        func = self.metadata['func']

        if func == 'exp':
            outputs['y'] = np.exp(inputs['x'])
        elif func == 'cos':
            outputs['y'] = np.cos(inputs['x'])
        elif func == 'sin':
            outputs['y'] = np.sin(inputs['x'])
        else:
            outputs['y'] = func(inputs['x'])
