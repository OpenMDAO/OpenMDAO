"""
A component that computes y = func(x), where func
is a function given as metadata.
"""

from types import FunctionType

from openmdao.api import ExplicitComponent

class UnitaryFunctionComp(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('func', types=FunctionType)

    def setup(self):
        self.add_input('x')
        self.add_output('y')
        self.declare_partials('y', 'x', method='fd')

    def compute(self, inputs, outputs):
        func = self.metadata['func']
        outputs['y'] = func(inputs['x'])
