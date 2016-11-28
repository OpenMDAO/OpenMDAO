"""Define the explicit test component (simple)."""
from __future__ import division, print_function

import numpy
import scipy.sparse

from openmdao.api import ExplicitComponent


class TestExplCompSimple(ExplicitComponent):

    def initialize_variables(self):
        self.add_input('length', val=1.)
        self.add_input('width', val=1.)
        self.add_output('area', val=1.)

    def compute(self, inputs, outputs):
        outputs['area'] = inputs['length'] * inputs['width']


class TestExplCompSimpleDense(TestExplCompSimple):

    def compute_jacobian(self, inputs, outputs, jacobian):
        jacobian['area', 'length'] = inputs['width']
        jacobian['area', 'width'] = inputs['length']


class TestExplCompSimpleSpmtx(TestExplCompSimple):

    def compute_jacobian(self, inputs, outputs, jacobian):
        jacobian['area', 'length'] = scipy.sparse.csr_matrix(
            (inputs['width'], (0, 0)))
        jacobian['area', 'width'] = scipy.sparse.csr_matrix(
            (inputs['length'], (0, 0)))


class TestExplCompSimpleSparse(TestExplCompSimple):

    def compute_jacobian(self, inputs, outputs, jacobian):
        jacobian['area', 'length'] = (inputs['width'], 0, 0)
        jacobian['area', 'width'] = (inputs['length'], 0, 0)
