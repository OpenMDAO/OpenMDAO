"""Define the explicit test component (array)."""
from __future__ import division, print_function

import numpy
import scipy.sparse

from openmdao.api import ExplicitComponent


class TestExplCompArray(ExplicitComponent):

    def initialize(self):
        self.metadata.declare('thickness', value=1.)

    def initialize_variables(self):
        self.add_input('lengths', val=numpy.ones((2, 2)))
        self.add_input('widths', val=numpy.ones((2, 2)))
        self.add_output('areas', val=numpy.ones((2, 2)))
        self.add_output('total_volume', val=1.)

    def compute(self, inputs, outputs):
        thk = self.metadata['thickness']

        outputs['areas'] = inputs['lengths'] * inputs['widths']
        outputs['total_volume'] = numpy.sum(outputs['areas']) * thk


class TestExplCompArrayDense(TestExplCompArray):

    def compute_partials(self, inputs, outputs, partials):
        thk = self.metadata['thickness']

        partials['areas', 'lengths'] = numpy.diag(inputs['widths'].flatten())
        partials['areas', 'widths'] = numpy.diag(inputs['lengths'].flatten())
        partials['total_volume', 'lengths'] = inputs['widths'].flatten() * thk
        partials['total_volume', 'widths'] = inputs['lengths'].flatten() * thk


class TestExplCompArraySpmtx(TestExplCompArray):

    def compute_partials(self, inputs, outputs, partials):
        thk = self.metadata['thickness']

        inds = numpy.arange(4)
        partials['areas', 'lengths'] = scipy.sparse.csr_matrix(
            (inputs['widths'].flatten(), (inds, inds)))
        partials['areas', 'widths'] = scipy.sparse.csr_matrix(
            (inputs['lengths'].flatten(), (inds, inds)))
        partials['total_volume', 'lengths'] = scipy.sparse.csr_matrix(
            (inputs['widths'].flatten() * thk, ([0], inds)))
        partials['total_volume', 'widths'] = scipy.sparse.csr_matrix(
            (inputs['lengths'].flatten() * thk, ([0], inds)))


class TestExplCompArraySparse(TestExplCompArray):

    def compute_partials(self, inputs, outputs, partials):
        thk = self.metadata['thickness']

        inds = numpy.arange(4)
        partials['areas', 'lengths'] = (inputs['widths'].flatten(), inds, inds)
        partials['areas', 'widths'] = (inputs['lengths'].flatten(), inds, inds)
        partials['total_volume', 'lengths'] = (
            inputs['widths'].flatten() * thk, [0], inds)
        partials['total_volume', 'widths'] = (
            inputs['lengths'].flatten() * thk, [0], inds)
