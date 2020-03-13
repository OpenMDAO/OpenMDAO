"""Define the explicit test component (array)."""
import numpy as np
import scipy.sparse

import openmdao.api as om


class TestExplCompArray(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('thickness', default=1.)

    def setup(self):
        self.add_input('lengths', val=np.ones((2, 2)))
        self.add_input('widths', val=np.ones((2, 2)))
        self.add_output('areas', val=np.ones((2, 2)))
        self.add_output('total_volume', val=1.)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        thk = self.options['thickness']

        outputs['areas'] = inputs['lengths'] * inputs['widths']
        outputs['total_volume'] = np.sum(outputs['areas']) * thk


class TestExplCompArrayDense(TestExplCompArray):

    def compute_partials(self, inputs, partials):
        thk = self.options['thickness']

        partials['areas', 'lengths'] = np.diag(inputs['widths'].flatten())
        partials['areas', 'widths'] = np.diag(inputs['lengths'].flatten())
        partials['total_volume', 'lengths'] = inputs['widths'].flatten() * thk
        partials['total_volume', 'widths'] = inputs['lengths'].flatten() * thk


class TestExplCompArraySpmtx(TestExplCompArray):

    def compute_partials(self, inputs, partials):
        thk = self.options['thickness']

        inds = np.arange(4)
        partials['areas', 'lengths'] = scipy.sparse.csr_matrix(
            (inputs['widths'].flatten(), (inds, inds)))
        partials['areas', 'widths'] = scipy.sparse.csr_matrix(
            (inputs['lengths'].flatten(), (inds, inds)))
        partials['total_volume', 'lengths'] = scipy.sparse.csr_matrix(
            (inputs['widths'].flatten() * thk, ([0], inds)))
        partials['total_volume', 'widths'] = scipy.sparse.csr_matrix(
            (inputs['lengths'].flatten() * thk, ([0], inds)))


class TestExplCompArraySparse(TestExplCompArray):

    def compute_partials(self, inputs, partials):
        thk = self.options['thickness']

        inds = np.arange(4)
        partials['areas', 'lengths'] = (inputs['widths'].flatten(), inds, inds)
        partials['areas', 'widths'] = (inputs['lengths'].flatten(), inds, inds)
        partials['total_volume', 'lengths'] = (
            inputs['widths'].flatten() * thk, [0], inds)
        partials['total_volume', 'widths'] = (
            inputs['lengths'].flatten() * thk, [0], inds)
