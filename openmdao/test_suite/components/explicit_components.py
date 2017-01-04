"""Define the test explicit component classes."""
from __future__ import division, print_function
import numpy
import scipy.sparse

from six import iteritems
from six.moves import range

from openmdao.api import ExplicitComponent


class TestExplCompNondLinear(ExplicitComponent):
    """Test explicit component, non-distributed, linear."""

    def initialize(self):
        self.metadata.declare('num_input', type_=int, value=1,
                              desc='number of input variables to declare')
        self.metadata.declare('num_output', type_=int, value=1,
                              desc='number of output variables to declare')
        self.metadata.declare('var_shape', value=(1,),
                              desc='input/output variable shapes')

    def initialize_variables(self):
        var_shape = self.metadata['var_shape']
        size = numpy.prod(var_shape)

        self.metadata['in_names'] = ['input_%i' % in_ind for in_ind in
                                     range(self.metadata['num_input'])]
        self.metadata['out_names'] = ['output_%i' % out_ind for out_ind in
                                     range(self.metadata['num_output'])]

        for in_name in self.metadata['in_names']:
            self.add_input(in_name, shape=var_shape,
                           indices=numpy.arange(size))

        for out_name in self.metadata['out_names']:
            self.add_output(out_name, shape=var_shape)

        self.coeffs = {}
        self.rhs_coeffs = {}

        for out_name in self.metadata['out_names']:
            mtx = numpy.ones(size)
            self.rhs_coeffs[out_name] = mtx
            for in_name in self.metadata['in_names']:
                mtx = numpy.ones((size, size)) * 0.01
                self.coeffs[out_name, in_name] = mtx

    def compute(self, inputs, outputs):
        for out_name in self.metadata['out_names']:
            op = outputs._views_flat[out_name]
            op[:] = -self.rhs_coeffs[out_name]
            for in_name in self.metadata['in_names']:
                mtx = self.coeffs[out_name, in_name]
                ip = inputs._views_flat[in_name]
                op += mtx.dot(ip)

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        if self.metadata['jacobian_type'] == 'matvec':
            if mode == 'fwd':
                for out_name in d_outputs:
                    d_op = d_outputs._views_flat[out_name]
                    for in_name in d_inputs:
                        mtx = self.coeffs[out_name, in_name]
                        d_ip = d_inputs._views_flat[in_name]
                        d_op += mtx.dot(d_ip)
            elif mode == 'rev':
                for out_name in d_outputs:
                    d_op = d_outputs._views_flat[out_name]
                    for in_name in d_inputs:
                        mtx = self.coeffs[out_name, in_name]
                        d_ip = d_inputs._views_flat[in_name]
                        d_ip += mtx.T.dot(d_op)

    def compute_jacobian(self, inputs, outputs, jacobian):
        def get_jac(key):
            if self.metadata['partial_type'] == 'array':
                jac = self.coeffs[key]
            elif self.metadata['partial_type'] == 'sparse':
                jac = scipy.sparse.csr_matrix(self.coeffs[key])
            elif self.metadata['partial_type'] == 'aij':
                shape = self.coeffs[key].shape
                irows = numpy.zeros(shape, int)
                icols = numpy.zeros(shape, int)
                for indr in range(shape[0]):
                    for indc in range(shape[1]):
                        irows[indr, indc] = indr
                        icols[indr, indc] = indc
                data = self.coeffs[key].flatten()
                rows = irows.flatten()
                cols = icols.flatten()
                jac = (data, rows, cols)
            return jac

        if self.metadata['jacobian_type'] != 'matvec':
            coeffs = self.coeffs
            for out_name in self.metadata['out_names']:
                for in_name in self.metadata['in_names']:
                    jacobian[out_name, in_name] = get_jac((out_name, in_name))
