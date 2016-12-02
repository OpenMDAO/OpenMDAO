"""Define the test implicit component classes."""
from __future__ import division, print_function
import numpy
import scipy.sparse

from six import iteritems
from six.moves import range

from openmdao.api import ImplicitComponent


class TestImplCompNondLinear(ImplicitComponent):
    """Test implicit component, non-distributed, linear."""

    def initialize(self):
        self.metadata.declare('num_input', typ=int, value=1,
                              desc='number of input variables to declare')
        self.metadata.declare('num_output', typ=int, value=1,
                              desc='number of output variables to declare')
        self.metadata.declare('var_shape', value=(1,),
                              desc='input/output variable shapes')

    def initialize_variables(self):
        var_shape = self.metadata['var_shape']
        size = numpy.prod(var_shape)

        self.metadata['ip_names'] = ['input_%i' % ip_ind for ip_ind in
                                     range(self.metadata['num_input'])]
        self.metadata['op_names'] = ['output_%i' % op_ind for op_ind in
                                     range(self.metadata['num_output'])]

        for ip_name in self.metadata['ip_names']:
            self.add_input(ip_name, shape=var_shape,
                           indices=numpy.arange(size))

        for op_name in self.metadata['op_names']:
            self.add_output(op_name, shape=var_shape)

        self.coeffs = {}
        self.rhs_coeffs = {}

        for re_name in self.metadata['op_names']:
            mtx = numpy.ones(size)
            self.rhs_coeffs[re_name] = mtx
            for op_name in self.metadata['op_names']:
                mtx = numpy.ones((size, size)) * 0.01
                if re_name == op_name:
                    numpy.fill_diagonal(mtx, 1)
                self.coeffs[re_name, op_name] = mtx
            for ip_name in self.metadata['ip_names']:
                mtx = numpy.ones((size, size)) * 0.01
                self.coeffs[re_name, ip_name] = mtx

    def apply_nonlinear(self, inputs, outputs, residuals):
        for re_name in self.metadata['op_names']:
            re = residuals._views_flat[re_name]
            re[:] = -self.rhs_coeffs[re_name]
            for op_name in self.metadata['op_names']:
                mtx = self.coeffs[re_name, op_name]
                op = outputs._views_flat[op_name]
                re += mtx.dot(op)
            for ip_name in self.metadata['ip_names']:
                mtx = self.coeffs[re_name, ip_name]
                ip = inputs._views_flat[ip_name]
                re += mtx.dot(ip)

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if self.metadata['jacobian_type'] == 'matvec':
            if mode == 'fwd':
                for re_name in d_residuals:
                    d_re = d_residuals._views_flat[re_name]
                    for op_name in d_outputs:
                        mtx = self.coeffs[re_name, op_name]
                        d_op = d_outputs._views_flat[op_name]
                        d_re += mtx.dot(d_op)
                    for ip_name in d_inputs:
                        mtx = self.coeffs[re_name, ip_name]
                        d_ip = d_inputs._views_flat[ip_name]
                        d_re += mtx.dot(d_ip)
            elif mode == 'rev':
                for re_name in d_residuals:
                    d_re = d_residuals._views_flat[re_name]
                    for op_name in d_outputs:
                        mtx = self.coeffs[re_name, op_name]
                        d_op = d_outputs._views_flat[op_name]
                        d_op += mtx.T.dot(d_re)
                    for ip_name in d_inputs:
                        mtx = self.coeffs[re_name, ip_name]
                        d_ip = d_inputs._views_flat[ip_name]
                        d_ip += mtx.T.dot(d_re)

    def linearize(self, inputs, outputs, jacobian):
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
            for re_name in self.metadata['op_names']:
                for op_name in self.metadata['op_names']:
                    jacobian[re_name, op_name] = get_jac((re_name, op_name))
                for ip_name in self.metadata['ip_names']:
                    jacobian[re_name, ip_name] = get_jac((re_name, ip_name))
