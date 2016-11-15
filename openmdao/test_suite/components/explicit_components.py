"""Define the test explicit component classes."""
from __future__ import division, print_function
import numpy

from six import iteritems
from six.moves import range

from openmdao.api import ExplicitComponent


class TestExplCompNondLinear(ExplicitComponent):
    """Test explicit component, non-distributed, linear."""

    def initialize(self):
        self.metadata.declare('num_input', typ=int, value=1)
        self.metadata.declare('num_output', typ=int, value=1)
        self.metadata.declare('var_shape', value=(1,))

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

        for op_name in self.metadata['op_names']:
            mtx = numpy.ones(size)
            self.rhs_coeffs[op_name] = mtx
            for ip_name in self.metadata['ip_names']:
                mtx = numpy.ones((size, size)) * 0.01
                self.coeffs[op_name, ip_name] = mtx

    def compute(self, inputs, outputs):
        var_shape = self.metadata['var_shape']
        size = numpy.prod(var_shape)

        for op_name in self.metadata['op_names']:
            op = outputs._views[op_name].reshape(size)
            op[:] = -self.rhs_coeffs[op_name]
            for ip_name in self.metadata['ip_names']:
                mtx = self.coeffs[op_name, ip_name]
                ip = inputs._views[ip_name].reshape(size)
                op += mtx.dot(ip)

    def compute_jacvec_product(self, inputs, outputs,
                               d_inputs, d_outputs, mode):
        var_shape = self.metadata['var_shape']
        size = numpy.prod(var_shape)

        if mode == 'fwd':
            for op_name in d_outputs:
                d_op = d_outputs._views[op_name].reshape(size)
                for ip_name in d_inputs:
                    mtx = self.coeffs[op_name, ip_name]
                    d_ip = d_inputs._views[ip_name].reshape(size)
                    d_op += mtx.dot(d_ip)
        elif mode == 'rev':
            for op_name in d_outputs:
                d_op = d_outputs._views[op_name].reshape(size)
                for ip_name in d_inputs:
                    mtx = self.coeffs[op_name, ip_name]
                    d_ip = d_inputs._views[ip_name].reshape(size)
                    d_ip += mtx.T.dot(d_op)
