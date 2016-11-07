"""Define the test implicit component classes."""
from __future__ import division, print_function
import numpy

from six import iteritems
from six.moves import range

from openmdao.api import ImplicitComponent


class TestImplCompNondLinear(ImplicitComponent):
    """Test implicit component, non-distributed, linear."""

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
            self.add_input(ip_name)

        for op_name in self.metadata['op_names']:
            self.add_output(op_name, shape=var_shape)

        self.coeffs = {}

        for re_name in self.metadata['op_names']:
            for op_name in self.metadata['op_names']:
                mtx = numpy.random.rand(size, size)
                numpy.fill_diagonal(mtx, 10)
                self.coeffs[re_name, op_name] = mtx
            for ip_name in self.metadata['ip_names']:
                mtx = numpy.random.rand(size, size)
                self.coeffs[re_name, ip_name] = mtx

    def apply_nonlinear(self, inputs, outputs, residuals):
        for re_name in self.metadata['op_names']:
            residuals[re_name] = 0.0
            for op_name in self.metadata['op_names']:
                mtx = self.coeffs[re_name, op_name]
                re = residuals._views[re_name].flatten()
                op = outputs[op_name].flatten()
                re += mtx.dot(op)
            for ip_name in self.metadata['ip_names']:
                mtx = self.coeffs[re_name, ip_name]
                re = residuals._views[re_name].flatten()
                ip = inputs[ip_name].flatten()
                re += mtx.dot(ip)

    def apply_linear(self, inputs, outputs,
                     d_inputs, d_outputs, d_residuals, mode):
        if mode == 'fwd':
            for re_name in d_residuals:
                for op_name in d_outputs:
                    mtx = self.coeffs[re_name, op_name]
                    d_re = d_residuals._views[re_name].flatten()
                    d_op = d_outputs[op_name].flatten()
                    d_re += mtx.dot(d_op)
                for ip_name in d_inputs:
                    mtx = self.coeffs[re_name, ip_name]
                    d_re = d_residuals._views[re_name].flatten()
                    d_ip = d_inputs[ip_name].flatten()
                    d_re += mtx.dot(d_ip)
        if mode == 'fwd':
            for re_name in d_residuals:
                for op_name in d_outputs:
                    mtx = self.coeffs[re_name, op_name]
                    d_re = d_residuals[re_name].flatten()
                    d_op = d_outputs._views[op_name].flatten()
                    d_op += mtx.T.dot(d_re)
                for ip_name in d_inputs:
                    mtx = self.coeffs[re_name, ip_name]
                    d_re = d_residuals[re_name].flatten()
                    d_ip = d_inputs._views[ip_name].flatten()
                    d_ip += mtx.T.dot(d_re)
