from __future__ import division
import numpy
try:
    from petsc4py import PETSc
except:
    pass


class Jacobian(object):

    def __init__(self):
        self._top_name = None
        self._top_system = None
        self._assembler = None
        self._system = None

        self._dict = {}
        self._mtx = {}

    def _process_key(self, key):
        op_name, ip_name = key
        outputs = self._system._outputs
        inputs = self._system._inputs

        op_size = len(outputs[op_name])
        op_ind = self._system._variable_allprocs_indices['output'][op_name]
        if ip_name in inputs:
            ip_size = len(inputs[ip_name])
            ip_ind = self._system._variable_allprocs_indices['input'][ip_name]
        elif ip_name in outputs:
            ip_size = len(outputs[ip_name])
            ip_ind = self._system._variable_allprocs_indices['output'][ip_name]

        return op_ind, ip_ind, op_size, ip_size

    def _negate(self, key):
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)
        jac = self._dict[op_ind, ip_ind]

        if numpy.isscalar(jac):
            self._dict[op_ind, ip_ind] = -jac
        elif type(jac) == numpy.ndarray:
            self._dict[op_ind, ip_ind] = -jac
        elif len(jac) == 3:
            self._dict[op_ind, ip_ind][0] = -self._dict[op_ind, ip_ind][0]

    def __contains__(self, key):
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)
        return (op_ind, ip_ind) in self._dict

    def __setitem__(self, key, jac):
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)

        if numpy.isscalar(jac):
            jac = numpy.array([jac]).reshape((op_size, ip_size))
        elif type(jac) is list:
            jac = numpy.array(jac).reshape((op_size, ip_size))

        self._dict[op_ind, ip_ind] = jac

    def __getitem__(self, key):
        op_ind, ip_ind, op_size, ip_size = self._process_key(key)
        return self._dict[op_ind, ip_ind]


class DefaultJacobian(Jacobian):

    def _initialize(self):
        pass

    def _update(self):
        pass

    def _apply(self, d_inputs, d_outputs, d_residuals, op_names, ip_names,
               mode, var_ind_range):
        if mode == 'fwd':
            d_residuals.set_const(0.0)
        elif mode == 'rev':
            d_inputs.set_const(0.0)
            d_outputs.set_const(0.0)

        for (op_name, ip_name) in self._dict:
            jac = self._dict[op_name, ip_name]

            if op_name in op_names and ip_name in op_names:
                if mode == 'fwd':
                    d_residuals[op_name] += jac.dot(d_outputs[ip_name])
                if mode == 'rev':
                    d_outputs[ip_name] = jac.T.dot(d_residuals[op_name])

            if op_name in op_names and ip_name in ip_names:
                if mode == 'fwd':
                    d_residuals[op_name] += jac.dot(d_inputs[ip_name])
                if mode == 'rev':
                    d_inputs[ip_name] += jac.T.dot(d_residuals[op_name])


class DenseJacobian(object):

    def _initialize(self):
        sizes = self._assembler.variable_sizes
        set_indices = self._assembler.variable_set_indices
        iproc = self._system._comm.rank + self._system._proc_range[0]

        ip_nvar_set = len(self._assembler.variable_set_IDs['input'])
        op_nvar_set = len(self._assembler.variable_set_IDs['output'])

        for ip_ivar_set in xrange(ip_nvar_set):
            ip_bool = set_indices['input'][:, 0] == ip_ivar_set
            ip_inds = set_indices['input'][ip_bool, 1]
            if len(ip_inds) > 0:
                sizes_array = sizes['input'][ip_ivar_set]
                ind1 = numpy.sum(sizes_array[iproc, :ip_inds[0]])
                ind2 = numpy.sum(sizes_array[iproc, :ip_inds[-1]+1])
                ip_size = ind2 - ind1
            else:
                ip_size = 0

            for op_ivar_set in xrange(op_nvar_set):
                op_bool = set_indices['output'][:, 0] == op_ivar_set
                op_inds = set_indices['output'][op_bool, 1]
                if len(op_inds) > 0:
                    sizes_array = sizes['output'][op_ivar_set]
                    ind1 = numpy.sum(sizes_array[oproc, :op_inds[0]])
                    ind2 = numpy.sum(sizes_array[oproc, :op_inds[-1]+1])
                    op_size = ind2 - ind1
                else:
                    op_size = 0

                if ip_size > 0 and op_size > 0:
                    array = numpy.zeros((op_size, ip_size))
                    self._mtx[op_ivar_set, ip_ivar_set] = array

    def _update(self):
        names = self._system.variable_myproc_names
        indices = self._system.variable_myproc_indices
        sizes = self._assembler.variable_sizes
        set_indices = self._assembler.variable_set_indices
        iproc = self._system._comm.rank + self._system._proc_range[0]

        for ip_ind in xrange(len(names['input'])):
            ip_name = names['input'][ip_ind]
            ip_ivar_all = indices['input'][ip_ind]
            ip_ivar_set, ip_ivar = set_indices[ip_ivar_all, :]
            sizes_array = sizes['input'][ip_ivar_set]
            ip_ind1 = numpy.sum(sizes_array[iproc, :ip_ivar])
            ip_ind2 = numpy.sum(sizes_array[iproc, :ip_ivar+1])

            for op_ind in xrange(len(names['output'])):
                op_name = names['output'][op_ind]
                op_ivar_all = indices['output'][op_ind]
                op_ivar_set, op_ivar = set_indices[op_ivar_all, :]
                sizes_array = sizes['output'][op_ivar_set]
                ip_ind1 = numpy.sum(sizes_array[iproc, :op_ivar])
                ip_ind2 = numpy.sum(sizes_array[iproc, :op_ivar+1])

                if (op_name, ip_name) in self._dict:
                    jac = self._dict[op_name, ip_name]
                    mtx = self._mtx[op_ivar_set, ip_ivar_set]
                    mtx[op_ind1:op_ind2, ip_ind1:ip_ind2] = jac
