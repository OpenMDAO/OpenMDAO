from __future__ import division
import numpy



class Assembler(object):

    def __init__(self, comm):
        """ Initializes all attributes """
        self.comm = comm

        self._variable_sizes = {'input': [], 'output': []}
        self._variable_set_IDs = {'input': {}, 'output': {}}
        self._variable_set_indices = {'input': None, 'output': None}

        self._input_IDs = None
        self._input_indices = None
        self._input_indices_meta = None



class DefaultAssembler(Assembler):

    def _setup_variables(self, sizes, variable_metadata, variable_indices):
        nproc = self.comm.size

        for typ in ['input', 'output']:
            nvar = len(variable_metadata[typ])
            nvar_all = sizes[typ]

            # Locally determine var_set for each var
            local_set_dict = {}
            for ivar in xrange(nvar):
                var = variable_metadata[typ][ivar]
                ivar_all = variable_indices[typ][ivar]
                local_set_dict[ivar_all] = var['var_set']

            # Broadcast ivar_all-iset pairs to all procs
            local_set_dicts_list = self.comm.allgather(local_set_dict)
            global_set_dict = {}
            for local_set_dict in local_set_dicts_list:
                global_set_dict.update(local_set_dict)

            # Compute set_name to ID maps
            unique_list = list(set(global_set_dict.values()))
            for set_name in unique_list:
                if set_name not in self._variable_set_IDs[typ]:
                    nset = len(self._variable_set_IDs[typ])
                    self._variable_set_IDs[typ][set_name] = nset

            # Compute _variable_set_indices and var_count
            var_count = numpy.zeros(len(self._variable_set_IDs[typ]), int)
            self._variable_set_indices[typ] = -numpy.ones((nvar_all, 2), int)
            for ivar_all in global_set_dict:
                set_name = global_set_dict[ivar_all]

                iset = self._variable_set_IDs[typ][set_name]
                ivar_set = var_count[iset]

                self._variable_set_indices[typ][ivar_all, 0] = iset
                self._variable_set_indices[typ][ivar_all, 1] = ivar_set

                var_count[iset] += 1

            # Allocate the size arrays using var_count
            self._variable_sizes[typ] = []
            for iset in xrange(len(self._variable_set_IDs[typ])):
                size = var_count[iset]
                array = numpy.zeros((nproc, size), int)
                self._variable_sizes[typ].append(array)

        # Populate the sizes arrays
        iproc = self.comm.rank
        typ = 'input'
        nvar = len(variable_metadata[typ])
        for ivar in xrange(nvar):
            var = variable_metadata[typ][ivar]
            size = numpy.prod(var['indices'].shape)
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self._variable_set_indices[typ][ivar_all, :]
            self._variable_sizes[typ][iset][iproc, ivar_set] = size
        typ = 'output'
        nvar = len(variable_metadata[typ])
        for ivar in xrange(nvar):
            var = variable_metadata[typ][ivar]
            size = numpy.prod(var['shape'])
            ivar_all = variable_indices[typ][ivar]
            iset, ivar_set = self._variable_set_indices[typ][ivar_all, :]
            self._variable_sizes[typ][iset][iproc, ivar_set] = size

        # Do an allgather on the sizes arrays
        if self.comm.size > 1:
            for typ in ['input', 'output']:
                nset = len(self._variable_sizes[typ])
                for iset in xrange(nset):
                    array = self._variable_sizes[typ][iset]
                    self.comm.Allgather(array[iproc, :], array)

    def _setup_connections(self, connections, _variable_allprocs_names):
        """ Identifies implicit connections, combines with explicit ones """
        nvar_input = len(_variable_allprocs_names['input'])
        _input_IDs = -numpy.ones(nvar_input, int)

        # Add explicit connections to the _input_IDs vector
        for ip_ID, op_ID in connections:
            _input_IDs[ip_ID] = op_ID

        # Loop over input variables
        for ip_ID in xrange(nvar_input):
            name = _variable_allprocs_names['input'][ip_ID]

            # If name is also an output variable, add this implicit connection
            if name in _variable_allprocs_names['output']:
                op_ID = _variable_allprocs_names['output'].index(name)
                _input_IDs[ip_ID] = op_ID

        self._input_IDs = _input_IDs

    def _setup__input_indices(self, input_metadata, var_indices):
        """ Assemble global list of input indices """
        # Compute total size of indices vector
        counter = 0
        for ind in xrange(len(input_metadata)):
            metadata = input_metadata[ind]
            counter += numpy.prod(metadata['indices'].shape)

        # Allocate arrays
        self._input_indices_meta = numpy.zeros((var_indices.shape[0], 2), int)
        self._input_indices = numpy.zeros(counter, int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind in xrange(len(input_metadata)):
            metadata = input_metadata[ind]
            ind2 += numpy.prod(metadata['indices'].shape)
            self._input_indices[ind1:ind2] = metadata['indices'].flatten()
            ivar_all = var_indices[ind]
            self._input_indices_meta[ivar_all, :] = [ind1, ind2]
            ind1 += numpy.prod(metadata['indices'].shape)

    def _compute_transfers(self, nsub_allprocs, var_range,
                          _subsystems_myproc, _subsystems_inds):
        ip_set_indices = self._variable_set_indices['input']
        op_set_indices = self._variable_set_indices['output']

        ip_ind1, ip_ind2 = var_range['input']
        op_ind1, op_ind2 = var_range['output']
        ip_isub_var = -numpy.ones(ip_ind2 - ip_ind1, int)
        op_isub_var = -numpy.ones(op_ind2 - op_ind1, int)
        for ind in xrange(len(_subsystems_myproc)):
            subsys = _subsystems_myproc[ind]
            isub = _subsystems_inds[ind]

            sub_var_range = subsys._variable_allprocs_range
            sub_ip_ind1, sub_ip_ind2 = sub_var_range['input']
            sub_op_ind1, sub_op_ind2 = sub_var_range['output']
            for ip_ind in xrange(ip_ind1, ip_ind2):
                if sub_ip_ind1 <= ip_ind < sub_ip_ind2:
                    ip_isub_var[ip_ind - ip_ind1] = isub
            for op_ind in xrange(op_ind1, op_ind2):
                if sub_op_ind1 <= op_ind < sub_op_ind2:
                    op_isub_var[op_ind - op_ind1] = isub

        xfer_ip_inds = {}
        xfer_op_inds = {}
        fwd_xfer_ip_inds = [{} for sub_ind in xrange(nsub_allprocs)]
        fwd_xfer_op_inds = [{} for sub_ind in xrange(nsub_allprocs)]
        rev_xfer_ip_inds = [{} for sub_ind in xrange(nsub_allprocs)]
        rev_xfer_op_inds = [{} for sub_ind in xrange(nsub_allprocs)]
        for iset in xrange(len(self._variable_sizes['input'])):
            for jset in xrange(len(self._variable_sizes['output'])):
                xfer_ip_inds[iset, jset] = []
                xfer_op_inds[iset, jset] = []
                for sub_ind in xrange(nsub_allprocs):
                    fwd_xfer_ip_inds[sub_ind][iset, jset] = []
                    fwd_xfer_op_inds[sub_ind][iset, jset] = []
                    rev_xfer_ip_inds[sub_ind][iset, jset] = []
                    rev_xfer_op_inds[sub_ind][iset, jset] = []

        ip_ind1, ip_ind2 = var_range['input']
        op_ind1, op_ind2 = var_range['output']
        for ip_ind in xrange(ip_ind1, ip_ind2):
            op_ind = self._input_IDs[ip_ind]
            if op_ind1 <= op_ind < op_ind2:

                ip_isub = ip_isub_var[ip_ind - ip_ind1]
                op_isub = op_isub_var[op_ind - op_ind1]

                if ip_isub != -1 and ip_isub != op_isub:
                    ip_iset, ip_ivar_set = ip_set_indices[ip_ind, :]
                    op_iset, op_ivar_set = op_set_indices[op_ind, :]

                    ip_sizes = self._variable_sizes['input'][ip_iset]
                    op_sizes = self._variable_sizes['output'][op_iset]

                    ind1, ind2 = self._input_indices_meta[ip_ivar_set, :]
                    inds = self._input_indices[ind1:ind2]

                    output_inds = numpy.zeros(inds.shape[0], int)
                    ind1, ind2 = 0, 0
                    for iproc in xrange(self.comm.size):
                        ind2 += op_sizes[iproc, op_ivar_set]

                        on_iproc = numpy.logical_and(ind1 <= inds,
                                                     inds <  ind2)
                        offset = -ind1
                        offset += numpy.sum(op_sizes[:iproc, :])
                        offset += numpy.sum(op_sizes[iproc, :op_ivar_set])
                        output_inds[on_iproc] = inds[on_iproc] + offset

                        ind1 += op_sizes[iproc, op_ivar_set]

                    iproc = self.comm.rank

                    ind1 = numpy.sum(ip_sizes[:iproc, :])
                    ind1 += numpy.sum(ip_sizes[iproc, :ip_ivar_set])
                    ind2 = numpy.sum(ip_sizes[:iproc, :])
                    ind2 += numpy.sum(ip_sizes[iproc, :ip_ivar_set+1])
                    input_inds = numpy.arange(ind1, ind2)

                    xfer_ip_inds[ip_iset, op_iset].append(input_inds)
                    xfer_op_inds[ip_iset, op_iset].append(output_inds)

                    # rev mode wouldn't work for GS with a parallel group
                    if op_isub != -1:
                        key = (ip_iset, op_iset)
                        fwd_xfer_ip_inds[ip_isub][key].append(input_inds)
                        fwd_xfer_op_inds[ip_isub][key].append(output_inds)
                        rev_xfer_ip_inds[op_isub][key].append(input_inds)
                        rev_xfer_op_inds[op_isub][key].append(output_inds)

        def merge(indices_list):
            if len(indices_list) > 0:
                return numpy.concatenate(indices_list)
            else:
                return numpy.array([], int)

        for iset in xrange(len(self._variable_sizes['input'])):
            for jset in xrange(len(self._variable_sizes['output'])):
                xfer_ip_inds[iset, jset] = merge(xfer_ip_inds[iset, jset])
                xfer_op_inds[iset, jset] = merge(xfer_op_inds[iset, jset])
                for sub_ind in xrange(nsub_allprocs):
                    fwd_xfer_ip_inds[sub_ind][iset, jset] = \
                        merge(fwd_xfer_ip_inds[sub_ind][iset, jset])
                    fwd_xfer_op_inds[sub_ind][iset, jset] = \
                        merge(fwd_xfer_op_inds[sub_ind][iset, jset])
                    rev_xfer_ip_inds[sub_ind][iset, jset] = \
                        merge(rev_xfer_ip_inds[sub_ind][iset, jset])
                    rev_xfer_op_inds[sub_ind][iset, jset] = \
                        merge(rev_xfer_op_inds[sub_ind][iset, jset])

        xfer_indices = [xfer_ip_inds, xfer_op_inds,
                        fwd_xfer_ip_inds, fwd_xfer_op_inds,
                        rev_xfer_ip_inds, rev_xfer_op_inds]
        return xfer_indices
