"""Define the base Assembler class."""

from __future__ import division
import numpy

from six.moves import range

from openmdao.utils.units import conversion_to_base_units, convert_units


class Assembler(object):
    """Base Assembler class.

    The primary purpose of the Assembler class is to set up transfers.

    Attributes
    ----------
    _comm : MPI.comm or <FakeComm>
        MPI communicator object.
    _variable_sizes_all : {'input': ndarray[nproc, nvar],
                           'output': ndarray[nproc, nvar]}
        local variable size arrays, num procs x num vars.
    _variable_sizes : {'input': list of ndarray[nproc, nvar],
                       'output': list of ndarray[nproc, nvar]}
        list of local variable size arrays, num procs x num vars by var_set.
    _variable_set_IDs : {'input': {}, 'output': {}}
        dictionary mapping var_set names to their IDs.
    _variable_set_indices : {'input': ndarray[nvar_all, 2],
                             'output': ndarray[nvar_all, 2]}
        the first column is the var_set ID and
        the second column is the variable index within the var_set.
    _input_src_ids : int ndarray[num_input_var]
        the output variable ID for each input variable ID.
    _src_indices : int ndarray[:]
        all the input indices vectors concatenated together.
    _src_indices_range : int ndarray[num_input_var_all, 2]
        the initial and final indices for the indices vector for each input.
    _src_units : [str, ...]
        list of src units whose length is the number of input variables.
    _src_scaling_0 : ndarray(nvar_in)
        list of 0th order scaling coefficients (i.e., a0: in y = a1 * x + a0).
    _src_scaling_1 : ndarray(nvar_in)
        list of 1st order scaling coefficients (i.e., a1: in y = a1 * x + a0).
    """

    def __init__(self, comm):
        """Initialize all attributes.

        Args
        ----
        comm : MPI.Comm or <FakeComm>
            same instance as the <Problem>'s communicator.
        """
        self._comm = comm

        self._variable_sizes_all = {'input': None, 'output': None}
        self._variable_sizes = {'input': [], 'output': []}
        self._variable_set_IDs = {'input': {}, 'output': {}}
        self._variable_set_indices = {'input': None, 'output': None}

        self._input_src_ids = None
        self._src_indices = None
        self._src_indices_range = None

        self._src_units = []
        self._src_scaling_0 = None
        self._src_scaling_1 = None

    def _setup_variables(self, nvars, variable_metadata, variable_indices):
        """Compute the variable sets and sizes.

        Sets the following attributes:
            _variable_sizes_all
            _variable_sizes
            _variable_set_IDs
            _variable_set_indices

        Args
        ----
        nvars : {'input': int, 'output': int}
            global number of variables.
        variable_metadata : {'input': list, 'output': list}
            list of metadata dictionaries of variables that exist on this proc.
        variable_indices : {'input': ndarray[:], 'output': ndarray[:]}
            integer arrays of global indices of variables on this proc.
        """
        nproc = self._comm.size

        for typ in ['input', 'output']:
            nvar = len(variable_metadata[typ])
            nvar_all = nvars[typ]

            # Locally determine var_set for each var
            local_set_dict = {}
            for ivar, meta in enumerate(variable_metadata[typ]):
                ivar_all = variable_indices[typ][ivar]
                local_set_dict[ivar_all] = meta['var_set']

            # Broadcast ivar_all-iset pairs to all procs
            if self._comm.size > 1:
                global_set_dict = {}
                for local_set_dict in self._comm.allgather(local_set_dict):
                    global_set_dict.update(local_set_dict)
            else:
                global_set_dict = local_set_dict

            # Compute set_name to ID maps
            for iset, set_name in enumerate(set(global_set_dict.values())):
                self._variable_set_IDs[typ][set_name] = iset

            # Compute _variable_set_indices and var_count
            var_count = numpy.zeros(len(self._variable_set_IDs[typ]), int)
            self._variable_set_indices[typ] = -numpy.ones((nvar_all, 2), int)
            for ivar_all in global_set_dict:
                set_name = global_set_dict[ivar_all]

                iset = self._variable_set_IDs[typ][set_name]

                self._variable_set_indices[typ][ivar_all, 0] = iset
                self._variable_set_indices[typ][ivar_all, 1] = var_count[iset]

                var_count[iset] += 1

            # Allocate the size arrays using var_count
            self._variable_sizes[typ] = []
            for iset in range(len(self._variable_set_IDs[typ])):
                self._variable_sizes[typ].append(
                    numpy.zeros((nproc, var_count[iset]), int))

            self._variable_sizes_all[typ] = numpy.zeros(
                (nproc, numpy.sum(var_count)), int)

        # Populate the sizes arrays
        iproc = self._comm.rank
        for typ in ['input', 'output']:
            for ivar, meta in enumerate(variable_metadata[typ]):
                size = numpy.prod(meta['shape'])
                ivar_all = variable_indices[typ][ivar]
                iset, ivar_set = self._variable_set_indices[typ][ivar_all, :]
                self._variable_sizes[typ][iset][iproc, ivar_set] = size
                self._variable_sizes_all[typ][iproc, ivar_all] = size

        # Do an allgather on the sizes arrays
        if self._comm.size > 1:
            for typ in ['input', 'output']:
                nset = len(self._variable_sizes[typ])
                for iset in range(nset):
                    array = self._variable_sizes[typ][iset]
                    self._comm.Allgather(array[iproc, :], array)
                self._comm.Allgather(self._variable_sizes_all[typ][iproc, :],
                                     self._variable_sizes_all[typ])

    def _setup_connections(self, connections, variable_allprocs_names):
        """Identify implicit connections and combine with explicit ones.

        Sets the following attributes:
            _input_src_ids

        Args
        ----
        connections : [(int, int), ...]
            index pairs representing user defined variable connections
            (ip_ind, op_ind).
        variable_allprocs_names : {'input': [str, ...], 'output': [str, ...]}
            list of names of all owned variables, not just on current proc.
        """
        out_names = variable_allprocs_names['output']
        nvar_input = len(variable_allprocs_names['input'])
        _input_src_ids = -numpy.ones(nvar_input, int)

        # Add user defined connections to the _input_src_ids vector
        # and inconns
        for ip_ID, op_ID in connections:
            _input_src_ids[ip_ID] = op_ID

        # Loop over input variables
        for ip_ID, name in enumerate(variable_allprocs_names['input']):

            # If name is also an output variable, add this implicit connection
            for op_ID, oname in enumerate(out_names):
                if name == oname:
                    _input_src_ids[ip_ID] = op_ID
                    break

        self._input_src_ids = _input_src_ids

    def _setup_src_indices(self, input_metadata, myproc_var_global_indices):
        """Assemble global list of src_indices.

        Sets the following attributes:
            _src_indices
            _src_indices_range

        Args
        ----
        input_metadata : [{}, ...]
            list of metadata dictionaries of inputs that exist on this proc.
        myproc_var_global_indices : ndarray[:]
            integer arrays of global indices of variables on this proc.
        """
        # Compute total size of indices vector
        total_idx_size = 0
        sizes = numpy.zeros(len(input_metadata), dtype=int)

        for ind, metadata in enumerate(input_metadata):
            sizes[ind] = numpy.prod(metadata['shape'])

        total_idx_size = numpy.sum(sizes)

        # Allocate arrays
        self._src_indices = numpy.zeros(total_idx_size, int)
        self._src_indices_range = numpy.zeros(
            (myproc_var_global_indices.shape[0], 2), int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind, metadata in enumerate(input_metadata):
            isize = sizes[ind]
            ind2 += isize
            indices = metadata['indices']
            if indices is None:
                self._src_indices[ind1:ind2] = numpy.arange(isize, dtype=int)
            else:
                self._src_indices[ind1:ind2] = indices.flat
            self._src_indices_range[myproc_var_global_indices[ind], :] = [ind1,
                                                                          ind2]
            ind1 += isize

    def _setup_src_data(self, variable_metadata, variable_indices):
        """Compute and store unit/scaling information for inputs.

        Args
        ----
        variable_metadata : list of dict
            list of metadata dictionaries for outputs of root system.
        variable_indices : int ndarray
            global indices of outputs that exist on this processor.
        """
        nvar_out = len(variable_metadata)

        # List of src units; to check compatability with input units
        op_units = [None for ind in range(nvar_out)]
        # List of unit_type IDs
        op_int = numpy.empty(nvar_out, int)
        # The two columns correspond to ref0 and ref
        op_flt = numpy.empty((nvar_out, 2))

        # Get unit type as well as ref0 and ref in standard units
        op_int[:] = variable_indices
        for ivar_out, meta in enumerate(variable_metadata):
            # ref0 and ref are the values of the variable in the specified
            # units at which the scaled values are 0 and 1, respectively
            op_units[ivar_out] = meta['units']
            op_flt[ivar_out, 0] = meta['ref0']
            op_flt[ivar_out, 1] = meta['ref'] - meta['ref0']

        # Broadcast to all procs
        if self._comm.size > 1:
            op_units_raw = self._comm.allgather(op_units)
            op_int_raw = self._comm.allgather(op_int)
            op_flt_raw = self._comm.allgather(op_flt)

            op_units = []
            for str_list in op_units_raw:
                op_units.extend(str_list)
            op_int = numpy.vstack(op_int_raw)
            op_flt = numpy.vstack(op_flt_raw)

        # Now, we can store ref0 and ref for each input
        nvar_in = len(self._input_src_ids)
        self._src_units = [None for ind in range(nvar_in)]
        self._src_scaling_0 = numpy.empty(nvar_in)
        self._src_scaling_1 = numpy.empty(nvar_in)
        for ivar_in, ivar_out in enumerate(self._input_src_ids):
            if ivar_out != -1:
                ind = numpy.where(op_int == ivar_out)[0][0]
                self._src_units[ivar_in] = op_units[ind]
                self._src_scaling_0[ivar_in] = op_flt[ind, 0]
                self._src_scaling_1[ivar_in] = op_flt[ind, 1]
            else:
                self._src_units[ivar_in] = ''
                self._src_scaling_0[ivar_in] = 0.
                self._src_scaling_1[ivar_in] = 1.

    def _compute_transfers(self, nsub_allprocs, var_range,
                           subsystems_myproc, subsystems_inds):
        """Compute the transfer indices.

        Must be implemented by the subclass.

        Args
        ----
        nsub_allprocs : int
            number of subsystems on all procs.
        var_range : [int, int]
            variable index range for the current system.
        subsystems_myproc : [System, ...]
            list of subsystems on my proc.
        subsystems_inds : [int, ...]
            list of indices of subsystems on this proc among all subsystems.

        Returns
        -------
        xfer_ip_inds : dict of int ndarray[:]
            input indices of global transfer.
        xfer_op_inds : dict of int ndarray[:]
            output indices of global transfer.
        fwd_xfer_ip_inds : [dict of int ndarray[:], ...]
            list of input indices of forward transfers.
        fwd_xfer_op_inds : [dict of int ndarray[:], ...]
            list of output indices of forward transfers.
        rev_xfer_ip_inds : [dict of int ndarray[:], ...]
            list of input indices of reverse transfers.
        rev_xfer_op_inds : [dict of int ndarray[:], ...]
            list of output indices of reverse transfers.
        """
        pass
