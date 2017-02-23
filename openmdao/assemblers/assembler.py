"""Define the base Assembler class."""

from __future__ import division

from itertools import product
import numpy

from six.moves import range

from openmdao.utils.units import conversion_to_base_units, convert_units


class Assembler(object):
    """
    Base Assembler class.

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
    _src_scaling : ndarray[nvar_in, 2]
        scaling coefficients such that physical_unscaled = c0 + c1 * unitless_scaled
        and c0, c1 are the two columns of this array.
    """

    def __init__(self, comm):
        """
        Initialize all attributes.

        Parameters
        ----------
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
        self._src_scaling = None

    def _setup_variables(self, nvars, variable_metadata, variable_indices):
        """
        Compute the variable sets and sizes.

        Sets the following attributes:
            _variable_sizes_all
            _variable_sizes
            _variable_set_IDs
            _variable_set_indices

        Parameters
        ----------
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
        """
        Identify implicit connections and combine with explicit ones.

        Sets the following attributes:
            _input_src_ids

        Parameters
        ----------
        connections : [(int, int), ...]
            index pairs representing user defined variable connections
            (in_ind, out_ind).
        variable_allprocs_names : {'input': [str, ...], 'output': [str, ...]}
            list of names of all owned variables, not just on current proc.
        """
        out_names = variable_allprocs_names['output']
        nvar_input = len(variable_allprocs_names['input'])
        _input_src_ids = -numpy.ones(nvar_input, int)

        # Add user defined connections to the _input_src_ids vector
        # and inconns
        for in_ID, out_ID in connections:
            _input_src_ids[in_ID] = out_ID

        # Loop over input variables
        for in_ID, name in enumerate(variable_allprocs_names['input']):

            # If name is also an output variable, add this implicit connection
            for out_ID, oname in enumerate(out_names):
                if name == oname:
                    _input_src_ids[in_ID] = out_ID
                    break

        self._input_src_ids = _input_src_ids

    def _setup_src_indices(self, metadata, myproc_var_global_indices,
                           var_pathdict, var_allprocs_pathnames):
        """
        Assemble global list of src_indices.

        Sets the following attributes:
            _src_indices
            _src_indices_range

        Parameters
        ----------
        metadata : {'input': [{}, ...], 'output': [{}, ...]}
            list of metadata dictionaries of variables that exist on this proc.
        myproc_var_global_indices : ndarray[:]
            integer arrays of global indices of variables on this proc.
        var_pathdict : dict
            dict that maps absolute pathname to promoted name, global and local index.
        var_allprocs_pathnames : {'input': [], 'output': []}
            absolute pathnames for each input and output var.
        """
        input_metadata = metadata['input']
        output_metadata = metadata['output']

        # Compute total size of indices vector
        total_idx_size = 0
        sizes = numpy.zeros(len(input_metadata), dtype=int)

        for ind, meta in enumerate(input_metadata):
            sizes[ind] = numpy.prod(meta['shape'])

        total_idx_size = numpy.sum(sizes)

        # Allocate arrays
        self._src_indices = numpy.zeros(total_idx_size, int)
        self._src_indices_range = numpy.zeros(
            (myproc_var_global_indices.shape[0], 2), int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind, meta in enumerate(input_metadata):
            isize = sizes[ind]
            ind2 += isize
            src_indices = meta['src_indices']
            if src_indices is None:
                self._src_indices[ind1:ind2] = numpy.arange(isize, dtype=int)
            elif src_indices.ndim == 1:
                self._src_indices[ind1:ind2] = src_indices
            else:
                src_id = self._input_src_ids[myproc_var_global_indices[ind]]
                if src_id == -1:  # input is not connected
                    self._src_indices[ind1:ind2] = numpy.arange(isize, dtype=int)
                else:
                    pdata = var_pathdict[var_allprocs_pathnames['output'][src_id]]
                    # TODO: the src may not be in this processes and we need its shape
                    if pdata.myproc_idx is None:
                        raise NotImplementedError("accessing source metadata from "
                                                  "another process isn't supported "
                                                  "yet.")
                    src_shape = output_metadata[pdata.myproc_idx]['shape']
                    if len(src_shape) == 1:
                        self._src_indices[ind1:ind2] = src_indices.flat
                    else:
                        tgt_shape = meta['shape']
                        # loop over src_indices tuples to get indices into the source
                        entries = [list(range(x)) for x in tgt_shape]
                        cols = numpy.vstack(src_indices[i] for i in product(*entries))
                        dimidxs = [cols[:, i] for i in range(cols.shape[1])]
                        self._src_indices[ind1:ind2] = numpy.ravel_multi_index(dimidxs, src_shape)

            self._src_indices_range[myproc_var_global_indices[ind], :] = [ind1,
                                                                          ind2]
            ind1 += isize

    def _setup_src_data(self, variable_metadata, variable_indices):
        """
        Compute and store unit/scaling information for inputs.

        Parameters
        ----------
        variable_metadata : list of dict
            list of metadata dictionaries for outputs of root system.
        variable_indices : int ndarray
            global indices of outputs that exist on this processor.
        """
        nvar_out = len(variable_metadata)

        # The out_* variables are lists of units, output indices, and scaling coeffs.
        # for local outputs. These will initialized, then broadcast to all processors
        # since not all variables are declared on all processors, then their data will
        # be put in the _src_units and _src_scaling_0/1 attributes, where they are
        # ordered by target input, rather than all the outputs in order.

        # List of units of locally declared output variables.
        out_units = [meta['units'] for meta in variable_metadata]

        # List of global indices of the locally declared output variables.
        out_inds = variable_indices

        # List of scaling coefficients such that
        # physical_unscaled = c0 + c1 * unitless_scaled
        # where c0 and c1 are the two columns of out_scaling.
        # Below, ref0 and ref are the values of the variable in the specified
        # units at which the scaled values are 0 and 1, respectively.
        out_scaling = numpy.empty((nvar_out, 2))
        for ivar_out, meta in enumerate(variable_metadata):
            out_scaling[ivar_out, 0] = meta['ref0']
            out_scaling[ivar_out, 1] = meta['ref'] - meta['ref0']

        # Broadcast to all procs
        if self._comm.size > 1:
            out_units_raw = self._comm.allgather(out_units)
            out_inds_raw = self._comm.allgather(out_inds)
            out_scaling_raw = self._comm.allgather(out_scaling)

            out_units = []
            for str_list in out_units_raw:
                out_units.extend(str_list)
            out_inds = numpy.vstack(out_inds_raw)
            out_scaling = numpy.vstack(out_scaling_raw)

        # Now, we can store the units and scaling coefficients by input
        # by referring to the out_* variables via the input-to-src mapping
        # which is called _input_src_ids.
        nvar_in = len(self._input_src_ids)
        self._src_units = [None for ind in range(nvar_in)]
        self._src_scaling = numpy.empty((nvar_in, 2))
        for ivar_in, ivar_out in enumerate(self._input_src_ids):
            if ivar_out != -1:
                ind = numpy.where(out_inds == ivar_out)[0][0]
                self._src_units[ivar_in] = out_units[ind]
                self._src_scaling[ivar_in, :] = out_scaling[ind, :]
            else:
                self._src_units[ivar_in] = None
                self._src_scaling[ivar_in, :] = [0., 1.]

    def _compute_transfers(self, nsub_allprocs, var_range,
                           subsystems_myproc, subsystems_inds):
        """
        Compute the transfer indices.

        Must be implemented by the subclass.

        Parameters
        ----------
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
        xfer_in_inds : dict of int ndarray[:]
            input indices of global transfer.
        xfer_out_inds : dict of int ndarray[:]
            output indices of global transfer.
        fwd_xfer_in_inds : [dict of int ndarray[:], ...]
            list of input indices of forward transfers.
        fwd_xfer_out_inds : [dict of int ndarray[:], ...]
            list of output indices of forward transfers.
        rev_xfer_in_inds : [dict of int ndarray[:], ...]
            list of input indices of reverse transfers.
        rev_xfer_out_inds : [dict of int ndarray[:], ...]
            list of output indices of reverse transfers.
        """
        pass
