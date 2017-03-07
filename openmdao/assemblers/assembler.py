"""Define the base Assembler class."""

from __future__ import division

from itertools import product
import numpy as np
import warnings

from six.moves import range

from openmdao.utils.units import conversion_to_base_units, convert_units, is_compatible
from openmdao.utils.general_utils import warn_deprecation


class Assembler(object):
    """
    Base Assembler class. The primary purpose of the Assembler class is to set up transfers.

    In attribute names:
        abs / abs_name : absolute, unpromoted variable name, seen from root (unique).
        rel / rel_name : relative, unpromoted variable name, seen from current system (unique).
        prom / prom_name : relative, promoted variable name, seen from current system (non-unique).
        idx : global variable index among variables on all procs (input/output indices separate).
        my_idx : index among variables in this system, on this processor (I/O indices separate).
        io : indicates explicitly that input and output variables are combined in the same dict.

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
    _varx_allprocs_abs2idx_io : dict
        Dictionary mapping absolute names to global indices.
        Both inputs and outputs are contained in one combined dictionary.
        For the global indices, input and output variable indices are tracked separately.
    _varx_allprocs_abs2meta_io : dict
        Dictionary mapping absolute names to metadata dictionaries.
        Both inputs and outputs are contained in one combined dictionary.
    _varx_allprocs_abs_names : {'input': [str, ...], 'output': [str, ...]}
        List of absolute names of all owned variables, on all procs (maps idx to abs_name).
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

        # [REFACTOR]
        self._varx_allprocs_abs2idx_io = {}
        self._varx_allprocs_abs2meta_io = {}
        self._varx_allprocs_abs_names = {'input': [], 'output': []}

        self._input_src_ids = None
        self._src_indices = None
        self._src_indices_range = None

        self._src_units = []
        self._src_scaling = None

    def _setupx_variables(self, allprocs_abs_names):
        """
        Compute absolute name to/from idx maps for variables on all procs.

        Sets the following attributes:
            _varx_allprocs_abs_names
            _varx_allprocs_abs2idx_io

        Parameters
        ----------
        allprocs_abs_names : {'input': [str, ...], 'output': [str, ...]}
            List of absolute names of all owned variables, on all procs (maps idx to abs_name).
        """
        self._varx_allprocs_abs_names = allprocs_abs_names
        self._varx_allprocs_abs2idx_io = {}
        for type_ in ['input', 'output']:
            for idx, abs_name in enumerate(allprocs_abs_names[type_]):
                self._varx_allprocs_abs2idx_io[abs_name] = idx

    def _setup_variables(self, data, abs_names):
        """
        Compute the variable sets and sizes.

        Sets the following attributes:
            _variable_sizes_all
            _variable_sizes
            _variable_set_IDs
            _variable_set_indices

        Parameters
        ----------
        data : {vname1: {'metadata': {}, ...}, ...}
            dict of abs var name to data dict for variables on this proc.
        abs_names : {'input': [str, ...], 'output': [str, ...]}
            lists of absolute names of input and output variables on this proc.
        """
        nproc = self._comm.size
        indices = self._varx_allprocs_abs2idx_io

        for typ in ['input', 'output']:
            nvar = len(abs_names[typ])
            nvar_all = len(self._varx_allprocs_abs_names[typ])

            # Locally determine var_set for each var
            local_set_dict = {}
            for ivar, absname in enumerate(abs_names[typ]):
                ivar_all = indices[absname]
                local_set_dict[ivar_all] = data[absname]['metadata']['var_set']

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
            var_count = np.zeros(len(self._variable_set_IDs[typ]), int)
            self._variable_set_indices[typ] = -np.ones((nvar_all, 2), int)
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
                    np.zeros((nproc, var_count[iset]), int))

            self._variable_sizes_all[typ] = np.zeros(
                (nproc, np.sum(var_count)), int)

        # Populate the sizes arrays
        iproc = self._comm.rank
        for typ in ['input', 'output']:
            for ivar, absname in enumerate(abs_names[typ]):
                size = np.prod(data[absname]['metadata']['shape'])
                ivar_all = indices[absname]
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

    def _setup_connections(self, connections, prom2abs, abs2data):
        """
        Identify implicit connections and combine with explicit ones.

        Sets the following attributes:
            _input_src_ids

        Parameters
        ----------
        connections : [(int, int), ...]
            index pairs representing user defined variable connections
            (in_ind, out_ind).
        prom2abs : {'input': dict, 'output': dict}
            Mapping of promoted name to absolute names (global)
        abs2data : {str: {}, ...}
            Mapping of absolute pathname to data dict  (local)
        """
        out_paths = self._varx_allprocs_abs_names['output']
        in_paths = self._varx_allprocs_abs_names['input']
        input_src_ids = np.full(len(in_paths), -1, dtype=int)
        output_tgt_ids = [[] for i in range(len(out_paths))]
        abs2idx = self._varx_allprocs_abs2idx_io
        prom2abs_out = prom2abs['output']
        prom2abs_in = prom2abs['input']

        # Add user defined connections to the _input_src_ids vector
        for in_ID, out_ID in connections:
            input_src_ids[in_ID] = out_ID
            output_tgt_ids[out_ID].append(in_ID)

        # Add connections for any promoted input names that match promoted
        # output names.
        for prom in prom2abs_out:
            if prom in prom2abs_in:
                oidx = abs2idx[prom2abs_out[prom][0]]
                for ipath in prom2abs_in[prom]:
                    iidx = abs2idx[ipath]
                    input_src_ids[iidx] = oidx
                    output_tgt_ids[oidx].append(iidx)

        # Now check unit compatability for each connection
        for out_ID, in_IDs in enumerate(output_tgt_ids):
            if in_IDs:
                if out_paths[out_ID] not in abs2data:
                    # TODO: we need to gather unit info. Otherwise we can't
                    # check units for connections that cross proc boundaries.
                    continue
                odata = abs2data[out_paths[out_ID]]
                out_units = odata['metadata']['units']
                in_unit_list = []
                for in_ID in in_IDs:
                    # TODO: fix this after we have gathered metadata for units,
                    # but for now, if any input is out-of-process, skip all of
                    # the units checks
                    if in_paths[in_ID] not in abs2data:
                        in_unit_list = []
                        break
                    idata = abs2data[in_paths[in_ID]]
                    in_unit_list.append((idata['metadata']['units'], in_ID))

                if out_units:
                    for in_units, in_ID in in_unit_list:
                        if not in_units:
                            warnings.warn("Output '%s' with units of '%s' is "
                                          "connected to input '%s' which has no"
                                          " units." % (out_paths[out_ID],
                                                       out_units,
                                                       in_paths[in_ID]))
                        elif not is_compatible(in_units, out_units):
                            raise RuntimeError("Output units of '%s' for '%s' are"
                                               " incompatible with input units of "
                                               "'%s' for '%s'." %
                                               (out_units, out_paths[out_ID],
                                                in_units, in_paths[in_ID]))
                else:
                    for u, in_ID in in_unit_list:
                        if u is not None:
                            warnings.warn("Input '%s' with units of '%s' is "
                                          "connected to output '%s' which has "
                                          "no units." % (in_paths[in_ID], u,
                                                         out_paths[out_ID]))

        self._input_src_ids = input_src_ids

    def _setup_src_indices(self, data, abs_names):
        """
        Assemble global list of src_indices.

        Sets the following attributes:
            _src_indices
            _src_indices_range

        Parameters
        ----------
        data : {vname1: {'metadata': {}, ...}, ...}
            dict of abs var name to data dict for variables on this proc.
        abs_names : {'input': [str, ...], 'output': [str, ...]}
            lists of absolute names of input and output variables on this proc.
        """
        abs2idx = self._varx_allprocs_abs2idx_io
        indices = self._varx_allprocs_abs2idx_io
        out_all_paths = self._varx_allprocs_abs_names['output']
        in_paths = abs_names['input']

        # Compute total size of indices vector
        total_idx_size = 0
        sizes = np.zeros(len(in_paths), dtype=int)

        for ind, abs_name in enumerate(in_paths):
            sizes[ind] = np.prod(data[abs_name]['metadata']['shape'])

        total_idx_size = np.sum(sizes)

        # Allocate arrays
        self._src_indices = np.zeros(total_idx_size, int)
        self._src_indices_range = np.zeros((len(in_paths), 2), int)

        # Populate arrays
        ind1, ind2 = 0, 0
        for ind, abs_name in enumerate(in_paths):
            in_meta = data[abs_name]['metadata']
            isize = sizes[ind]
            ind2 += isize
            src_indices = in_meta['src_indices']
            if src_indices is None:
                self._src_indices[ind1:ind2] = np.arange(isize, dtype=int)
            elif src_indices.ndim == 1:
                self._src_indices[ind1:ind2] = src_indices
            else:
                src_id = self._input_src_ids[indices[abs_name]]
                if src_id == -1:  # input is not connected
                    self._src_indices[ind1:ind2] = np.arange(isize, dtype=int)
                else:
                    # TODO: the src may not be in this processes and we need its shape
                    if out_all_paths[src_id] not in data:
                        raise NotImplementedError("accessing source metadata from "
                                                  "another process isn't supported "
                                                  "yet.")
                    src_shape = data[out_all_paths[src_id]]['metadata']['shape']
                    if len(src_shape) == 1:
                        self._src_indices[ind1:ind2] = src_indices.flat
                    else:
                        tgt_shape = in_meta['shape']
                        # loop over src_indices tuples to get indices into the source
                        entries = [list(range(x)) for x in tgt_shape]
                        cols = np.vstack(src_indices[i] for i in product(*entries))
                        dimidxs = [cols[:, i] for i in range(cols.shape[1])]
                        self._src_indices[ind1:ind2] = np.ravel_multi_index(dimidxs, src_shape)

            self._src_indices_range[indices[abs_name], :] = [ind1, ind2]
            ind1 += isize

    def _setup_src_data(self, abs_out_names, data):
        """
        Compute and store unit/scaling information for inputs.

        Parameters
        ----------
        abs_out_names : list of str
            list of absolute names of outputs for the root system on this proc.
        data : {str: {}, ...}
            Mapping of absolute name to data dict for vars on this proc.
        """
        nvar_out = len(abs_out_names)
        indices = self._varx_allprocs_abs2idx_io

        # The out_* variables are lists of units, output indices, and scaling coeffs.
        # for local outputs. These will initialized, then broadcast to all processors
        # since not all variables are declared on all processors, then their data will
        # be put in the _src_units and _src_scaling_0/1 attributes, where they are
        # ordered by target input, rather than all the outputs in order.

        # List of units of locally declared output variables.
        out_units = [data[name]['metadata']['units'] for name in abs_out_names]

        # List of global indices of the locally declared output variables.
        out_inds = [indices[name] for name in abs_out_names]

        # List of scaling coefficients such that
        # physical_unscaled = c0 + c1 * unitless_scaled
        # where c0 and c1 are the two columns of out_scaling.
        # Below, ref0 and ref are the values of the variable in the specified
        # units at which the scaled values are 0 and 1, respectively.
        out_scaling = np.empty((nvar_out, 2))
        for ivar_out, abs_name in enumerate(abs_out_names):
            meta = data[abs_name]['metadata']
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
            out_inds = np.vstack(out_inds_raw)
            out_scaling = np.vstack(out_scaling_raw)

        # Now, we can store the units and scaling coefficients by input
        # by referring to the out_* variables via the input-to-src mapping
        # which is called _input_src_ids.
        nvar_in = len(self._input_src_ids)
        self._src_units = [None for ind in range(nvar_in)]
        self._src_scaling = np.empty((nvar_in, 2))
        for ivar_in, ivar_out in enumerate(self._input_src_ids):
            if ivar_out != -1:
                ind = np.where(out_inds == ivar_out)[0][0]
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
