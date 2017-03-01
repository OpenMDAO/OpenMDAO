"""Define the Group class."""
from __future__ import division

from six import iteritems, string_types
from collections import Iterable

import numpy

from openmdao.core.system import System, PathData
from openmdao.solvers.nl_bgs import NonlinearBlockGS
from openmdao.solvers.ln_bgs import LinearBlockGS
from openmdao.utils.general_utils import warn_deprecation
from openmdao.utils.units import is_compatible


class Group(System):
    """
    Class used to group systems together; instantiate or inherit.
    """

    def __init__(self, **kwargs):
        """
        Set the solvers to nonlinear and linear block Gauss--Seidel by default.

        Parameters
        ----------
        **kwargs : dict
            dict of arguments available here and in all descendants of this
            Group.
        """
        super(Group, self).__init__(**kwargs)

        # TODO: we cannot set the solvers with property setters at the moment
        # because our lint check thinks that we are defining new attributes
        # called nl_solver and ln_solver without documenting them.
        if not self._nl_solver:
            self._nl_solver = NonlinearBlockGS()
        if not self._ln_solver:
            self._ln_solver = LinearBlockGS()

    def add(self, name, subsys, promotes=None):
        """
        Deprecated version of <Group.add_subsystem>.

        Parameters
        ----------
        name : str
            Name of the subsystem being added
        subsys : System
            An instantiated, but not-yet-set up system object.
        promotes : iter of str, optional
            A list of variable names specifying which subsystem variables
            to 'promote' up to this group. This is for backwards compatibility
            with older versions of OpenMDAO.
        """
        warn_deprecation('This method provides backwards compatibility with '
                         'OpenMDAO <= 1.x ; use add_subsystem instead.')

        self.add_subsystem(name, subsys, promotes=promotes)

    def add_subsystem(self, name, subsys, promotes=None,
                      promotes_inputs=None, promotes_outputs=None):
        """
        Add a subsystem.

        Parameters
        ----------
        name : str
            Name of the subsystem being added
        subsys : <System>
            An instantiated, but not-yet-set up system object.
        promotes : str, iter of str, optional
            One or a list of variable names specifying which subsystem variables
            to 'promote' up to this group. This is for backwards compatibility
            with older versions of OpenMDAO.
        promotes_inputs : str, iter of str, optional
            One or a list of input variable names specifying which subsystem input
            variables to 'promote' up to this group.
        promotes_outputs : str, iter of str, optional
            One or a list of output variable names specifying which subsystem output
            variables to 'promote' up to this group.

        Returns
        -------
        <System>
            the subsystem that was passed in. This is returned to
            enable users to instantiate and add a subsystem at the
            same time, and get the pointer back.
        """
        for sub in self._subsystems_allprocs:
            if name == sub.name:
                raise RuntimeError("Subsystem name '%s' is already used." %
                                   name)

        self._subsystems_allprocs.append(subsys)
        subsys.name = name

        # If we're given a string, turn into a list
        if isinstance(promotes, string_types):
            promotes = [promotes]
        if isinstance(promotes_inputs, string_types):
            promotes_inputs = [promotes_inputs]
        if isinstance(promotes_outputs, string_types):
            promotes_outputs = [promotes_outputs]

        if promotes:
            subsys._var_promotes['any'] = set(promotes)
        if promotes_inputs:
            subsys._var_promotes['input'] = set(promotes_inputs)
        if promotes_outputs:
            subsys._var_promotes['output'] = set(promotes_outputs)

        return subsys

    def connect(self, out_name, in_name, src_indices=None):
        """
        Connect output out_name to input in_name in this namespace.

        Parameters
        ----------
        out_name : str
            name of the output (source) variable to connect
        in_name : str or [str, ... ] or (str, ...)
            name of the input or inputs (target) variable to connect
        src_indices : collection of int optional
            When an input variable connects to some subset of an array output
            variable, you can specify which indices of the source to be
            transferred to the input here.
        """
        # if src_indices argument is given, it should be valid
        if isinstance(src_indices, string_types):
            if isinstance(in_name, string_types):
                in_name = [in_name]
            in_name.append(src_indices)
            raise TypeError("src_indices must be an index array, did you mean"
                            " connect('%s', %s)?" % (out_name, in_name))

        if isinstance(src_indices, Iterable):
            src_indices = numpy.atleast_1d(src_indices)

        if isinstance(src_indices, numpy.ndarray):
            if not numpy.issubdtype(src_indices.dtype, numpy.integer):
                raise TypeError("src_indices must contain integers, but src_indices for "
                                "connection from '%s' to '%s' is %s." %
                                (out_name, in_name, src_indices.dtype.type))

        # if multiple targets are given, recursively connect to each
        if isinstance(in_name, (list, tuple)):
            for name in in_name:
                self.connect(out_name, name, src_indices)
            return

        # target should not already be connected
        if in_name in self._var_connections:
            srcname = self._var_connections[in_name][0]
            raise RuntimeError("Input '%s' is already connected to '%s'." %
                               (in_name, srcname))

        # source and target should not be in the same system
        if out_name.rsplit('.', 1)[0] == in_name.rsplit('.', 1)[0]:
            raise RuntimeError("Output and input are in the same System for " +
                               "connection from '%s' to '%s'." % (out_name, in_name))

        self._var_connections[in_name] = (out_name, src_indices)

    def _setup_connections(self):
        """
        Recursively assemble a list of input-output connections.

        Sets the following attributes:
            _var_connections_indices
        """
        # Perform recursion and assemble pairs from subsystems
        pairs = []
        for subsys in self._subsystems_myproc:
            subsys._setup_connections()
            if subsys.comm.rank == 0:
                pairs.extend(subsys._var_connections_indices)

        # Do an allgather to gather from root procs of all subsystems
        if self.comm.size > 1:
            pairs_raw = self.comm.allgather(pairs)
            pairs = []
            for sub_pairs in pairs_raw:
                pairs.extend(sub_pairs)

        allprocs_in_names = self._var_allprocs_names['input']
        myproc_in_names = self._var_myproc_names['input']
        myproc_out_names = self._var_myproc_names['output']
        allprocs_out_names = self._var_allprocs_names['output']
        input_meta = self._var_myproc_metadata['input']
        output_meta = self._var_myproc_metadata['output']

        in_offset = self._var_allprocs_range['input'][0]
        out_offset = self._var_allprocs_range['output'][0]

        # Loop through user-defined connections
        for in_name, (out_name, src_indices) \
                in iteritems(self._var_connections):

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if out_name not in allprocs_out_names:
                raise NameError("Output '%s' does not exist for connection "
                                "in '%s' from '%s' to '%s'." %
                                (out_name, self.name, out_name, in_name))

            if in_name not in allprocs_in_names:
                raise NameError("Input '%s' does not exist for connection "
                                "in '%s' from '%s' to '%s'." %
                                (in_name, self.name, out_name, in_name))

            # throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            out_subsys = out_name.rsplit('.', 1)[0] if '.' in out_name \
                else self._find_subsys_with_promoted_name(out_name, 'output')

            in_subsys = in_name.rsplit('.', 1)[0] if '.' in in_name \
                else self._find_subsys_with_promoted_name(in_name, 'input')

            if out_subsys == in_subsys:
                raise RuntimeError("Output and input are in the same System " +
                                   "for connection in '%s' from '%s' to '%s'." %
                                   (self.name, out_name, in_name))

            out_path = self._var_name2path['output'][out_name]
            pdata = self._var_pathdict[out_path]
            # TODO: we need to allgather unit information. Otherwise we can't
            # error check units for any connections that cross processes
            # because the metadata isn't available.
            if pdata.myproc_idx is not None:
                out_units = output_meta[pdata.myproc_idx]['units']
                in_paths = self._var_name2path['input'][in_name]

                for in_path in in_paths:
                    pdata = self._var_pathdict[in_path]
                    if pdata.myproc_idx is None:
                        continue
                    in_units = input_meta[pdata.myproc_idx]['units']

                    # throw an error if one of input and output is unitless,
                    # but the other isn't
                    if (out_units and not in_units or
                            in_units and not out_units):
                        if out_units:
                            out_units = "has units '%s'" % out_units
                        else:
                            out_units = "is unitless"
                        if in_units:
                            in_units = "has units '%s'" % in_units
                        else:
                            in_units = "is unitless"
                        raise RuntimeError("Units must be specified for both or"
                                           " neither side of connection in "
                                           "'%s': '%s' %s but '%s' %s." %
                                           (self.name, out_name, out_units,
                                            in_name, in_units))

                    # throw an error if the input and output units are not
                    # compatible
                    if not is_compatible(in_units, out_units):
                        raise RuntimeError("Output and input units are not "
                                           "compatible for connection in '%s':"
                                           " '%s' has units '%s' but '%s' has "
                                           "units '%s'." %
                                           (self.name, out_name, out_units,
                                            in_name, in_units))

            for in_index, name in enumerate(allprocs_in_names):
                if name == in_name:
                    try:
                        out_index = allprocs_out_names.index(out_name)
                    except ValueError:
                        continue
                    else:
                        pairs.append((in_index + in_offset,
                                      out_index + out_offset))

                    if src_indices is not None:
                        # set the 'src_indices' metadata in the input variable
                        try:
                            in_myproc_index = myproc_in_names.index(in_name)
                        except ValueError:
                            pass
                        else:
                            meta = input_meta[in_myproc_index]
                            if meta['src_indices'] is not None:
                                raise RuntimeError("%s: src_indices has been defined "
                                                   "in both connect('%s', '%s') "
                                                   "and add_input('%s', ...)." %
                                                   (self.pathname, out_name,
                                                    in_name, in_name))
                            meta['src_indices'] = numpy.atleast_1d(src_indices)

                        # set src_indices to None to avoid unnecessary repeat
                        # of setting indices and shape metadata when we have
                        # multiple inputs promoted to the same name.
                        src_indices = None

        self._var_connections_indices = pairs

    def _find_subsys_with_promoted_name(self, var_name, io_type='output'):
        """
        Find subsystem that contains promoted variable.

        Parameters
        ----------
        var_name : str
            variable name
        io_type : str
            'output' or 'input'.

        Returns
        -------
        str
            name of subsystem, None if not found.
        """
        for subsys in self._subsystems_allprocs:
            for name, prom_name in iteritems(subsys._var_maps[io_type]):
                if var_name == prom_name:
                    return subsys.name
        return None

    def initialize_variables(self):
        """
        Set up variable name and metadata lists.
        """
        self._var_pathdict = {}
        self._var_name2path = {'input': {}, 'output': {}}

        start = len(self.pathname) + 1 if self.pathname else 0
        for typ in ['input', 'output']:
            my_idx_dict = {}  # maps absolute path to myproc idx
            myproc_names = self._var_myproc_names[typ]
            name2path = self._var_name2path[typ]

            for subsys in self._subsystems_myproc:
                # Assemble the names list from subsystems
                subsys._var_maps[typ] = subsys._get_maps(typ)
                paths = subsys._var_allprocs_pathnames[typ]

                for idx, subname in enumerate(subsys._var_allprocs_names[typ]):
                    name = subsys._var_maps[typ][subname]
                    self._var_allprocs_names[typ].append(name)
                    self._var_allprocs_pathnames[typ].append(paths[idx])
                    my_idx_dict[paths[idx]] = len(myproc_names)
                    myproc_names.append(paths[idx][start:])

                # Assemble the metadata list from the subsystems
                metadata = subsys._var_myproc_metadata[typ]
                self._var_myproc_metadata[typ].extend(metadata)

            # The names list is on all procs, allgather all names
            if self.comm.size > 1:

                # One representative proc from each sub_comm adds names
                sub_comm = self._subsystems_myproc[0].comm
                if sub_comm.rank == 0:
                    names = (self._var_allprocs_names[typ],
                             self._var_allprocs_pathnames[typ])
                else:
                    names = ([], [])

                # Every proc on this comm now has global variable names
                self._var_allprocs_names[typ] = []
                self._var_allprocs_pathnames[typ] = []
                for names, pathnames in self.comm.allgather(names):
                    self._var_allprocs_names[typ].extend(names)
                    self._var_allprocs_pathnames[typ].extend(pathnames)

            for idx, name in enumerate(self._var_allprocs_names[typ]):
                path = self._var_allprocs_pathnames[typ][idx]
                self._var_pathdict[path] = PathData(name, idx,
                                                    my_idx_dict.get(path), typ)
                if name in name2path:
                    if typ is 'input':
                        name2path[name].append(path)
                    else:
                        raise RuntimeError("Output name '%s' refers to "
                                           "multiple outputs: %s." %
                                           (name, [path, name2path[name]]))
                else:
                    if typ is 'input':
                        name2path[name] = [path]
                    else:
                        name2path[name] = path

    def _setupx_variables_myproc(self):
        """
        Compute variable dict/list for variables on the current processor.

        Sets the following attributes:
            _varx_abs2data_io
            _varx_abs_names
        """
        self._varx_abs2data_io = {}
        for type_ in ['input', 'output']:
            self._varx_abs_names[type_] = []

        # Perform recursion to populate the dict and list bottom-up
        for subsys in self._subsystems_myproc:
            subsys._setupx_variables_myproc()

            var_maps = {'input': subsys._get_maps('input'), 'output': subsys._get_maps('output')}

            for type_ in ['input', 'output']:

                # Assemble _varx_abs2data_io and _varx_abs_names by concatenating from subsys.
                for abs_name in subsys._varx_abs_names[type_]:
                    sub_data = subsys._varx_abs2data_io[abs_name]

                    sub_prom_name = sub_data['prom']
                    metadata = sub_data['metadata']

                    prom_name = var_maps[type_][sub_prom_name]
                    if self.pathname == '':
                        rel_name = abs_name
                    else:
                        rel_name = abs_name[len(self.pathname) + 1:]
                    self._varx_abs2data_io[abs_name] = {'prom': prom_name, 'rel': rel_name,
                                                        'my_idx': len(self._varx_abs_names[type_]),
                                                        'type_': type_, 'metadata': metadata}
                    self._varx_abs_names[type_].append(abs_name)

    def _setupx_variable_allprocs_names(self):
        """
        Get the names for variables on all processors.

        Also, compute allprocs var counts and store in _varx_allprocs_idx_range.

        Returns
        -------
        {'input': [str, ...], 'output': [str, ...]}
            List of absolute names of owned variables existing on current proc.
        """
        allprocs_abs_names = {'input': [], 'output': []}

        # First, concatenate the allprocs variable names from subsystems on my proc.
        for subsys in self._subsystems_myproc:
            subsys_allprocs_abs_names = subsys._setupx_variable_allprocs_names()

            for type_ in ['input', 'output']:
                allprocs_abs_names[type_].extend(subsys_allprocs_abs_names[type_])

        # For _varx_allprocs_prom2abs_set, essentially invert the abs2prom map in
        # _varx_abs2data_io capturing at least the local maps.
        self._varx_allprocs_prom2abs_set = {'input': {}, 'output': {}}
        for abs_name, data in iteritems(self._varx_abs2data_io):
            type_ = data['type_']
            prom_name = data['prom']
            if prom_name not in self._varx_allprocs_prom2abs_set[type_]:
                self._varx_allprocs_prom2abs_set[type_][prom_name] = [abs_name]
            else:
                self._varx_allprocs_prom2abs_set[type_][prom_name].append(abs_name)

        # If we're running in parallel, gather contributions from other procs.
        if self.comm.size > 1:
            for type_ in ['input', 'output']:
                sub_comm = self._subsystems_myproc[0].comm
                if sub_comm.rank == 0:
                    raw = (allprocs_abs_names[type_], self._varx_allprocs_prom2abs_set[type_])
                else:
                    raw = ([], {})

                allprocs_abs_names[type_] = []
                allprocs_prom2abs_set = {}
                for abs_names, prom2abs_set in self.comm.allgather(raw):
                    allprocs_abs_names[type_].extend(abs_names)
                    for prom_name, abs_names_set in iteritems(prom2abs_set):
                        if prom_name not in allprocs_prom2abs_set:
                            allprocs_prom2abs_set[prom_name] = abs_names_set
                        else:
                            allprocs_prom2abs_set[prom_name].extend(abs_names_set)

                for prom_name, abs_names_set in iteritems(allprocs_prom2abs_set):
                    allprocs_prom2abs_set[prom_name] = set(abs_names_set)
                self._varx_allprocs_prom2abs_set[type_] = allprocs_prom2abs_set

        # Now allprocs_abs_names is ready, so we get use it to
        # count the total number of allprocs variables and put it in _varx_allprocs_idx_range.
        for type_ in ['input', 'output']:
            self._varx_allprocs_idx_range[type_] = [0, len(allprocs_abs_names[type_])]

        return allprocs_abs_names

    def _setupx_variable_allprocs_indices(self, global_index):
        """
        Compute the global index range for variables on all processors.

        Computes the following attributes:
            _varx_allprocs_idx_range

        Parameters
        ----------
        global_index : {'input': int, 'output': int}
            current global variable counter.
        """
        # First, apply the global_index offset to make _varx_allprocs_idx_range correct.
        for type_ in ['input', 'output']:
            for ind in range(2):
                self._varx_allprocs_idx_range[type_][ind] += global_index[type_]

        # Pre-recursion: compute index to pass to subsystems.
        # This index is the number of variables on procs before current proc
        # Necessary because of multiple global counters on different procs
        if self.comm.size > 1:
            subsys0 = self._subsystems_myproc[0]
            for type_ in ['input', 'output']:
                # Note: the following is valid because _varx_allprocs_idx_range
                # contains [0, # allprocs vars] at this point because
                # _setupx_variable_allprocs_names has been run but the recursion
                # for the current method has not been performed yet.
                local_var_size = subsys0._varx_allprocs_idx_range[type_][1]

                # Compute the variable count list; 0 on rank > 0 procs
                sub_comm = subsys0.comm
                if sub_comm.rank == 0:
                    nvar_myproc = local_var_size
                else:
                    nvar_myproc = 0
                nvar_allprocs = self.comm.allgather(nvar_myproc)

                # Compute the offset
                iproc = self.comm.rank
                nvar_myproc = local_var_size
                global_index[type_] += numpy.sum(nvar_allprocs[:iproc + 1]) - nvar_myproc

        # Perform recursion
        for subsys in self._subsystems_myproc:
            subsys_allprocs_abs_names = subsys._setupx_variable_allprocs_indices(global_index)

        # Reset index dict to the global variable counter on all procs.
        # Necessary for younger siblings to have proper index values.
        for type_ in ['input', 'output']:
            global_index[type_] = self._varx_allprocs_idx_range[type_][1]

    def get_subsystem(self, name):
        """
        Return the system called 'name' in the current namespace.

        Parameters
        ----------
        name : str
            name of the desired system in the current namespace.

        Returns
        -------
        System or None
            System if found else None.
        """
        system = self
        for subname in name.split('.'):
            for sub in system._subsystems_allprocs:
                if sub.name == subname:
                    system = sub
                    break
            else:
                return None
        return system

    def _apply_nonlinear(self):
        """
        Compute residuals. The model is assumed to be in a scaled state.
        """
        self._transfers[None](self._inputs, self._outputs, 'fwd')
        # Apply recursion
        for subsys in self._subsystems_myproc:
            subsys._apply_nonlinear()

    def _solve_nonlinear(self):
        """
        Compute outputs. The model is assumed to be in a scaled state.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self._nl_solver.solve()

    def _apply_linear(self, vec_names, mode, var_inds=None):
        """
        Compute jac-vec product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.
        var_inds : [int, int, int, int] or None
            ranges of variable IDs involved in this matrix-vector product.
            The ordering is [lb1, ub1, lb2, ub2].
        """
        with self._jacobian_context() as J:
            # Use global Jacobian
            if self._owns_global_jac:
                for vec_name in vec_names:
                    with self._matvec_context(vec_name, var_inds, mode) as vecs:
                        d_inputs, d_outputs, d_residuals = vecs
                        J._apply(d_inputs, d_outputs, d_residuals, mode)
            # Apply recursion
            else:
                if mode == 'fwd':
                    for vec_name in vec_names:
                        d_inputs = self._vectors['input'][vec_name]
                        d_outputs = self._vectors['output'][vec_name]
                        self._vector_transfers[vec_name][None](
                            d_inputs, d_outputs, mode)

                for subsys in self._subsystems_myproc:
                    subsys._apply_linear(vec_names, mode, var_inds)

                if mode == 'rev':
                    for vec_name in vec_names:
                        d_inputs = self._vectors['input'][vec_name]
                        d_outputs = self._vectors['output'][vec_name]
                        self._vector_transfers[vec_name][None](
                            d_inputs, d_outputs, mode)

    def _solve_linear(self, vec_names, mode):
        """
        Apply inverse jac product. The model is assumed to be in a scaled state.

        Parameters
        ----------
        vec_names : [str, ...]
            list of names of the right-hand-side vectors.
        mode : str
            'fwd' or 'rev'.

        Returns
        -------
        boolean
            Failure flag; True if failed to converge, False is successful.
        float
            relative error.
        float
            absolute error.
        """
        return self._ln_solver.solve(vec_names, mode)

    def _linearize(self):
        """
        Compute jacobian / factorization. The model is assumed to be in a scaled state.
        """
        with self._jacobian_context() as J:
            for subsys in self._subsystems_myproc:
                subsys._linearize()

            # Update jacobian
            if self._owns_global_jac:
                J._update()

        if self._ln_solver is not None:
            self._ln_solver._linearize()
