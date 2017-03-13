"""Define the Group class."""
from __future__ import division

from six import iteritems, string_types
from collections import Iterable, Counter

import numpy as np

from openmdao.core.system import System, PathData
from openmdao.solvers.nl_runonce import NLRunOnce
from openmdao.solvers.ln_runonce import LNRunOnce
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
            self._nl_solver = NLRunOnce()
        if not self._ln_solver:
            self._ln_solver = LNRunOnce()

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

        Returns
        -------
        System
            The System that was passed in.
        """
        warn_deprecation('This method provides backwards compatibility with '
                         'OpenMDAO <= 1.x ; use add_subsystem instead.')

        return self.add_subsystem(name, subsys, promotes=promotes)

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
            src_indices = np.atleast_1d(src_indices)

        if isinstance(src_indices, np.ndarray):
            if not np.issubdtype(src_indices.dtype, np.integer):
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

    def set_order(self, new_order):
        """
        Specify a new execution order for this system.

        Parameters
        ----------
        new_order : list of str
            List of system names in desired new execution order.
        """
        # Make sure the new_order is valid. It must contain all subsystems
        # in this model.
        newset = set(new_order)
        olddict = {s.name: s for s in self._subsystems_allprocs}
        oldset = set(olddict)

        if oldset != newset:
            msg = []

            missing = oldset - newset
            if missing:
                msg.append("%s: %s expected in subsystem order and not found." %
                           (self.pathname, sorted(missing)))

            extra = newset - oldset
            if extra:
                msg.append("%s: subsystem(s) %s found in subsystem order but don't exist." %
                           (self.pathname, sorted(extra)))

            raise ValueError('\n'.join(msg))

        # Don't allow duplicates either.
        if len(newset) < len(new_order):
            dupes = [key for key, val in iteritems(Counter(new_order)) if val > 1]
            raise ValueError("%s: Duplicate name(s) found in subsystem order list: %s" %
                             (self.pathname, sorted(dupes)))

        self._subsystems_allprocs = [olddict[name] for name in new_order]

    def _setup_connections(self):
        """
        Recursively assemble a list of input-output connections.

        Sets the following attributes:
            _var_connections_abs
        """
        # Perform recursion and assemble pairs from subsystems
        pairs = []
        for subsys in self._subsystems_myproc:
            subsys._setup_connections()
            if subsys.comm.rank == 0:
                pairs.extend(subsys._var_connections_abs)

        # Do an allgather to gather from root procs of all subsystems
        if self.comm.size > 1:
            pairs_raw = self.comm.allgather(pairs)
            pairs = []
            for sub_pairs in pairs_raw:
                pairs.extend(sub_pairs)

        in_offset = self._var_allprocs_idx_range['input'][0]
        out_offset = self._var_allprocs_idx_range['output'][0]

        abs2data = self._var_abs2data_io
        prom2abs_in = self._var_allprocs_prom2abs_list['input']
        prom2abs_out = self._var_allprocs_prom2abs_list['output']

        # Loop through user-defined connections
        for in_name, (out_name, src_indices) \
                in iteritems(self._var_connections):

            # throw an exception if either output or input doesn't exist
            # (not traceable to a connect statement, so provide context)
            if out_name not in prom2abs_out:
                raise NameError("Output '%s' does not exist for connection "
                                "in '%s' from '%s' to '%s'." %
                                (out_name, self.pathname, out_name, in_name))

            if in_name not in prom2abs_in:
                raise NameError("Input '%s' does not exist for connection "
                                "in '%s' from '%s' to '%s'." %
                                (in_name, self.pathname, out_name, in_name))

            # throw an exception if output and input are in the same system
            # (not traceable to a connect statement, so provide context)
            abs_out = prom2abs_out[out_name][0]
            out_subsys = abs_out.rsplit('.', 1)[0]
            for abs_in in prom2abs_in[in_name]:
                in_subsys = abs_in.rsplit('.', 1)[0]
                if out_subsys == in_subsys:
                    raise RuntimeError("Output and input are in the same System " +
                                       "for connection in '%s' from '%s' to '%s'." %
                                       (self.pathname, out_name, in_name))

                if src_indices is not None:
                    meta = abs2data[abs_in]['metadata']
                    if meta['src_indices'] is not None:
                        raise RuntimeError("%s: src_indices has been defined "
                                           "in both connect('%s', '%s') "
                                           "and add_input('%s', ...)." %
                                           (self.pathname, out_name,
                                            in_name, in_name))
                    meta['src_indices'] = np.atleast_1d(src_indices)

                pairs.append((abs_in, abs_out))

        self._var_connections_abs = pairs

    def initialize_variables(self):
        """
        Set up variable name and metadata lists.
        """
        pass

    def _setup_variables(self):
        """
        Compute variable dict/list for variables on the current processor.

        Sets the following attributes:
            _var_abs2data_io
            _var_abs_names
            _var_allprocs_prom2abs_list

        Returns
        -------
        {'input': [str, ...], 'output': [str, ...]}
            List of absolute names of owned variables existing on current proc.
        """
        super(Group, self)._setup_variables()

        self._var_abs2data_io = {}
        for type_ in ['input', 'output']:
            self._var_abs_names[type_] = []

        name_offset = len(self.pathname) + 1 if self.pathname else 0
        allprocs_abs_names = {'input': [], 'output': []}

        # Perform recursion to populate the dict and list bottom-up
        for isub, subsys in enumerate(self._subsystems_myproc):
            subsys_allprocs_abs_names = subsys._setup_variables()

            var_maps = subsys._get_maps()
            for type_ in ['input', 'output']:
                # concatenate the allprocs variable names from subsystems on my proc.
                allprocs_abs_names[type_].extend(subsys_allprocs_abs_names[type_])

                # Assemble _var_abs2data_io and _var_abs_names by concatenating from subsystems.
                for abs_name in subsys._var_abs_names[type_]:
                    sub_data = subsys._var_abs2data_io[abs_name]

                    self._var_abs2data_io[abs_name] = {
                        'prom': var_maps[type_][sub_data['prom']],
                        'rel': abs_name[name_offset:] if name_offset > 0 else abs_name,
                        'my_idx': len(self._var_abs_names[type_]),
                        'type': type_,
                        'metadata': sub_data['metadata']
                    }
                    self._var_abs_names[type_].append(abs_name)

        # For _var_allprocs_prom2abs_list, essentially invert the abs2prom map in
        # _var_abs2data_io to capture at least the local maps.
        self._var_allprocs_prom2abs_list = {'input': {}, 'output': {}}
        for abs_name, data in iteritems(self._var_abs2data_io):
            type_ = data['type']
            prom_name = data['prom']
            if prom_name not in self._var_allprocs_prom2abs_list[type_]:
                self._var_allprocs_prom2abs_list[type_][prom_name] = [abs_name]
            else:
                self._var_allprocs_prom2abs_list[type_][prom_name].append(abs_name)

        for prom_name, lst in iteritems(self._var_allprocs_prom2abs_list['output']):
            if len(lst) > 1:
                raise RuntimeError("Output name '%s' refers to "
                                   "multiple outputs: %s." %
                                   (prom_name, sorted(lst)))

        # If we're running in parallel, gather contributions from other procs.
        if self.comm.size > 1:
            for type_ in ['input', 'output']:
                sub_comm = self._subsystems_myproc[0].comm
                if sub_comm.rank == 0:
                    raw = (allprocs_abs_names[type_], self._var_allprocs_prom2abs_list[type_])
                else:
                    raw = ([], {})

                allprocs_abs_names[type_] = []
                allprocs_prom2abs_list = {}
                for abs_names, prom2abs_list in self.comm.allgather(raw):
                    allprocs_abs_names[type_].extend(abs_names)
                    for prom_name, abs_names_list in iteritems(prom2abs_list):
                        if prom_name not in allprocs_prom2abs_list:
                            allprocs_prom2abs_list[prom_name] = abs_names_list
                        else:
                            allprocs_prom2abs_list[prom_name].extend(abs_names_list)

                self._var_allprocs_prom2abs_list[type_] = allprocs_prom2abs_list

        # We use allprocs_abs_names to count the total number of allprocs variables
        # and put it in _var_allprocs_idx_range.
        for type_ in ['input', 'output']:
            self._var_allprocs_idx_range[type_] = [0, len(allprocs_abs_names[type_])]

        return allprocs_abs_names

    def _setup_variable_indices(self, global_index):
        """
        Compute the global index range for variables on all processors.

        Computes the following attributes:
            _var_allprocs_idx_range

        Parameters
        ----------
        global_index : {'input': int, 'output': int}
            current global variable counter.
        """
        # At this point, _var_allprocs_idx_range is correct except for an offset.
        # We apply the global_index offset to make _var_allprocs_idx_range correct.
        for type_ in ['input', 'output']:
            for ind in range(2):
                self._var_allprocs_idx_range[type_][ind] += global_index[type_]

        # Pre-recursion: compute index to pass to subsystems.
        # This index is the number of variables on procs before current proc
        # Necessary because of multiple global counters on different procs
        if self.comm.size > 1:
            subsys0 = self._subsystems_myproc[0]
            for type_ in ['input', 'output']:
                # Note: the following is valid because _var_allprocs_idx_range
                # contains [0, # allprocs vars] at this point because
                # _setup_variables has been run but the recursion
                # for the current method has not been performed yet.
                local_var_size = subsys0._var_allprocs_idx_range[type_][1]

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
                global_index[type_] += np.sum(nvar_allprocs[:iproc + 1]) - nvar_myproc

        # Perform recursion
        for subsys in self._subsystems_myproc:
            subsys._setup_variable_indices(global_index)

        # Reset index dict to the global variable counter on all procs.
        # Necessary for younger siblings to have proper index values.
        for type_ in ['input', 'output']:
            global_index[type_] = self._var_allprocs_idx_range[type_][1]

    def _setup_partials(self):
        """
        Set up partial derivative sparsity structures and approximation schemes.
        """
        for subsys in self._subsystems_myproc:
            subsys._setup_partials()

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
        with self.jacobian_context() as J:
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
        with self.jacobian_context() as J:
            for subsys in self._subsystems_myproc:
                subsys._linearize()

            # Update jacobian
            if self._owns_global_jac:
                J._update()

        if self._nl_solver is not None:
            self._nl_solver._linearize()

        if self._ln_solver is not None:
            self._ln_solver._linearize()
