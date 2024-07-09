"""Define the ParallelGroup class."""

from openmdao.core.group import Group
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.graph_utils import get_sccs_topo


class ParallelGroup(Group):
    """
    Class used to group systems together to be executed in parallel.

    Parameters
    ----------
    **kwargs : dict
        Dict of arguments available here and in all descendants of this Group.
    """

    def __init__(self, **kwargs):
        """
        Set the mpi_proc_allocator option to 'parallel'.
        """
        super().__init__(**kwargs)
        self._mpi_proc_allocator.parallel = True

    def _configure(self):
        """
        Configure our model recursively to assign any children settings.

        Highest system's settings take precedence.
        """
        super()._configure()
        if self.comm.size > 1:
            self._has_guess = any(self.comm.allgather(self._has_guess))

    def _get_sys_promotion_tree(self, tree):
        tree = super()._get_sys_promotion_tree(tree)

        if self.comm.size > 1:
            prefix = self.pathname + '.' if self.pathname else ''
            subtree = {n: data for n, data in tree.items() if n.startswith(prefix)}
            for sub in self.comm.allgather(subtree):  # TODO: make this more efficient
                for n, data in sub.items():
                    if n not in tree:
                        tree[n] = data

        return tree

    def _ordered_comp_name_iter(self):
        """
        Yield contained component pathnames in order of execution.

        For components within ParallelGroups, true execution order is unknown so components
        will be ordered by rank within a ParallelGroup.
        """
        if self.comm.size > 1:
            names = []
            for s in self._subsystems_myproc:
                if isinstance(s, Group):
                    names.extend(s._ordered_comp_name_iter())
                else:
                    names.append(s.pathname)
            seen = set()
            for ranknames in self.comm.allgather(names):
                for name in ranknames:
                    if name not in seen:
                        yield name
                        seen.add(name)
        else:
            yield from super()._ordered_comp_name_iter()

    def _check_order(self):
        """
        Check if auto ordering is needed.

        Returns
        -------
        dict
            Lists of out-of-order connections.
        """
        if self.options['auto_order']:
            issue_warning("auto_order is not supported in ParallelGroup. "
                          "Ignoring auto_order option.", prefix=self.msginfo)

        return {}

    def comm_info_iter(self):
        """
        Yield comm size and rank for this system and all subsystems.

        Yields
        ------
        tuple
            A tuple of the form (abs_name, comm_size).
        """
        if self.comm.size > 1:
            for info in self.comm.allgather(list(super().comm_info_iter())):
                yield from info
        else:
            yield from super().comm_info_iter()

    def _declared_partials_iter(self):
        """
        Iterate over all declared partials.

        Yields
        ------
        key : tuple (of, wrt)
            Subjacobian key.
        """
        if self.comm.size > 1:
            if self._gather_full_data():
                gathered = self.comm.allgather(list(self._subjacs_info.keys()))
            else:
                gathered = self.comm.allgather([])
            seen = set()
            for keylist in gathered:
                for key in keylist:
                    if key not in seen:
                        yield key
                        seen.add(key)
        else:
            yield from super()._declared_partials_iter()

    def _get_missing_partials(self, missing):
        """
        Store information about any missing partial derivatives in the 'missing' dict.

        Parameters
        ----------
        missing : dict
            Dictionary containing list of missing derivatives (of, wrt) keyed by system pathname.
        """
        if self.comm.size > 1:
            msng = {}
            super()._get_missing_partials(msng)
            if self._gather_full_data():
                gathered = self.comm.allgather(msng)
            else:
                gathered = self.comm.allgather({})
            seen = set()
            for rankdict in gathered:
                for sysname, mset in rankdict.items():
                    if sysname not in seen:
                        missing[sysname] = mset
                        seen.add(sysname)
        else:
            super()._get_missing_partials(missing)

    def _get_relevance_modifiers(self, grad_groups, always_opt_comps):
        """
        Collect information from the model that will modify the relevance graph of the model.

        Parameters
        ----------
        grad_groups : set
            Set of groups having nonlinear solvers that use gradients.
        always_opt_comps : set
            Set of components that are to be included in every iteration of the optimization,
            even if they aren't relevant in terms of data flow.
        """
        if self.comm.size > 1:
            gg = set()
            aoc = set()
            super()._get_relevance_modifiers(gg, aoc)
            if self._gather_full_data():
                gathered = self.comm.allgather((gg, aoc))
            else:
                gathered = self.comm.allgather((set(), set()))

            for g, a in gathered:
                grad_groups.update(g)
                always_opt_comps.update(a)
        else:
            super()._get_relevance_modifiers(grad_groups, always_opt_comps)

    def _setup_ordering(self, parent):
        if self.comm.size > 1:
            return self.comm.allreduce(super()._setup_ordering(parent))
        else:
            return super()._setup_ordering(parent)

    def _update_data_order(self, parent=None):
        if self.comms.size > 1:
            allprocs_abs2meta_keys = {'input': [], 'output': []}
            if self._gather_full_data():
                for io in ('input', 'output'):
                    for subsys in self._subsystems_myproc:
                        allprocs_abs2meta_keys[io].extend(subsys._var_allprocs_abs2meta[io])

            old_allprocs_abs2meta = self._var_allprocs_abs2meta
            self._var_allprocs_abs2meta = {'input': {}, 'output': {}}

            # get ordering for var_allprocs_abs2meta keys from all procs.  The metadata has
            # already been gathered in _setup_var_data and it just needs to be reordered.
            for proc_keys in self.comm.allgather(allprocs_abs2meta_keys):
                for io, keylist in proc_keys.items():
                    abs2meta = self._var_allprocs_abs2meta[io]
                    old = old_allprocs_abs2meta[io]
                    for k in keylist:
                        abs2meta[k] = old[k]

            self._var_abs2meta = {'input': {}, 'output': {}}
            self._var_allprocs_abs2idx = {'input': {}, 'output': {}}
            for io in ('input', 'output'):
                abs2meta = self._var_abs2meta[io]
                for subsys in self._subsystems_myproc:
                    abs2meta.update(subsys._var_abs2meta[io])

                self._var_allprocs_abs2idx[io].update(
                    {n: i for i, n in enumerate(self._var_allprocs_abs2meta[io])})

            self._reordered = False
        else:
            super()._update_data_order(parent)

    def iter_group_sccs(self, recurse=True, use_abs_names=True, all_groups=False):
        """
        Yield strongly connected components of the group's subsystem graph.

        Parameters
        ----------
        recurse : bool
            If True, recurse into subgroups.
        use_abs_names : bool
            If True, return absolute names, otherwise return relative names.
        all_groups : bool
            If True, yield all groups, not just those with one or more cycles.

        Yields
        ------
        str, list of sets of str
            Group pathname and list of sets of subsystems in any strongly connected components
            in this Group.
        """
        if self.comm.size > 1:
            lst = list(super().iter_group_sccs(recurse=recurse, use_abs_names=use_abs_names,
                                               all_groups=all_groups))
            if self._gather_full_data():
                gathered = self.comm.allgather(lst)
            else:
                gathered = self.comm.allgather([])

            for ranklist in gathered:
                for tup in ranklist:
                    yield tup
        else:
            yield from super().iter_group_sccs(recurse=recurse, use_abs_names=use_abs_names,
                                               all_groups=all_groups)

    def all_system_visitor(self, func, predicate=None, recurse=True, include_self=True):
        """
        Apply a function to all subsystems.

        Parameters
        ----------
        func : callable
            A callable that takes a System as its only argument.
        predicate : callable or None
            A callable that takes a System as its only argument and returns -1, 0, or 1.
            If it returns 1, apply the function to the system.
            If it returns 0, don't apply the function, but continue on to the system's subsystems.
            If it returns -1, don't apply the function and don't continue on to the system's
            subsystems.
            If predicate is None, the function is always applied.
        recurse : bool
            If True, function is applied to all subsystems of subsystems.
        include_self : bool
            If True, apply the function to the System itself.

        Yields
        ------
        object
            The result of the function called on each system.
        """
        if self.comm.size > 1:
            lst = list(super().all_system_visitor(func, predicate, recurse=recurse,
                                                  include_self=include_self))
            if self._gather_full_data():
                gathered = self.comm.allgather(lst)
            else:
                gathered = self.comm.allgather([])

            for ranklist in gathered:
                for tup in ranklist:
                    yield tup
        else:
            yield from super().all_system_visitor(func, predicate, recurse=recurse,
                                                  include_self=include_self)
