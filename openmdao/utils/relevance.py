"""
Class definitions for Relevance and related classes.
"""

import sys
from pprint import pprint
from contextlib import contextmanager
from collections import defaultdict
import numpy as np
from openmdao.utils.general_utils import all_ancestors


def _get_seed_map(fwd_seeds, rev_seeds, nvars, nsystems):
    """
    Return a map of fwdseed/revseed pairings to var/sys relevance arrays.

    Parameters
    ----------
    fwd_seeds : iter of str
        Iterator over forward seed variable names.
    rev_seeds : iter of str
        Iterator over reverse seed variable names.
    nvars : int
        Number of variables in the graph.
    nsystems : int
        Number of systems in the graph.

    Returns
    -------
    dict
        Nested dict of the form {fwdseed: {revseed: rel_arrays}}.
    """
    seedmap = {}
    for f in fwd_seeds:
        seedmap[f] = fmap = {}
        for r in rev_seeds:
            fmap[r] = (np.zeros(nvars, dtype=bool),
                       np.zeros(nsystems, dtype=bool))


def get_relevance(model, of, wrt):
    """
    Return a Relevance object for the given design vars, and responses.

    Parameters
    ----------
    model : <Group>
        The top level group in the system hierarchy.
    of : dict
        Dictionary of 'of' variables.  Keys don't matter.
    wrt : dict
        Dictionary of 'wrt' variables.  Keys don't matter.

    Returns
    -------
    Relevance
        Relevance object.
    """
    if not model._use_derivatives or (not of and not wrt):
        # in this case, an permanantly inactive relevance object is returned
        of = {}
        wrt = {}

    return Relevance(model, wrt, of)


class Relevance(object):
    """
    Class that computes relevance based on a data flow graph.

    Parameters
    ----------
    group : <System>
        The top level group in the system hierarchy.
    fwd_meta : dict
        Dictionary of design variable metadata.  Keys don't matter.
    rev_meta : dict
        Dictionary of response variable metadata.  Keys don't matter.

    Attributes
    ----------
    _graph : <nx.DirectedGraph>
        Dependency graph.  Dataflow graph containing both variables and systems.
    _all_vars : dict
        dict of all variables in the graph mapped to the row index into the variable
        relevance array.
    _rel_var_array : ndarray
        Bool array of shape (nvars, nseeds) where nvars is the number of variables in the graph
        and nseeds is the number of unique seed pairs.  If _rel_var_array[i, j] is True, then
        variable i is relevant to seed pair j.
    _rel_sys_array : ndarray
        Bool array of shape (nsystems, nseeds) where nsystems is the number of systems in the graph
        and nseeds is the number of unique seed pairs.  If _rel_sys_array[i, j] is True, then
        system i is relevant to seed pair j.
    _seed_vars : dict
        Maps direction to currently active seed variable names.
    _all_seed_vars : dict
        Maps direction to all seed variable names.
    _active : bool or None
        If True, relevance is active.  If False, relevance is inactive.  If None, relevance is
        uninitialized.
    _use_pre_opt_post : bool
        If True, factor pre_opt_post status into relevance.
    """

    def __init__(self, model, fwd_meta, rev_meta):
        """
        Initialize all attributes.
        """
        assert model.pathname == '', "Relevance can only be initialized on the top level Group."

        self._relevant_vars = {}  # maps (varname, direction) to variable set checker
        self._relevant_systems = {}  # maps (varname, direction) to relevant system sets
        self._active = None  # allow relevance to be turned on later
        self._graph = model._dataflow_graph
        self._use_pre_opt_post = model._problem_meta['group_by_pre_opt_post']

        # seed var(s) for the current derivative operation
        self._seed_vars = {'fwd': frozenset(), 'rev': frozenset()}
        # all seed vars for the entire derivative computation
        self._all_seed_vars = {'fwd': frozenset(), 'rev': frozenset()}

        self._set_all_seeds(model, fwd_meta, rev_meta)

        if not (fwd_meta and rev_meta):
            self._active = False  # relevance will never be active

    def __repr__(self):
        """
        Return a string representation of the Relevance.

        Returns
        -------
        str
            String representation of the Relevance.
        """
        return f"Relevance({self._seed_vars}, active={self._active})"

    def _intersect_arrays(self, arrmap, seeds):
        """
        Return the intersection of the relevance arrays for the given seeds.

        Parameters
        ----------
        arrmap : dict
            dict of the form {seed: (rel_var_array, rel_sys_array)}
        seeds : iter of str
            Iterator over seed variable names.

        Returns
        -------
        tuple
            Tuple of the form (var_array, sys_array) where var_array and sys_array are the
            intersection of the relevance arrays for the given seeds.
        """
        for i, seed in enumerate(seeds):
            varr, sarr = arrmap[seed]
            if i == 0:
                var_array = varr.copy()
                sys_array = sarr.copy()
            else:
                var_array &= varr
                sys_array &= sarr

        return var_array, sys_array

    def _get_single_seeds_map(self, group, seed_meta, direction, all_systems, all_vars):
        """
        Return the relevance arrays for each individual seed.

        Parameters
        ----------
        group : <Group>
            The top level group in the system hierarchy.
        seed_meta : dict
            Dictionary of metadata for the seeds.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
        all_systems : set
            Set of all systems in the graph.
        all_vars : set
            Set of all variables in the graph.

        Returns
        -------
        dict
            Dict of the form {seed: (var_array, sys_array)} where var_array and sys_array are the
            relevance arrays for the given seed.
        bool
            True if any of the seeds use parallel derivative coloring.
        """
        nprocs = group.comm.size
        has_par_derivs = False
        seed_map = {}

        for meta in seed_meta.values():
            src = meta['source']
            local = nprocs > 1 and meta['parallel_deriv_color'] is not None
            has_par_derivs |= local
            depnodes = self._dependent_nodes(src, direction, local=local)
            rel_systems = _vars2systems(depnodes)
            rel_vars = depnodes - all_systems
            varray = np.zeros(len(all_vars), dtype=bool)
            for rel_var in rel_vars:
                varray[self._all_vars[rel_var]] = True
            sarray = np.zeros(len(all_systems), dtype=bool)
            for rel_sys in rel_systems:
                sarray[self._all_systems[rel_sys]] = True
            seed_map[src] = (varray, sarray)

        return seed_map, has_par_derivs

    def _set_all_seeds(self, group, fwd_meta, rev_meta):
        """
        Set the full list of seeds to be used to determine relevance.

        This should only be called once, at __init__ time.

        Parameters
        ----------
        group : <Group>
            The top level group in the system hierarchy.
        fwd_meta : dict
            Dictionary of metadata for forward derivatives.
        rev_meta : dict
            Dictionary of metadata for reverse derivatives.
        """
        fwd_seeds = frozenset([m['source'] for m in fwd_meta.values()])
        rev_seeds = frozenset([m['source'] for m in rev_meta.values()])

        self._seed_rel_dict = _get_seed_map(fwd_seeds, rev_seeds)

        nprocs = group.comm.size
        setup_par_derivs = False

        # this set contains all variables and some or all components
        # in the graph.  Components are included if all of their outputs
        # depend on all of their inputs.
        all_systems = _vars2systems(self._graph.nodes())
        all_vars = {n for n in self._graph.nodes() if n not in all_systems}

        # create mappings of var and system names to indices into the var/system
        # relevance arrays.
        self._all_systems = {n: i for i, n in enumerate(all_systems)}
        self._all_vars = {n: i for i, n in enumerate(all_vars)}

        self._fseed_map, fhas_par_derivs = self._get_single_seeds_map(group, fwd_meta, 'fwd',
                                                                      all_systems, all_vars)
        self._rseed_map, rhas_par_derivs = self._get_single_seeds_map(group, rev_meta, 'rev',
                                                                      all_systems, all_vars)
        has_par_derivs = fhas_par_derivs or rhas_par_derivs

        self._seed_map = seed_map = {}
        for fsrc, fdata in fseed_map.items():
            seed_map[fsrc] = sub = {}
            for rsrc, rdata in rseed_map.items():
                sub[rsrc] = (fdata[0] | rdata[0], fdata[1] | rdata[1])

        # now add entries for each (fseed, all_rseeds) and each (rseed, all_fseeds)
        all_rseed_varray =
        for fsrc, fdata in fseed_map.items():
            seed_map[(fsrc, 'all')] = (fdata[0], fdata[1] | np.any([rdata[1] for rdata in rseed_map.values()], axis=0))

        self._all_seed_vars['fwd'] = self._seed_vars['fwd'] = fwd_seeds
        self._all_seed_vars['rev'] = self._seed_vars['rev'] = rev_seeds

        if has_par_derivs:
            self._setup_par_deriv_relevance(group, rev_meta, fwd_meta)

    @contextmanager
    def active(self, active):
        """
        Context manager for activating/deactivating relevance.

        Parameters
        ----------
        active : bool
            If True, activate relevance.  If False, deactivate relevance.

        Yields
        ------
        None
        """
        if not self._active:  # if already inactive from higher level, don't change it
            yield
        else:
            save = self._active
            self._active = active
            try:
                yield
            finally:
                self._active = save

    def relevant_vars(self, name, direction, inputs=True, outputs=True):
        """
        Return a set of variables relevant to the given dv/response in the given direction.

        Parameters
        ----------
        name : str
            Name of the variable of interest.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
        inputs : bool
            If True, include inputs.
        outputs : bool
            If True, include outputs.

        Returns
        -------
        set
            Set of the relevant variables.
        """
        if inputs and outputs:
            return self._relevant_vars[name, direction].to_set()
        elif inputs:
            return self._apply_filter(self._relevant_vars[name, direction], _is_input)
        elif outputs:
            return self._apply_filter(self._relevant_vars[name, direction], _is_output)
        else:
            return set()

    @contextmanager
    def all_seeds_active(self, active=True):
        """
        Context manager where all seeds are active.

        This assumes that the relevance object itself is active.

        Parameters
        ----------
        active : bool
            If True, assuming relevance is already active, activate all seeds, else do nothing.

        Yields
        ------
        None
        """
        # if already inactive from higher level, or 'active' parameter is False, don't change it
        if not active or self._active is False:
            yield
        else:
            save = {'fwd': self._seed_vars['fwd'], 'rev': self._seed_vars['rev']}
            save_active = self._active
            self._active = True
            self.reset_to_all_seeds()
            try:
                yield
            finally:
                self._seed_vars = save
                self._active = save_active

    @contextmanager
    def seeds_active(self, fwd_seeds=None, rev_seeds=None, local=False):
        """
        Context manager where the specified seeds are active.

        This assumes that the relevance object itself is active.

        Parameters
        ----------
        fwd_seeds : iter of str or None
            Iterator over forward seed variable names. If None use current active seeds.
        rev_seeds : iter of str or None
            Iterator over reverse seed variable names. If None use current active seeds.
        local : bool
            If True, include only local variables.

        Yields
        ------
        None
        """
        if self._active is False:  # if already inactive from higher level, don't change anything
            yield
        else:
            save = {'fwd': self._seed_vars['fwd'], 'rev': self._seed_vars['rev']}
            save_active = self._active
            self._active = True
            self._set_seeds(fwd_seeds, rev_seeds, local)
            try:
                yield
            finally:
                self._seed_vars = save
                self._active = save_active

    def _setup_par_deriv_relevance(self, group, responses, desvars):
        pd_err_chk = defaultdict(dict)
        mode = group._problem_meta['mode']  # 'fwd', 'rev', or 'auto'

        if mode in ('fwd', 'auto'):
            for desvar, response, relset in self.iter_seed_pair_relevance(inputs=True):
                if desvar in desvars and self._graph.nodes[desvar]['local']:
                    dvcolor = desvars[desvar]['parallel_deriv_color']
                    if dvcolor:
                        pd_err_chk[dvcolor][desvar] = relset

        if mode in ('rev', 'auto'):
            for desvar, response, relset in self.iter_seed_pair_relevance(outputs=True):
                if response in responses and self._graph.nodes[response]['local']:
                    rescolor = responses[response]['parallel_deriv_color']
                    if rescolor:
                        pd_err_chk[rescolor][response] = relset

        # check to make sure we don't have any overlapping dependencies between vars of the
        # same color
        errs = {}
        for pdcolor, dct in pd_err_chk.items():
            for vname, relset in dct.items():
                for n, nds in dct.items():
                    if vname != n and relset.intersection(nds):
                        if pdcolor not in errs:
                            errs[pdcolor] = []
                        errs[pdcolor].append(vname)

        all_errs = group.comm.allgather(errs)
        msg = []
        for errdct in all_errs:
            for color, names in errdct.items():
                vtype = 'design variable' if mode == 'fwd' else 'response'
                msg.append(f"Parallel derivative color '{color}' has {vtype}s "
                           f"{sorted(names)} with overlapping dependencies on the same rank.")

        if msg:
            raise RuntimeError('\n'.join(msg))

    def _set_seeds(self, fwd_seeds, rev_seeds, local=False):
        """
        Set the seed(s) to determine relevance for a given variable in a given direction.

        Parameters
        ----------
        fwd_seeds : iter of str or None
            Iterator over forward seed variable names. If None use current active seeds.
        rev_seeds : iter of str or None
            Iterator over reverse seed variable names. If None use current active seeds.
        local : bool
            If True, update relevance set if necessary to include only local variables.
        """
        if fwd_seeds:
            fwd_seeds = frozenset(fwd_seeds)
        else:
            fwd_seeds = self._all_seed_vars['fwd']

        if rev_seeds:
            rev_seeds = frozenset(rev_seeds)
        else:
            rev_seeds = self._all_seed_vars['rev']

        self._seed_vars['fwd'] = fwd_seeds
        self._seed_vars['rev'] = rev_seeds

    def reset_to_all_seeds(self):
        """
        Reset the seeds to the full set of seeds.
        """
        self._seed_vars['fwd'] = self._all_seed_vars['fwd']
        self._seed_vars['rev'] = self._all_seed_vars['rev']

    def is_relevant(self, name):
        """
        Return True if the given variable is relevant.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        bool
            True if the given variable is relevant.
        """
        if not self._active:
            return True

        for seed in self._seed_vars['fwd']:
            if name in self._relevant_vars[seed, 'fwd']:
                for tgt in self._seed_vars['rev']:
                    if name in self._relevant_vars[tgt, 'rev']:
                        return True

        return False

    def is_relevant_system(self, name):
        """
        Return True if the given named system is relevant.

        Parameters
        ----------
        name : str
            Name of the System.

        Returns
        -------
        bool
            True if the given system is relevant.
        """
        if not self._active:
            return True

        for seed in self._seed_vars['fwd']:
            if name in self._relevant_systems[seed, 'fwd']:
                for tgt in self._seed_vars['rev']:
                    if name in self._relevant_systems[tgt, 'rev']:
                        return True
        return False

    def system_filter(self, systems, relevant=True, linear=True):
        """
        Filter the given iterator of systems to only include those that are relevant.

        Parameters
        ----------
        systems : iter of Systems
            Iterator over systems.
        relevant : bool
            If True, return only relevant systems.  If False, return only irrelevant systems.
        linear : bool
            If True, use linear relevance, which can be less conservative than nonlinear relevance
            if group_by_pre_opt_post is True at Problem level.

        Yields
        ------
        System
            Relevant system.
        """
        if self._active:
            for system in systems:
                if relevant == self.is_relevant_system(system.pathname):
                    yield system
                # if grouping by pre_opt_post and we're doing some nonlinear operation, the
                # 'systems' list being passed in has already been filtered by pre_opt_post status.
                # We have to respect that status here (for nonlinear) to avoid skipping components
                # that have the 'always_opt' option set.
                elif relevant and not linear and self._use_pre_opt_post:
                    yield system
        elif relevant:
            yield from systems

    def _init_relevance_set(self, varname, direction, local=False):
        """
        Return a SetChecker for variables and components for the given variable.

        The SetChecker determines all relevant variables/systems found in the
        relevance graph starting at the given variable and moving in the given
        direction. It is determined lazily and cached for future use.

        Parameters
        ----------
        varname : str
            Name of the variable.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
            'fwd' will find downstream nodes, 'rev' will find upstream nodes.
        local : bool
            If True, update relevance set if necessary to include only local variables.
        """
        key = (varname, direction)
        if key not in self._relevant_vars:
            assert direction in ('fwd', 'rev'), "direction must be 'fwd' or 'rev'"

            # first time we've seen this varname/direction pair, so we need to
            # compute the set of relevant variables and the set of relevant systems
            # and store them for future use.
            depnodes = self._dependent_nodes(varname, direction, local=local)

            rel_systems = _vars2systems(depnodes)

            rel_vars = depnodes - self._all_systems

            self._relevant_systems[key] = _get_set_checker(rel_systems, self._all_systems)
            self._relevant_vars[key] = _get_set_checker(rel_vars, self._all_vars)

    def iter_seed_pair_relevance(self, fwd_seeds=None, rev_seeds=None, inputs=False, outputs=False):
        """
        Yield all relevant variables for each pair of seeds.

        Parameters
        ----------
        fwd_seeds : iter of str or None
            Iterator over forward seed variable names. If None use current registered seeds.
        rev_seeds : iter of str or None
            Iterator over reverse seed variable names. If None use current registered seeds.
        inputs : bool
            If True, include inputs.
        outputs : bool
            If True, include outputs.

        Yields
        ------
        set
            Set of names of relevant variables.
        """
        filt = _get_io_filter(inputs, outputs)
        if filt is True:  # everything is filtered out
            return

        if fwd_seeds is None:
            fwd_seeds = self._seed_vars['fwd']
        if rev_seeds is None:
            rev_seeds = self._seed_vars['rev']

        if isinstance(fwd_seeds, str):
            fwd_seeds = [fwd_seeds]
        if isinstance(rev_seeds, str):
            rev_seeds = [rev_seeds]

        for seed in fwd_seeds:
            fwdvars = self._relevant_vars[seed, 'fwd'].to_set()
            for rseed in rev_seeds:
                if rseed in fwdvars:
                    inter = self._relevant_vars[rseed, 'rev'].intersection(fwdvars)
                    if inter:
                        yield seed, rseed, self._apply_filter(inter, filt)

    def _apply_filter(self, names, filt):
        """
        Return only the nodes from the given set of nodes that pass the given filter.

        Parameters
        ----------
        names : iter of str
            Iterator of node names.
        filt : callable
            Filter function taking a graph node as an argument and returning True if the node
            should be included in the output.  If True, no filtering is done.  If False, the
            returned set will be empty.

        Returns
        -------
        set
            Set of node names that passed the filter.
        """
        if not filt:  # no filtering needed
            if isinstance(names, set):
                return names
            return set(names)
        elif filt is True:
            return set()

        # filt is a function.  Apply it to named graph nodes.
        return set(self._filter_nodes_iter(names, filt))

    def _filter_nodes_iter(self, names, filt):
        """
        Return only the nodes from the given set of nodes that pass the given filter.

        Parameters
        ----------
        names : iter of str
            Iterator over node names.
        filt : callable
            Filter function taking a graph node as an argument and returning True if the node
            should be included in the output.

        Yields
        ------
        str
            Node name that passed the filter.
        """
        nodes = self._graph.nodes
        for n in names:
            if filt(nodes[n]):
                yield n

    def _all_relevant(self, fwd_seeds, rev_seeds, inputs=True, outputs=True):
        """
        Return all relevant inputs, outputs, and systems for the given seeds.

        This is primarily used as a convenience function for testing and is not particularly
        efficient.

        Parameters
        ----------
        fwd_seeds : iter of str
            Iterator over forward seed variable names.
        rev_seeds : iter of str
            Iterator over reverse seed variable names.
        inputs : bool
            If True, include inputs.
        outputs : bool
            If True, include outputs.

        Returns
        -------
        tuple
            (set of relevant inputs, set of relevant outputs, set of relevant systems)
            If a given inputs/outputs is False, the corresponding set will be empty. The
            returned systems will be the set of all systems containing any
            relevant variables based on the values of inputs and outputs, i.e. if outputs is False,
            the returned systems will be the set of all systems containing any relevant inputs.
        """
        relevant_vars = set()
        for _, _, relvars in self.iter_seed_pair_relevance(fwd_seeds, rev_seeds, inputs, outputs):
            relevant_vars.update(relvars)
        relevant_systems = _vars2systems(relevant_vars)

        inputs = set(self._filter_nodes_iter(relevant_vars, _is_input))
        outputs = set(self._filter_nodes_iter(relevant_vars, _is_output))

        return inputs, outputs, relevant_systems

    def _dependent_nodes(self, start, direction, local=False):
        """
        Return set of all connected nodes in the given direction starting at the given node.

        Parameters
        ----------
        start : str
            Name of the starting node.
        direction : str
            If 'fwd', traverse downstream.  If 'rev', traverse upstream.
        local : bool
            If True, include only local variables.

        Returns
        -------
        set
            Set of all dependent nodes.
        """
        if start in self._graph:
            if local and not self._graph.nodes[start]['local']:
                return set()
            stack = [start]
            visited = {start}

            if direction == 'fwd':
                fnext = self._graph.successors
            elif direction == 'rev':
                fnext = self._graph.predecessors
            else:
                raise ValueError("direction must be 'fwd' or 'rev'")

            while stack:
                src = stack.pop()
                for tgt in fnext(src):
                    if tgt not in visited:
                        if local:
                            node = self._graph.nodes[tgt]
                            if 'local' in node and not node['local']:
                                return visited

                        visited.add(tgt)
                        stack.append(tgt)

            return visited

        return set()

    def dump(self, out_stream=sys.stdout):
        """
        Print out the current relevance information.

        Parameters
        ----------
        out_stream : file-like or None
            Where to send human readable output.  Defaults to sys.stdout.
        """
        print("Systems:", file=out_stream)
        pprint(self._relevant_systems, stream=out_stream)
        print("Variables:", file=out_stream)
        pprint(self._relevant_vars, stream=out_stream)


def _vars2systems(nameiter):
    """
    Return a set of all systems containing the given variables or components.

    This includes all ancestors of each system, including ''.

    Parameters
    ----------
    nameiter : iter of str
        Iterator of variable or component pathnames.

    Returns
    -------
    set
        Set of system pathnames.
    """
    systems = {''}  # root group is always there
    for name in nameiter:
        sysname = name.rpartition('.')[0]
        if sysname not in systems:
            systems.update(all_ancestors(sysname))

    return systems


def _get_set_checker(relset, allset):
    """
    Return a SetChecker for the given sets.

    The SetChecker will be inverted if that will use less memory than a non-inverted checker.

    Parameters
    ----------
    relset : set
        Set of relevant items.
    allset : set
        Set of all items.

    Returns
    -------
    SetChecker, InverseSetChecker
        Set checker for the given sets.
    """
    if len(allset) == len(relset):
        return SetChecker(set(), allset, invert=True)

    nrel = len(relset)

    # store whichever type of checker will use the least memory
    if nrel < (len(allset) - nrel):
        return SetChecker(relset)
    else:
        return SetChecker(allset - relset, allset, invert=True)


def _get_io_filter(inputs, outputs):
    if inputs and outputs:
        return False  # no filtering needed
    elif inputs:
        return _is_input
    elif outputs:
        return _is_output
    else:
        return True  # filter out everything


def _is_input(node):
    return node['type_'] == 'input'


def _is_output(node):
    return node['type_'] == 'output'
