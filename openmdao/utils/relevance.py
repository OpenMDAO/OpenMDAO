"""
Class definitions for Relevance and related classes.
"""

from contextlib import contextmanager
from collections import defaultdict
import atexit

import numpy as np

from openmdao.utils.general_utils import all_ancestors, _contains_all, get_rev_conns
from openmdao.utils.graph_utils import get_sccs_topo
from openmdao.utils.array_utils import array_hash
from openmdao.utils.om_warnings import issue_warning


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
        # in this case, a permanently inactive relevance object is returned
        # (so the contents of 'of' and 'wrt' don't matter). Make them empty to avoid
        # unnecessary setup.
        of = {}
        wrt = {}

    key = (id(model), tuple(sorted(wrt)), tuple(sorted(of)))
    cache = model._problem_meta['relevance_cache']
    if key in cache:
        return cache[key]

    relevance = Relevance(model, wrt, of, model._problem_meta['rel_array_cache'])
    cache[key] = relevance
    return relevance


class Relevance(object):
    """
    Class that computes relevance based on a data flow graph.

    It determines current relevance based on the current set of forward and reverse seed variables.
    Initial relevance is determined by starting at a given seed and traversing the data flow graph
    in the specified direction to find all relevant variables and systems.  That information is
    then represented as a boolean array where True means the variable or system is relevant to the
    seed.  Relevance with respect to groups of seeds, for example, one forward seed vs. all reverse
    seeds, is determined by combining the boolean relevance arrays for the individual seeds in the
    following manner:  (fwd_array1 | fwd_array2 | ...) & (rev_array1 | rev_array2 | ...). In other
    words, the union of the fwd arrays is intersected with the union of the rev arrays.

    The full set of fwd and rev seeds must be set at initialization time.  At any point after that,
    the set of active seeds can be changed using the set_seeds method, but those seeds must be
    subsets of the full set of seeds.

    Parameters
    ----------
    model : <Group>
        The top level group in the system hierarchy.
    fwd_meta : dict
        Dictionary of design variable metadata.  Keys don't matter.
    rev_meta : dict
        Dictionary of response variable metadata.  Keys don't matter.
    rel_array_cache : dict
        Cache of relevance arrays stored by array hash.

    Attributes
    ----------
    _graph : <nx.DirectedGraph>
        Dependency graph.  Dataflow graph containing both variables and systems.
    _var2idx : dict
        dict of all variables in the graph mapped to the row index into the variable
        relevance array.
    _sys2idx : dict
        dict of all systems in the graph mapped to the row index into the system
        relevance array.
    _seed_vars : dict
        Maps direction to currently active seed variable names.
    _all_seed_vars : dict
        Maps direction to all seed variable names.
    _active : bool or None
        If True, relevance is active.  If False, relevance is inactive.  If None, relevance is
        uninitialized.
    _seed_var_map : dict
        Nested dict of the form {fwdseed(s): {revseed(s): var_array, ...}}.
        Keys that contain multiple seeds are sorted tuples of seed names.
    _seed_sys_map : dict
        Nested dict of the form {fwdseed(s): {revseed(s): sys_array, ...}}.
        Keys that contain multiple seeds are sorted tuples of seed names.
    _single_seed2relvars : dict
        Dict of the form {'fwd': {seed: var_array}, 'rev': ...} where each seed is a
        key and var_array is the variable relevance array for the given seed.
    _single_seed2relsys : dict
        Dict of the form {'fwd': {seed: sys_array}, 'rev': ...} where each seed is a
        key and var_array is the system relevance array for the given seed.
    _nonlinear_sets : dict
        Dict of the form {'pre': pre_rel_array, 'iter': iter_rel_array, 'post': post_rel_array}.
    _current_rel_varray : ndarray
        Array representing the variable relevance for the currently active seeds.
    _current_rel_sarray : ndarray
        Array representing the system relevance for the currently active seeds.
    _rel_array_cache : dict
        Cache of relevance arrays stored by array hash.
    _no_dv_responses : list
        List of responses that have no relevant design variables.
    _redundant_adjoint_systems : set or None
        Set of systems that may benefit from caching RHS arrays and solutions to avoid some linear
        solves.
    _seed_cache : dict
        Maps seed variable names to the source of the seed.
    _rel_array_cache : dict
        Cache of relevance arrays stored by array hash.
    """

    def __init__(self, model, fwd_meta, rev_meta, rel_array_cache):
        """
        Initialize all attributes.
        """
        assert model.pathname == '', "Relevance can only be initialized on the top level Group."

        self._active = None  # allow relevance to be turned on later
        self._rel_array_cache = rel_array_cache
        self._graph = model._dataflow_graph
        self._rel_array_cache = {}
        self._no_dv_responses = []
        self._redundant_adjoint_systems = None
        self._seed_cache = {}

        # seed var(s) for the current derivative operation
        self._seed_vars = {'fwd': (), 'rev': ()}
        # all seed vars for the entire derivative computation
        self._all_seed_vars = {'fwd': (), 'rev': ()}

        self._set_all_seeds(model, fwd_meta, rev_meta)

        self._current_rel_varray = None
        self._current_rel_sarray = None

        self._setup_nonlinear_relevance(model, fwd_meta, rev_meta)

        # _pre_components and _post_components will be empty unless the user has set the
        # 'group_by_pre_opt_post' option to True in the Problem.
        if model._pre_components or model._post_components:
            self._setup_nonlinear_sets(model)
        else:
            self._nonlinear_sets = {}

        # setting _active to False here will permanantly disable relevance checking for this
        # relevance object.  The only way to *temporarily* disable relevance is to use the
        # active() context manager.
        if not (fwd_meta and rev_meta):
            self._active = False

    def __repr__(self):
        """
        Return a string representation of the Relevance.

        Returns
        -------
        str
            String representation of the Relevance.
        """
        return f"Relevance({self._seed_vars}, active={self._active})"

    def _to_seed(self, names):
        """
        Return the seed from the given iter of names.

        Cache the given names iter if it is hashable.

        Parameters
        ----------
        names : iter of str
            Iterator over names.

        Returns
        -------
        tuple
            Key tuple for the given names.
        """
        try:
            return self._seed_cache[names]
        except TypeError:  # names is not hashable
            issue_warning("Relevance seeds should be hashable, but the following seed is not: "
                          f"{names}. It will be converted to a hashable form, but this could "
                          "cause performance issues.", category=RuntimeWarning)
            hashable = False
        except KeyError:  # names is not in the cache
            hashable = names

        try:
            names = [self._seed_cache[n] for n in names]
        except KeyError:
            raise KeyError(f"One or more of the relevance seeds '{names}' is invalid.")

        seeds = tuple(sorted(names))

        if hashable:
            self._seed_cache[hashable] = seeds

        return seeds

    def _get_cached_array(self, arr):
        """
        Return the cached array if it exists, otherwise return the input array after caching it.

        Parameters
        ----------
        arr : ndarray
            Array to be cached.

        Returns
        -------
        ndarray
            Cached array if it exists, otherwise the input array.
        """
        hash = array_hash(arr)
        if hash in self._rel_array_cache:
            return self._rel_array_cache[hash]
        else:
            self._rel_array_cache[hash] = arr

        return arr

    def _setup_nonlinear_sets(self, model):
        """
        Set up the nonlinear sets for relevance checking.

        Parameters
        ----------
        model : <Group>
            The top level group in the system hierarchy.
        """
        pre_systems = set()
        for compname in model._pre_components:
            pre_systems.update(all_ancestors(compname))
        if pre_systems:
            pre_systems.add('')  # include top level group

        post_systems = set()
        for compname in model._post_components:
            post_systems.update(all_ancestors(compname))
        if post_systems:
            post_systems.add('')

        pre_array = self._sys2rel_array(pre_systems)
        post_array = self._sys2rel_array(post_systems)

        if model._iterated_components is _contains_all:
            iter_array = np.ones(len(self._all_systems), dtype=bool)
        else:
            iter_systems = set()
            for compname in model._iterated_components:
                iter_systems.update(all_ancestors(compname))
            if iter_systems:
                iter_systems.add('')

            iter_array = self._sys2rel_array(iter_systems)

        self._nonlinear_sets = {'pre': pre_array, 'iter': iter_array, 'post': post_array}

    def _single_seed_array_iter(self, group, seed_meta, direction, all_systems):
        """
        Yield the relevance arrays for each individual seed and direction for variables and systems.

        The relevance arrays are boolean ndarrays of length nvars and nsystems, respectively.
        All of the variables and systems in the graph map to an index into these arrays and
        if the value at that index is True, then the variable or system is relevant to the seed.

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

        Yields
        ------
        str
            Name of the seed variable.
        bool
            True if the seed uses parallel derivative coloring.
        ndarray
            Boolean relevance array for the variables.
        ndarray
            Boolean relevance array for the systems.
        """
        nprocs = group.comm.size

        for meta in seed_meta.values():
            src = meta['source']
            local = nprocs > 1 and meta['parallel_deriv_color'] is not None
            if local:
                if src in group._var_abs2meta['output']:  # src is local
                    depnodes = self._dependent_nodes(src, direction, local=local)
                    group.comm.bcast(depnodes, root=group._owning_rank[src])
                else:
                    depnodes = group.comm.bcast(None, root=group._owning_rank[src])
            else:
                depnodes = self._dependent_nodes(src, direction, local=local)

            rel_systems = _vars2systems(depnodes)
            rel_vars = depnodes - all_systems

            yield (src, local, self._vars2rel_array(rel_vars), self._sys2rel_array(rel_systems))

    def _vars2rel_array(self, vars):
        """
        Return a relevance array for the given variables.

        Parameters
        ----------
        vars : iter of str
            Iterator over variable names.

        Returns
        -------
        ndarray
            Boolean relevance array.  True means name is relevant.
        """
        return self._names2rel_array(vars, self._var2idx)

    def _sys2rel_array(self, systems):
        """
        Return a relevance array for the given systems.

        Parameters
        ----------
        systems : iter of str
            Iterator over system names.

        Returns
        -------
        ndarray
            Boolean relevance array.  True means name is relevant.
        """
        return self._names2rel_array(systems, self._sys2idx)

    def _names2rel_array(self, names, names2inds):
        """
        Return a relevance array for the given names.

        Parameters
        ----------
        names : iter of str
            Iterator over names.
        names2inds : dict
            Dict of the form {name: index} where index is the index into the relevance array.

        Returns
        -------
        ndarray
            Boolean relevance array.  True means name is relevant.
        """
        rel_array = np.zeros(len(names2inds), dtype=bool)
        rel_array[[names2inds[n] for n in names]] = True

        return self._get_cached_array(rel_array)

    def _combine_relevance(self, fmap, fwd_seeds, rmap, rev_seeds):
        """
        Return the combined relevance arrays for the given seeds.

        Parameters
        ----------
        fmap : dict
            Dict of the form {seed: array} where array is the
            relevance arrays for the given seed.
        fwd_seeds : iter of str
            Iterator over forward seed variable names.
        rmap : dict
            Dict of the form {seed: array} where array is the
            relevance arrays for the given seed.
        rev_seeds : iter of str
            Iterator over reverse seed variable names.

        Returns
        -------
        ndarray
            Array representing the combined relevance arrays for the given seeds.
            The arrays are combined by taking the intersection of the relevance arrays for
            each fwd_seed/rev_seed pair and taking the union of each of those results.
        """
        combined = None
        for fseed in fwd_seeds:
            farr = fmap[fseed]
            for rseed in rev_seeds:
                if combined is None:
                    combined = farr & rmap[rseed]
                else:
                    combined |= (farr & rmap[rseed])

        return np.zeros(0, dtype=bool) if combined is None else self._get_cached_array(combined)

    def rel_vars_iter(self, rel_array, relevant=True):
        """
        Return an iterator of relevant variable names.

        Parameters
        ----------
        rel_array : ndarray
            Boolean relevance array.  True means name is relevant.
        relevant : bool
            If True, return only relevant names.  If False, return only irrelevant names.

        Yields
        ------
        str
            Name of the relevant variable.
        """
        yield from self._rel_names_iter(rel_array, self._var2idx, relevant)

    def _rel_names_iter(self, rel_array, all_names, relevant=True):
        """
        Return an iterator of names from the given relevance array.

        Parameters
        ----------
        rel_array : ndarray
            Boolean relevance array.  True means name is relevant.
        all_names : iter of str
            Iterator over the full set of names from the graph, either variables or systems.
        relevant : bool
            If True, return only relevant names.  If False, return only irrelevant names.

        Yields
        ------
        str
            Name from the given relevance array.
        """
        for n, rel in zip(all_names, rel_array):
            if rel == relevant:
                yield n

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
        fwd_seeds = []
        rev_seeds = []
        for name, meta in fwd_meta.items():
            src = meta['source']
            self._seed_cache[name] = src
            self._seed_cache[src] = src
            fwd_seeds.append(src)

        for name, meta in rev_meta.items():
            src = meta['source']
            self._seed_cache[name] = src
            self._seed_cache[src] = src
            rev_seeds.append(src)

        fwd_seeds = self._to_seed(tuple(fwd_seeds))
        rev_seeds = self._to_seed(tuple(rev_seeds))

        self._seed_var_map = seed_var_map = {}
        self._seed_sys_map = seed_sys_map = {}

        self._current_var_array = np.zeros(0, dtype=bool)
        self._current_sys_array = np.zeros(0, dtype=bool)

        self._all_seed_vars['fwd'] = fwd_seeds
        self._all_seed_vars['rev'] = rev_seeds

        self._single_seed2relvars = {'fwd': {}, 'rev': {}}
        self._single_seed2relsys = {'fwd': {}, 'rev': {}}

        if not fwd_meta or not rev_meta:
            return

        # this set contains all variables and some or all components
        # in the graph.  Components are included if all of their outputs
        # depend on all of their inputs.
        all_vars = set()
        all_systems = {''}
        for node, data in self._graph.nodes(data=True):
            if 'type_' in data:
                all_vars.add(node)
                sysname = node.rpartition('.')[0]
                if sysname not in all_systems:
                    all_systems.update(all_ancestors(sysname))
            elif node not in all_systems:
                all_systems.update(all_ancestors(node))

        self._all_systems = all_systems
        all_vars = sorted(all_vars)

        # create mappings of var and system names to indices into the var/system
        # relevance arrays.
        self._sys2idx = {n: i for i, n in enumerate(sorted(all_systems))}
        self._var2idx = {n: i for i, n in enumerate(sorted(all_vars))}

        meta = {'fwd': fwd_meta, 'rev': rev_meta}

        # map each seed to its variable and system relevance arrays
        has_par_derivs = {}
        for io in ('fwd', 'rev'):
            for seed, local, var_array, sys_array in self._single_seed_array_iter(group, meta[io],
                                                                                  io, all_systems):
                self._single_seed2relvars[io][seed] = self._get_cached_array(var_array)
                self._single_seed2relsys[io][seed] = self._get_cached_array(sys_array)
                if local:
                    has_par_derivs[seed] = io

        # in seed_map, add keys for both fseed and (fseed,) and similarly for rseed
        # because both forms of keys may be used depending on the context.
        for fseed, fvarr in self._single_seed2relvars['fwd'].items():
            fsarr = self._single_seed2relsys['fwd'][fseed]
            seed_var_map[fseed] = seed_var_map[(fseed,)] = vsub = {}
            seed_sys_map[fseed] = seed_sys_map[(fseed,)] = ssub = {}
            for rseed, rvarr in self._single_seed2relvars['rev'].items():
                rsysarr = self._single_seed2relsys['rev'][rseed]
                vsub[rseed] = vsub[(rseed,)] = self._get_cached_array(fvarr & rvarr)
                ssub[rseed] = ssub[(rseed,)] = self._get_cached_array(fsarr & rsysarr)

        # now add entries for each (fseed, all_rseeds) and each (all_fseeds, rseed)
        for fsrc, farr in self._single_seed2relvars['fwd'].items():
            seed_var_map[fsrc][rev_seeds] = \
                self._combine_relevance(self._single_seed2relvars['fwd'], [fsrc],
                                        self._single_seed2relvars['rev'], rev_seeds)
            seed_sys_map[fsrc][rev_seeds] = \
                self._combine_relevance(self._single_seed2relsys['fwd'], [fsrc],
                                        self._single_seed2relsys['rev'], rev_seeds)

        seed_var_map[fwd_seeds] = {}
        seed_sys_map[fwd_seeds] = {}
        for rsrc, rarr in self._single_seed2relvars['rev'].items():
            seed_var_map[fwd_seeds][rsrc] = \
                self._combine_relevance(self._single_seed2relvars['fwd'], fwd_seeds,
                                        self._single_seed2relvars['rev'], [rsrc])
            seed_sys_map[fwd_seeds][rsrc] = \
                self._combine_relevance(self._single_seed2relsys['fwd'], fwd_seeds,
                                        self._single_seed2relsys['rev'], [rsrc])

        # now add 'full' relevance for all seeds
        seed_var_map[fwd_seeds][rev_seeds] = \
            self._combine_relevance(self._single_seed2relvars['fwd'], fwd_seeds,
                                    self._single_seed2relvars['rev'], rev_seeds)
        seed_sys_map[fwd_seeds][rev_seeds] = \
            self._combine_relevance(self._single_seed2relsys['fwd'], fwd_seeds,
                                    self._single_seed2relsys['rev'], rev_seeds)

        self._set_seeds(fwd_seeds, rev_seeds)

        if has_par_derivs:
            self._par_deriv_err_check(group, rev_meta, fwd_meta)

        found = set()
        for fsrc, farr in self._single_seed2relvars['fwd'].items():
            for rsrc, rarr in self._single_seed2relvars['rev'].items():
                if rsrc not in found:
                    if (farr & rarr)[self._var2idx[fsrc]]:
                        found.add(rsrc)

        self._no_dv_responses = \
            [rsrc for rsrc in self._single_seed2relvars['rev'] if rsrc not in found]

    def get_redundant_adjoint_systems(self):
        """
        Find any systems that depend on responses that depend on other responses.

        If any are found, it may be worthwhile to cache RHS arrays and solutions in order to avoid
        some linear solves.

        Returns
        -------
        dict
            Mapping of systems to the set of adjoints that can cause unnecessary linear solves.
        """
        if self._redundant_adjoint_systems is None:
            self._redundant_adjoint_systems = defaultdict(set)
            resp2resp_deps = set()
            for rsrc, arr1 in self._single_seed2relvars['rev'].items():
                for rsrc2 in self._single_seed2relvars['rev']:
                    if rsrc2 != rsrc:
                        if arr1[self._var2idx[rsrc2]]:
                            # add dependent pairs of responses
                            resp2resp_deps.add((rsrc, rsrc2))

            if resp2resp_deps:
                fsystems = self._seed_sys_map[self._all_seed_vars['fwd']]
                for rsrc, rsrc2 in resp2resp_deps:
                    relarr = fsystems[rsrc] & fsystems[rsrc2]  # intersection
                    for relevant_system in self._rel_names_iter(relarr, self._sys2idx):
                        self._redundant_adjoint_systems[relevant_system].update((rsrc, rsrc2))

        return self._redundant_adjoint_systems

    @contextmanager
    def active(self, active):
        """
        Context manager for temporarily deactivating relevance.

        Note that if this relevance object is already inactive, this context manager will have no
        effect, i.e., calling this with active=True will not activate an inactive relevance object,
        but calling it with active=False will deactivate an active relevance object.

        The only way to activate an otherwise inactive relevance object is to use the
        all_seeds_active, seeds_active, or nonlinear_active context managers and this will only
        work if _active is None or True.

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
        names = self._rel_names_iter(self._single_seed2relvars[direction][name], self._var2idx)
        if inputs and outputs:
            return set(names)
        elif inputs:
            return self._apply_node_filter(names, _is_input)
        elif outputs:
            return self._apply_node_filter(names, _is_output)
        else:
            return set()

    @contextmanager
    def all_seeds_active(self):
        """
        Context manager where all seeds are active.

        If _active is False, this will have no effect.

        Yields
        ------
        None
        """
        # if already inactive from higher level, or 'active' parameter is False, don't change it
        if self._active is False:
            yield
        else:
            save = {'fwd': self._seed_vars['fwd'], 'rev': self._seed_vars['rev']}
            save_active = self._active
            self._active = True
            self._set_seeds(self._all_seed_vars['fwd'], self._all_seed_vars['rev'])
            try:
                yield
            finally:
                self._seed_vars = save
                self._active = save_active

    @contextmanager
    def seeds_active(self, fwd_seeds=None, rev_seeds=None):
        """
        Context manager where the specified seeds are active.

        If _active is False, this will have no effect.

        Parameters
        ----------
        fwd_seeds : iter of str or None
            Iterator over forward seed variable names. If None use current active seeds.
        rev_seeds : iter of str or None
            Iterator over reverse seed variable names. If None use current active seeds.

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
            if fwd_seeds is None:
                fwd_seeds = self._seed_vars['fwd']
            if rev_seeds is None:
                rev_seeds = self._seed_vars['rev']
            self._set_seeds(fwd_seeds, rev_seeds)
            try:
                yield
            finally:
                self._seed_vars = save
                self._active = save_active

    @contextmanager
    def nonlinear_active(self, name, active=True):
        """
        Context manager for activating a subset of systems using 'pre', 'post', or 'iter'.

        Parameters
        ----------
        name : str
            Name of the set to activate.
        active : bool
            If False, relevance is temporarily deactivated.

        Yields
        ------
        None
        """
        if not active or self._active is False or name not in self._nonlinear_sets:
            yield
        else:
            save_active = self._active
            save_relarray = self._current_rel_sarray
            self._active = True
            self._current_rel_sarray = self._nonlinear_sets[name]

            try:
                yield
            finally:
                self._active = save_active
                self._current_rel_sarray = save_relarray

    def _set_seeds(self, fwd_seeds, rev_seeds):
        """
        Set the seed(s) to determine relevance for a given variable in a given direction.

        Parameters
        ----------
        fwd_seeds : frozenset
            Set of forward seed variable names.
        rev_seeds : frozenset
            Set of reverse seed variable names.
        """
        fwd_seeds = self._to_seed(fwd_seeds)
        rev_seeds = self._to_seed(rev_seeds)

        self._seed_vars['fwd'] = fwd_seeds
        self._seed_vars['rev'] = rev_seeds

        self._current_rel_varray = self._get_rel_array(self._seed_var_map,
                                                       self._single_seed2relvars,
                                                       fwd_seeds, rev_seeds)
        self._current_rel_sarray = self._get_rel_array(self._seed_sys_map,
                                                       self._single_seed2relsys,
                                                       fwd_seeds, rev_seeds)

    def _get_rel_array(self, seed_map, single_seed2rel, fwd_seeds, rev_seeds):
        """
        Return the combined relevance array for the given seeds.

        If it doesn't exist, create it.

        Parameters
        ----------
        seed_map : dict
            Dict of the form {fwdseed: {revseed: rel_arrays}}.
        single_seed2rel : dict
            Dict of the form {'fwd': {seed: rel_array}, 'rev': ...} where each seed is a key and
            rel_array is the relevance array for the given seed.
        fwd_seeds : str or sorted tuple of str
            Iterator over forward seed variable names.
        rev_seeds : str or sorted tuple of str
            Iterator over reverse seed variable names.

        Returns
        -------
        ndarray
            Array representing the combined relevance arrays for the given seeds.
        """
        try:
            return seed_map[fwd_seeds][rev_seeds]
        except KeyError:
            # don't have a relevance array for this seed combo, so create it
            relarr = self._combine_relevance(single_seed2rel['fwd'], fwd_seeds,
                                             single_seed2rel['rev'], rev_seeds)
            if fwd_seeds not in seed_map:
                seed_map[fwd_seeds] = {}
            seed_map[fwd_seeds][rev_seeds] = relarr

        return relarr

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

        return self._current_rel_varray[self._var2idx[name]]

    def any_relevant(self, names):
        """
        Return True if any of the given variables are relevant.

        Parameters
        ----------
        names : iter of str
            Iterator over variable names.

        Returns
        -------
        bool
            True if any of the given variables are relevant.
        """
        if not self._active:
            return True

        for n in names:
            if self._current_rel_varray[self._var2idx[n]]:
                return True
        return False

    def is_relevant_system(self, name):
        """
        Return True if the given named system is relevant.

        Returns False if system has no subsystems with outputs.

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

        try:
            return self._current_rel_sarray[self._sys2idx[name]]
        except KeyError:
            return False

    def filter(self, systems, relevant=True):
        """
        Filter the given iterator of systems to only include those that are relevant.

        Parameters
        ----------
        systems : iter of Systems
            Iterator over systems.
        relevant : bool
            If True, return only relevant systems.  If False, return only irrelevant systems.

        Yields
        ------
        System
            Relevant system.
        """
        if self._active:
            for system in systems:
                if relevant == self.is_relevant_system(system.pathname):
                    yield system
        elif relevant:
            yield from systems

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
            for rseed in rev_seeds:
                inter = self._get_rel_array(self._seed_var_map, self._single_seed2relvars,
                                            seed, rseed)
                if np.any(inter):
                    inter = self._rel_names_iter(inter, self._var2idx)
                    yield seed, rseed, self._apply_node_filter(inter, filt)

    def _apply_node_filter(self, names, filt):
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

            if direction == 'fwd':
                fnext = self._graph.successors
            elif direction == 'rev':
                fnext = self._graph.predecessors
            else:
                raise ValueError("direction must be 'fwd' or 'rev'")

            stack = [start]
            visited = {start}

            while stack:
                src = stack.pop()
                for tgt in fnext(src):
                    if tgt not in visited:
                        if local:
                            node = self._graph.nodes[tgt]
                            # stop local traversal at the first non-local node
                            if 'local' in node and not node['local']:
                                return visited

                        visited.add(tgt)
                        stack.append(tgt)

            return visited

        return set()

    def _par_deriv_err_check(self, group, responses, desvars):
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

    def _setup_nonlinear_relevance(self, model, designvars, responses):
        """
        Set up the iteration lists containing the pre, iterated, and post subsets of systems.

        This should only be called on the top level Group.

        Parameters
        ----------
        model : <Group>
            The top level group in the system hierarchy.
        designvars : dict
            A dict of all design variables from the model.
        responses : dict
            A dict of all responses from the model.
        """
        # don't redo this if it's already done
        if model._pre_components is not None:
            return

        if not designvars or not responses or not model._problem_meta['group_by_pre_opt_post']:
            return

        model._pre_components = set()
        model._post_components = set()
        model._iterated_components = _contains_all

        # keep track of Groups with nonlinear solvers that use gradients (like Newton) and certain
        # linear solvers like DirectSolver. These groups and all systems they contain must be
        # grouped together into the same iteration list.
        grad_groups = set()
        always_opt = set()
        model._get_relevance_modifiers(grad_groups, always_opt)

        if '' in grad_groups:
            issue_warning("The top level group has a nonlinear solver that computes gradients, so "
                          "the entire model will be included in the optimization iteration.")
            return

        dvs = [meta['source'] for meta in designvars.values()]
        responses = [meta['source'] for meta in responses.values()]
        responses = set(responses)  # get rid of dups due to aliases

        graph = model.compute_sys_graph(comps_only=True, add_edge_info=False)

        auto_dvs = [dv for dv in dvs if dv.startswith('_auto_ivc.')]
        dv0 = auto_dvs[0] if auto_dvs else dvs[0].rpartition('.')[0]

        if auto_dvs:
            rev_conns = get_rev_conns(model._conn_global_abs_in2out)

            # add nodes for any auto_ivc vars that are dvs and connect to downstream component(s)
            for dv in auto_dvs:
                graph.add_node(dv, type_='output')
                inps = rev_conns.get(dv, ())
                for inp in inps:
                    inpcomp = inp.rpartition('.')[0]
                    graph.add_edge(dv, inpcomp)

        # One way to determine the contents of the pre/opt/post sets is to add edges from the
        # response variables to the design variables and vice versa, then find the strongly
        # connected components of the resulting graph.  get_sccs_topo returns the strongly
        # connected components in topological order, so we can use it to give us pre, iterated,
        # and post subsets of the systems.

        # add edges between response comps and design vars/comps to form a strongly
        # connected component for all nodes involved in the optimization iteration.
        for res in responses:
            resnode = res.rpartition('.')[0]
            for dv in dvs:
                dvnode = dv.rpartition('.')[0]
                if dvnode == '_auto_ivc':
                    # var node exists in graph so connect it to resnode
                    dvnode = dv  # use var name not comp name

                graph.add_edge(resnode, dvnode)
                graph.add_edge(dvnode, resnode)

        # loop 'always_opt' components into all responses to force them to be relevant during
        # optimization.
        for opt_sys in always_opt:
            for response in responses:
                rescomp = response.rpartition('.')[0]
                graph.add_edge(opt_sys, rescomp)
                graph.add_edge(rescomp, opt_sys)

        groups_added = set()

        if grad_groups:
            remaining = set(grad_groups)
            for name in sorted(grad_groups, key=lambda x: x.count('.')):
                prefix = name + '.'
                match = {n for n in remaining if n.startswith(prefix)}
                remaining -= match

            gradlist = '\n'.join(sorted(remaining))
            issue_warning("The following groups have a nonlinear solver that computes gradients "
                          f"and will be treated as atomic for the purposes of determining "
                          f"which systems are included in the optimization iteration: "
                          f"\n{gradlist}\n")

            # remaining groups are not contained within a higher level nl solver
            # using gradient group, so make new connections to/from them to
            # all systems that they contain.  This will force them to be
            # treated as 'atomic' within the graph, so that if they contain
            # any dv or response systems, or if their children are connected to
            # both dv *and* response systems, then all systems within them will
            # be included in the 'opt' set.  Note that this step adds some group nodes
            # to the graph where before it only contained component nodes and auto_ivc
            # var nodes.
            edges_to_add = []
            for grp in remaining:
                prefix = grp + '.'
                for node in graph:
                    if node.startswith(prefix):
                        groups_added.add(grp)
                        edges_to_add.append((grp, node))
                        edges_to_add.append((node, grp))

            graph.add_edges_from(edges_to_add)

        # this gives us the strongly connected components in topological order
        sccs = get_sccs_topo(graph)

        pre = addto = set()
        post = set()
        iterated = set()
        for strong_con in sccs:
            # because the sccs are in topological order and all design vars and
            # responses are in the iteration set, we know that until we
            # see a design var or response, we're in the pre-opt set.  Once we
            # see a design var or response, we're in the iterated set.  Once
            # we see an scc without a design var or response, we're in the
            # post-opt set.
            if dv0 in strong_con:
                for s in strong_con:
                    if 'type_' in graph.nodes[s]:
                        s = s.rpartition('.')[0]
                    if s not in iterated:
                        iterated.add(s)
                addto = post
            else:
                for s in strong_con:
                    if 'type_' in graph.nodes[s]:
                        s = s.rpartition('.')[0]
                    if s not in addto:
                        addto.add(s)

        auto_ivc = model._auto_ivc
        auto_dvs = set(auto_dvs)
        rev_conns = get_rev_conns(model._conn_global_abs_in2out)
        if '_auto_ivc' not in pre:
            in_pre = False
            for vname in auto_ivc._var_abs2prom['output']:
                if vname not in auto_dvs:
                    for tgt in rev_conns[vname]:
                        tgtcomp = tgt.rpartition('.')[0]
                        if tgtcomp in pre:
                            in_pre = True
                            break
                    if in_pre:
                        break
            if in_pre:
                pre.add('_auto_ivc')

        # if 'pre' contains nothing but _auto_ivc, then just make it empty
        if len(pre) == 1 and '_auto_ivc' in pre:
            pre.discard('_auto_ivc')

        model._pre_components = pre - groups_added
        model._post_components = post - groups_added
        model._iterated_components = iterated - groups_added

        # it's possible that some components could be in pre on some ranks and post in others
        # if they are not connected in any way to any components in the iterated set, so we
        # need to pick a rank and bcast the final pre and post sets to all ranks to ensure
        # consistency.
        if model.comm.size > 1:
            pre, post = model.comm.bcast((model._pre_components, model._post_components), root=0)
            model._pre_components = pre
            model._post_components = post

    def list_relevance(self, relevant=True, type='system'):
        """
        Return a list of relevant variables and systems for the given seeds.

        Parameters
        ----------
        relevant : bool
            If True, return only relevant variables and systems.  If False, return only irrelevant
            variables and systems.
        type : str
            If 'system', return only system names.  If 'var', return only variable names.

        Returns
        -------
        list of str
            List of (ir)relevant variables or systems.
        """
        if type == 'system':
            it = self._rel_names_iter(self._current_rel_sarray, self._sys2idx, relevant)
        else:
            it = self._rel_names_iter(self._current_rel_varray, self._var2idx, relevant)

        return list(it)


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


def _dump_seed_map(seed_map):
    """
    Print the contents of the given seed_map for debugging.

    Parameters
    ----------
    seed_map : dict
        Dict of the form {fwdseed: {revseed: rel_arrays}}.
    """
    for fseed, relmap in seed_map.items():
        for rseed, relarr in relmap.items():
            print(f'({fseed}, {rseed}) {np.asarray(relarr, dtype=np.uint8)}')
