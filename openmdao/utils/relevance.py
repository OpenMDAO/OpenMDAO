"""
Class definitions for Relevance and related classes.
"""

import sys
from pprint import pprint
from contextlib import contextmanager
from openmdao.utils.general_utils import all_ancestors, meta2src_iter
from openmdao.utils.om_warnings import issue_warning, DerivativesWarning


class SetChecker(object):
    """
    Class for checking if a given set of variables is in a relevant set of variables.

    Parameters
    ----------
    the_set : set
        Set of variables to check against.
    full_set : set or None
        Set of all variables.  Not used if _invert is False.
    invert : bool
        If True, the set is inverted.

    Attributes
    ----------
    _set : set
        Set of variables to check.
    _full_set : set or None
        Set of all variables.  None if _invert is False.
    _invert : bool
        If True, the set is inverted.
    """

    def __init__(self, the_set, full_set=None, invert=False):
        """
        Initialize all attributes.
        """
        assert not invert or full_set is not None, \
            "full_set must be provided if invert is True"
        self._set = the_set
        self._full_set = full_set
        self._invert = invert

    def __contains__(self, name):
        """
        Return True if the given name is in the set.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        bool
            True if the given name is in the set.
        """
        if self._invert:
            return name not in self._set
        return name in self._set

    def __iter__(self):
        """
        Return an iterator over the set.

        Returns
        -------
        iter
            Iterator over the set.
        """
        if self._invert:
            for name in self._full_set:
                if name not in self._set:
                    yield name
        else:
            yield from self._set

    def __repr__(self):
        """
        Return a string representation of the SetChecker.

        Returns
        -------
        str
            String representation of the SetChecker.
        """
        return f"SetChecker({sorted(self._set)}, invert={self._invert}"

    def __len__(self):
        """
        Return the number of elements in the set.

        Returns
        -------
        int
            Number of elements in the set.
        """
        if self._invert:
            return len(self._full_set) - len(self._set)
        return len(self._set)

    def to_set(self):
        """
        Return a set of names of relevant variables.

        Returns
        -------
        set
            Set of our entries.
        """
        if self._invert:
            if self._set:  # check to avoid a set copy
                return self._full_set - self._set
            return self._full_set
        return self._set

    def intersection(self, other_set):
        """
        Return a new set with elements common to the set and all others.

        Parameters
        ----------
        other_set : set
            Other set to check against.

        Returns
        -------
        set
            Set of common elements.
        """
        if self._invert:
            if self._set:
                return other_set - self._set
            return other_set

        return self._set.intersection(other_set)


_opposite = {'fwd': 'rev', 'rev': 'fwd'}


class Relevance(object):
    """
    Class that computes relevance based on a data flow graph.

    Parameters
    ----------
    group : <System>
        The top level group in the system hierarchy.
    desvars : dict
        Dictionary of design variables.  Keys don't matter.
    responses : dict
        Dictionary of response variables.  Keys don't matter.

    Attributes
    ----------
    _graph : <nx.DirectedGraph>
        Dependency graph.  Hybrid graph containing both variables and systems.
    _all_vars : set or None
        Set of all variables in the graph.  None if not initialized.
    _relevant_vars : dict
        Maps (varname, direction) to variable set checker.
    _relevant_systems : dict
        Maps (varname, direction) to relevant system sets.
    _seed_vars : dict
        Maps direction to currently active seed variable names.
    _all_seed_vars : dict
        Maps direction to all seed variable names.
    _local_seeds : set
        Set of seed vars restricted to local dependencies.
    _active : bool or None
        If True, relevance is active.  If False, relevance is inactive.  If None, relevance is
        uninitialized.
    """

    def __init__(self, group, desvars, responses):
        """
        Initialize all attributes.
        """
        self._all_vars = None  # set of all nodes in the graph (or None if not initialized)
        self._relevant_vars = {}  # maps (varname, direction) to variable set checker
        self._relevant_systems = {}  # maps (varname, direction) to relevant system sets
        # seed var(s) for the current derivative operation
        self._seed_vars = {'fwd': (), 'rev': ()}
        # all seed vars for the entire derivative computation
        self._all_seed_vars = {'fwd': (), 'rev': ()}
        self._local_seeds = set()  # set of seed vars restricted to local dependencies
        self._graph = self.get_relevance_graph(group, desvars, responses)
        self._active = None  # not initialized

        # for any parallel deriv colored dv/responses, update the relevant sets to include vars with
        # local only dependencies
        for meta in desvars.values():
            if meta['parallel_deriv_color'] is not None:
                self.set_seeds([meta['source']], 'fwd', local=True)
        for meta in responses.values():
            if meta['parallel_deriv_color'] is not None:
                self.set_seeds([meta['source']], 'rev', local=True)

        if desvars and responses:
            self.set_all_seeds([m['source'] for m in desvars.values()],
                               set(m['source'] for m in responses.values()))  # set removes dups
        else:
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

    def get_relevance_graph(self, group, desvars, responses):
        """
        Return a graph of the relevance between desvars and responses.

        This graph is the full hybrid graph after removal of components that don't have full
        ('*', '*') partial derivatives declared.  When such a component is removed, its inputs and
        outputs are connected to each other whenever there is a partial derivative declared between
        them.

        Parameters
        ----------
        group : <Group>
            The top level group in the system hierarchy.
        desvars : dict
            Dictionary of design variable metadata.
        responses : dict
            Dictionary of response variable metadata.

        Returns
        -------
        DiGraph
            Graph of the relevance between desvars and responses.
        """
        graph = group._get_hybrid_graph()

        # if doing top level FD/CS, don't update relevance graph based
        # on missing partials because FD/CS doesn't require that partials
        # are declared to compute derivatives
        if group._owns_approx_jac:
            return graph

        resps = set(meta2src_iter(responses.values()))

        # figure out if we can remove any edges based on zero partials we find
        # in components.  By default all component connected outputs
        # are also connected to all connected inputs from the same component.
        missing_partials = {}
        group._get_missing_partials(missing_partials)
        missing_responses = set()
        for pathname, missing in missing_partials.items():
            inputs = [n for n, _ in graph.in_edges(pathname)]
            outputs = [n for _, n in graph.out_edges(pathname)]

            graph.remove_node(pathname)

            for output in outputs:
                found = False
                for inp in inputs:
                    if (output, inp) not in missing:
                        graph.add_edge(inp, output)
                        found = True

                if not found and output in resps:
                    missing_responses.add(output)

        if missing_responses:
            msg = (f"Constraints or objectives [{', '.join(sorted(missing_responses))}] cannot"
                   " be impacted by the design variables of the problem because no partials "
                   "were defined for them in their parent component(s).")
            if group._problem_meta['singular_jac_behavior'] == 'error':
                raise RuntimeError(msg)
            else:
                issue_warning(msg, category=DerivativesWarning)

        return graph

    def relevant_vars(self, name, direction, inputs=True, outputs=True):
        """
        Return a set of variables relevant to the given variable in the given direction.

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
        self._init_relevance_set(name, direction)
        if inputs and outputs:
            return self._relevant_vars[name, direction].to_set()
        elif inputs:
            return self._apply_filter(self._relevant_vars[name, direction], _is_input)
        elif outputs:
            return self._apply_filter(self._relevant_vars[name, direction], _is_output)
        else:
            return set()

    def set_all_seeds(self, fwd_seeds, rev_seeds):
        """
        Set the full list of seeds to be used to determine relevance.

        Parameters
        ----------
        fwd_seeds : iter of str
            Iterator over forward seed variable names.
        rev_seeds : iter of str
            Iterator over reverse seed variable names.
        """
        self._all_seed_vars['fwd'] = self._seed_vars['fwd'] = tuple(sorted(fwd_seeds))
        self._all_seed_vars['rev'] = self._seed_vars['rev'] = tuple(sorted(rev_seeds))

        for s in fwd_seeds:
            self._init_relevance_set(s, 'fwd')
        for s in rev_seeds:
            self._init_relevance_set(s, 'rev')

        if self._active is None:
            self._active = True

    def set_seeds(self, seed_vars, direction, local=False):
        """
        Set the seed(s) to determine relevance for a given variable in a given direction.

        Parameters
        ----------
        seed_vars : str or iter of str
            Iterator over seed variable names.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
        local : bool
            If True, update relevance set if necessary to include only local variables.
        """
        if self._active is False:
            return  # don't set seeds if we're inactive

        if isinstance(seed_vars, str):
            seed_vars = [seed_vars]

        seed_vars = tuple(sorted(seed_vars))

        self._seed_vars[direction] = seed_vars
        self._seed_vars[_opposite[direction]] = self._all_seed_vars[_opposite[direction]]

        for s in seed_vars:
            self._init_relevance_set(s, direction, local=local)

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

        assert self._seed_vars['fwd'] and self._seed_vars['rev'], \
            "must call set_all_seeds and set_all_targets first"

        # return name in self._intersection_cache[self._seed_vars['fwd'], self._seed_vars['rev']]

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

    def system_filter(self, systems, relevant=True):
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
        if key not in self._relevant_vars or (local and key not in self._local_seeds):
            assert direction in ('fwd', 'rev'), "direction must be 'fwd' or 'rev'"

            # first time we've seen this varname/direction pair, so we need to
            # compute the set of relevant variables and the set of relevant systems
            # and store them for future use.
            depnodes = self._dependent_nodes(varname, direction, local=local)

            rel_systems = _vars2systems(depnodes)

            # this set contains all variables and some or all components
            # in the graph.  Components are included if all of their outputs
            # depend on all of their inputs.
            if self._all_vars is None:
                self._all_systems = allsystems = _vars2systems(self._graph.nodes())
                self._all_vars = {n for n in self._graph.nodes() if n not in allsystems}

            rel_vars = depnodes - self._all_systems

            if local:
                self._local_seeds.add(key)

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


def _is_discrete(node):
    return node['discrete']
