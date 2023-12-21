
from contextlib import contextmanager
from openmdao.utils.general_utils import _contains_all, all_ancestors


class SetChecker(object):
    """
    Class for checking if a given set of variables is in a relevant set of variables.

    Attributes
    ----------
    _set : set
        Set of variables to check.
    """
    def __init__(self, theset):
        self._set = theset

    def __contains__(self, name):
        return name in self._set


class InverseSetChecker(object):
    """
    Class for checking if a given set of variables is not in an irrelevant set of variables.

    Attributes
    ----------
    _set : set
        Set of variables to check.
    """
    def __init__(self, theset):
        self._set = theset

    def __contains__(self, name):
        return name not in self._set


class Relevance(object):
    """
    Relevance class.

    Attributes
    ----------
    graph : <nx.DirectedGraph>
        Dependency graph.  Hybrid graph containing both variables and systems.
    irrelevant_sets : dict
        Dictionary of irrelevant sets for each (varname, direction) pair.
        Sets will only be stored for variables that are either design variables
        in fwd mode, responses in rev mode, or variables passed directly to
        compute_totals or check_totals.
    """

    def __init__(self, graph):
        self._graph = graph  # relevance graph
        self._all_graph_nodes = None  # set of all nodes in the graph (or None if not initialized)
        self._var_set_checkers = {}  # maps (varname, direction) to variable set checker
        self._relevant_systems = {}  # maps (varname, direction) to relevant system sets
        self._seed_vars = {'fwd': (), 'rev': ()}  # seed vars for the current derivative computation
        self._target_vars = {'fwd': (), 'rev': ()}  # target vars for the current derivative
                                                    # computation
        self._active = True

    @contextmanager
    def inactive_context(self):
        """
        Context manager for deactivating relevance.
        """
        save = self._active
        self._active = False
        try:
            yield
        finally:
            self._active = save

    def set_targets(self, target_vars, direction):
        """
        Set the targets to be used to determine relevance for a given seed.

        Parameters
        ----------
        target_vars : iter of str
            Iterator over target variable names.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.

        Returns
        -------
        tuple
            Old tuple of target variables.
        """
        # print("Setting relevance targets to", target_vars)
        if isinstance(target_vars, str):
            target_vars = [target_vars]

        old_target_vars = self._target_vars[direction]

        self._target_vars[direction] = tuple(sorted(target_vars))

        opposite = 'rev' if direction == 'fwd' else 'rev'

        for t in self._target_vars[opposite]:
            self._get_relevance_set(t, opposite)

        return old_target_vars

    def set_seeds(self, seed_vars, direction):
        """
        Set the seed(s) to be used to determine relevance for a given variable.

        Parameters
        ----------
        seed_vars : str or iter of str
            Iterator over seed variable names.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.

        Returns
        -------
        tuple
            Old tuple of seed variables.
        """
        # print(direction, id(self), "Setting relevance seeds to", seed_vars)
        if isinstance(seed_vars, str):
            seed_vars = [seed_vars]

        old_seed_vars = self._seed_vars[direction]

        self._seed_vars[direction] = tuple(sorted(seed_vars))

        for s in self._seed_vars[direction]:
            self._get_relevance_set(s, direction)

        return old_seed_vars

    def is_relevant(self, name, direction):
        if not self._active:
            return True

        assert self._seed_vars[direction] and self._target_vars[direction], \
            "must call set_seeds and set_targets first"
        for seed in self._seed_vars[direction]:
            if name in self._get_relevance_set(seed, direction):
                for tgt in self._target_vars[direction]:
                    if name in self._get_relevance_set(tgt, direction):
                        return True
        return False

    def is_relevant_system(self, name, direction):
        if not self._active:
            return True

        assert direction in ('fwd', 'rev')
        if len(self._seed_vars[direction]) == 0 and len(self._target_vars[direction]) == 0:
            return True  # no relevance is set up, so assume everything is relevant

        opposite = 'rev' if direction == 'fwd' else 'fwd'

        # print(id(self), "is_relevant_system", name, direction)
        for seed in self._seed_vars[direction]:
            if name in self._relevant_systems[seed, direction]:
                # resolve target dependencies in opposite direction
                for tgt in self._target_vars[direction]:
                    self._get_relevance_set(tgt, opposite)
                    if name in self._relevant_systems[tgt, opposite]:
                        return True
        return False

    def system_filter(self, systems, direction, relevant=True):
        """
        Filter the given iterator of systems to only include those that are relevant.

        Parameters
        ----------
        systems : iter of Systems
            Iterator over systems.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
        relevant : bool
            If True, return only relevant systems.  If False, return only irrelevant systems.

        Yields
        ------
        System
            Relevant system.
        """
        if self._active:
            for system in systems:
                if relevant == self.is_relevant_system(system.pathname, direction):
                    yield system
        elif relevant:
            yield from systems

    def _get_relevance_set(self, varname, direction):
        """
        Return a SetChecker for variables and components for the given variable.

        The irrelevant set is determined lazily and cached for future use.

        Parameters
        ----------
        varname : str
            Name of the variable.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
            'fwd' will find downstream nodes, 'rev' will find upstream nodes.

        Returnss
        -------
        SetChecker, InverseSetChecker, or _contains_all
            Set checker for testing if any variable is relevant to the given variable.
        """
        try:
            return self._var_set_checkers[varname, direction]
        except KeyError:
            assert direction in ('fwd', 'rev'), "direction must be 'fwd' or 'rev'"
            # first time we've seen this varname/direction pair, so we need to
            # compute the set of relevant variables and the set of relevant systems
            # and store them for future use.
            key = (varname, direction)
            depnodes = self._dependent_nodes(varname, direction)

            if len(depnodes) < len(self._graph):
                # only create the full node set if we need it
                # this set contains all variables and some or all components
                # in the graph.  Components are included if all of their outputs
                # depend on all of their inputs.
                if self._all_graph_nodes is None:
                    self._all_graph_nodes = set(self._graph.nodes())

                rel_systems = set(all_ancestors(varname.rpartition('.')[0]))
                rel_systems.add('')  # root group is always relevant
                for name in depnodes:
                    sysname = name.rpartition('.')[0]
                    if sysname not in rel_systems:
                        rel_systems.update(all_ancestors(sysname))
                self._relevant_systems[key] = rel_systems

                irrelevant = self._all_graph_nodes - depnodes
                # store whichever type of checker will use the least memory
                if len(irrelevant) < len(depnodes):
                    self._var_set_checkers[key] = InverseSetChecker(irrelevant)
                else:
                    # remove systems from final var set
                    self._var_set_checkers[key] = SetChecker(depnodes - rel_systems)

            else:
                self._var_set_checkers[key] = _contains_all
                self._relevant_systems[key] = _contains_all

            return self._var_set_checkers[key]

    def _dependent_nodes(self, start, direction):
        """
        Return set of all connected nodes in the given direction starting at the given node.

        Parameters
        ----------
        start : hashable object
            Identifier of the starting node.
        direction : str
            If 'fwd', traverse downstream.  If 'rev', traverse upstream.

        Returns
        -------
        set
            Set of all dependent nodes.
        """
        if start in self._graph:
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
                        visited.add(tgt)
                        stack.append(tgt)

            return visited

        return set()
