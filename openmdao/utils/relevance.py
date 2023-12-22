
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


_opposite = {'fwd': 'rev', 'rev': 'fwd'}


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
        self._all_vars = None  # set of all nodes in the graph (or None if not initialized)
        self._relevant_vars = {}  # maps (varname, direction) to variable set checker
        self._relevant_systems = {}  # maps (varname, direction) to relevant system sets
        self._seed_vars = {'fwd': (), 'rev': (), None: ()}  # seed var(s) for the current derivative
                                                            # operation
        self._all_seed_vars = {'fwd': (), 'rev': (), None: ()}  # all seed vars for the entire
                                                                # derivative computation
        self._active = None  # not initialized

    @contextmanager
    def activity_context(self, active):
        """
        Context manager for activating/deactivating relevance.
        """
        self._check_active()
        if self._active is None:
            yield
        else:
            save = self._active
            self._active = active
            try:
                yield
            finally:
                self._active = save

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
        assert not isinstance(fwd_seeds, str), "fwd_seeds must be an iterator of strings"
        assert not isinstance(rev_seeds, str), "rev_seeds must be an iterator of strings"

        self._all_seed_vars['fwd'] = self._seed_vars['fwd'] = tuple(sorted(fwd_seeds))
        self._all_seed_vars['rev'] = self._seed_vars['rev'] = tuple(sorted(rev_seeds))

        for s in fwd_seeds:
            self._init_relevance_set(s, 'fwd')
        for s in rev_seeds:
            self._init_relevance_set(s, 'rev')

        if self._active is None:
            self._active = True

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
        if self._active is False:
            return self._seed_vars[direction]

        if isinstance(seed_vars, str):
            seed_vars = [seed_vars]

        old_seed_vars = self._seed_vars[direction]

        self._seed_vars[direction] = tuple(sorted(seed_vars))

        for s in self._seed_vars[direction]:
            self._init_relevance_set(s, direction)

        return old_seed_vars

    def _check_active(self):
        """
        Activate if all_seed_vars and all_target_vars are set and active is None.
        """
        if self._active is None and self._all_seed_vars['fwd'] and self._all_seed_vars['rev']:
            self._active = True
            return

    def is_relevant(self, name, direction):
        if not self._active:
            return True

        assert direction in ('fwd', 'rev')

        assert self._seed_vars[direction] and self._target_vars[_opposite[direction]], \
            "must call set_all_seeds and set_all_targets first"

        for seed in self._seed_vars[direction]:
            if name in self._relevant_vars[seed, direction]:
                opp = _opposite[direction]
                for tgt in self._seed_vars[opp]:
                    if name in self._relevant_vars[tgt, opp]:
                        return True
        return False

    def is_relevant_system(self, name, direction):
        if not self._active:  # False or None
            return True

        assert direction in ('fwd', 'rev')

        for seed in self._seed_vars[direction]:
            if name in self._relevant_systems[seed, direction]:
                # resolve target dependencies in opposite direction
                opp = _opposite[direction]
                for tgt in self._seed_vars[opp]:
                    if name in self._relevant_systems[tgt, opp]:
                        return True
        return False

    def is_total_relevant_system(self, name):
        if not self._active:  # False or None
            return True

        for direction, seeds in self._all_seed_vars.items():
            for seed in seeds:
                if name in self._relevant_systems[seed, direction]:
                    # resolve target dependencies in opposite direction
                    opp = _opposite[direction]
                    for tgt in self._all_seed_vars[opp]:
                        if name in self._relevant_systems[tgt, opp]:
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

    def total_system_filter(self, systems, relevant=True):
        """
        Filter the given iterator of systems to only include those that are relevant.

        Parameters
        ----------
        systems : iter of Systems
            Iterator over systems.
        direction : str or None
            Direction of the search for relevant variables.  'fwd' or 'rev'.  None means
            search in both directions.
        relevant : bool
            If True, return only relevant systems.  If False, return only irrelevant systems.

        Yields
        ------
        System
            Relevant system.
        """
        if self._active:
            for system in systems:
                if relevant == self.is_total_relevant_system(system.pathname):
                    yield system
        elif relevant:
            yield from systems

    def _init_relevance_set(self, varname, direction):
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
        """
        key = (varname, direction)
        if key not in self._relevant_vars:
            assert direction in ('fwd', 'rev'), "direction must be 'fwd' or 'rev'"

            # first time we've seen this varname/direction pair, so we need to
            # compute the set of relevant variables and the set of relevant systems
            # and store them for future use.
            depnodes = self._dependent_nodes(varname, direction)

            # this set contains all variables and some or all components
            # in the graph.  Components are included if all of their outputs
            # depend on all of their inputs.
            if self._all_vars is None:
                self._all_systems = _vars2systems(self._graph.nodes())
                self._all_vars = set(self._graph.nodes()) - self._all_systems

            rel_systems = _vars2systems(depnodes)
            self._relevant_systems[key] = _get_set_checker(rel_systems, self._all_systems)
            self._relevant_vars[key] = _get_set_checker(depnodes, self._all_vars)

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


def _vars2systems(nameiter):
    """
    Return a set of all systems containing the given variables or components.

    This includes all ancestors of each system.

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
    Return a SetChecker, InverseSetChecker, or _contains_all for the given sets.

    Parameters
    ----------
    relset : set
        Set of relevant items.
    allset : set
        Set of all items.

    Returns
    -------
    SetChecker, InverseSetChecker, or _contiains_all
        Set checker for the given sets.
    """
    if len(allset) == len(relset):
        return _contains_all

    inverse = allset - relset
    # store whichever type of checker will use the least memory
    if len(inverse) < len(relset):
        return InverseSetChecker(inverse)
    else:
        return SetChecker(relset)
