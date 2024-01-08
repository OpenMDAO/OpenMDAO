"""
Class definitions for Relevance and related classes.
"""

import sys
from pprint import pprint
from contextlib import contextmanager
from openmdao.utils.general_utils import _contains_all, all_ancestors, dprint


class SetChecker(object):
    """
    Class for checking if a given set of variables is in a relevant set of variables.

    Parameters
    ----------
    the_set : set
        Set of variables to check against.

    Attributes
    ----------
    _set : set
        Set of variables to check.
    """

    def __init__(self, the_set):
        """
        Initialize all attributes.
        """
        self._set = the_set

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
        return name in self._set

    def __repr__(self):
        """
        Return a string representation of the SetChecker.

        Returns
        -------
        str
            String representation of the SetChecker.
        """
        return f"SetChecker({sorted(self._set)})"


class InverseSetChecker(object):
    """
    Class for checking if a given set of variables is not in an irrelevant set of variables.

    Parameters
    ----------
    the_set : set
        Set of variables to check against.

    Attributes
    ----------
    _set : set
        Set of variables to check.
    """

    def __init__(self, the_set):
        """
        Initialize all attributes.
        """
        self._set = the_set

    def __contains__(self, name):
        """
        Return True if the given name is not in the set.

        Parameters
        ----------
        name : str
            Name of the variable.

        Returns
        -------
        bool
            True if the given name is not in the set.
        """
        return name not in self._set

    def __repr__(self):
        """
        Return a string representation of the InverseSetChecker.

        Returns
        -------
        str
            String representation of the InverseSetChecker.
        """
        return f"InverseSetChecker({sorted(self._set)})"


_opposite = {'fwd': 'rev', 'rev': 'fwd'}


class Relevance(object):
    """
    Relevance class.

    Parameters
    ----------
    graph : <nx.DirectedGraph>
        Dependency graph.  Hybrid graph containing both variables and systems.

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
    _active : bool or None
        If True, relevance is active.  If False, relevance is inactive.  If None, relevance is
        uninitialized.
    _force_total : bool
        If True, force use of total relevance (object is relevant if it is relevant for any
        seed/target combination).
    """

    def __init__(self, graph):
        """
        Initialize all attributes.
        """
        self._graph = graph
        self._all_vars = None  # set of all nodes in the graph (or None if not initialized)
        self._relevant_vars = {}  # maps (varname, direction) to variable set checker
        self._relevant_systems = {}  # maps (varname, direction) to relevant system sets
        # seed var(s) for the current derivative operation
        self._seed_vars = {'fwd': (), 'rev': (), None: ()}
        # all seed vars for the entire derivative computation
        self._all_seed_vars = {'fwd': (), 'rev': (), None: ()}
        self._active = None  # not initialized
        self._force_total = False

    @contextmanager
    def activity_context(self, active):
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
        self._check_active()
        if not self._active:  # if already inactive from higher level, don't change it
            yield
        else:
            save = self._active
            self._active = active
            try:
                yield
            finally:
                self._active = save

    @contextmanager
    def total_relevance_context(self):
        """
        Context manager for activating/deactivating forced total relevance.

        Yields
        ------
        None
        """
        self._check_active()
        if not self._active:  # if already inactive from higher level, don't change anything
            yield
        else:
            save = self._force_total
            self._force_total = True
            try:
                yield
            finally:
                self._force_total = save

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

        dprint("set all seeds to:", tuple(sorted(self._all_seed_vars['fwd'])), "for fwd")
        dprint("set all seeds to:", tuple(sorted(self._all_seed_vars['rev'])), "for rev")

        for s in fwd_seeds:
            self._init_relevance_set(s, 'fwd')
        for s in rev_seeds:
            self._init_relevance_set(s, 'rev')

        if self._active is None:
            self._active = True

    def reset_to_all_seeds(self):
        """
        Reset the seed vars to the full list of seeds.
        """
        dprint("reset all seeds to:", tuple(sorted(self._all_seed_vars['fwd'])), "for fwd")
        dprint("reset all seeds to:", tuple(sorted(self._all_seed_vars['rev'])), "for rev")
        self._seed_vars['fwd'] = self._all_seed_vars['fwd']
        self._seed_vars['rev'] = self._all_seed_vars['rev']

    def set_seeds(self, seed_vars, direction):
        """
        Set the seed(s) to be used to determine relevance for a given variable.

        Parameters
        ----------
        seed_vars : str or iter of str
            Iterator over seed variable names.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.
        """
        if self._active is False:
            return  # don't set seeds if we're inactive

        if isinstance(seed_vars, str):
            seed_vars = [seed_vars]

        dprint("set seeds to:", tuple(sorted(seed_vars)), "for", direction)
        self._seed_vars[direction] = tuple(sorted(seed_vars))
        self._seed_vars[_opposite[direction]] = self._all_seed_vars[_opposite[direction]]

        for s in self._seed_vars[direction]:
            self._init_relevance_set(s, direction)

    def _check_active(self):
        """
        Activate if all_seed_vars and all_target_vars are set and active is None.
        """
        if self._active is None and self._all_seed_vars['fwd'] and self._all_seed_vars['rev']:
            self._active = True

    def is_relevant(self, name, direction):
        """
        Return True if the given variable is relevant.

        Parameters
        ----------
        name : str
            Name of the variable.
        direction : str
            Direction of the search for relevant variables.  'fwd' or 'rev'.

        Returns
        -------
        bool
            True if the given variable is relevant.
        """
        if not self._active:
            return True

        assert direction in ('fwd', 'rev')

        assert self._seed_vars[direction] and self._seed_vars[_opposite[direction]], \
            "must call set_all_seeds and set_all_targets first"

        for seed in self._seed_vars[direction]:
            if name in self._relevant_vars[seed, direction]:
                opp = _opposite[direction]
                for tgt in self._seed_vars[opp]:
                    if name in self._relevant_vars[tgt, opp]:
                        return True

        return False

    def is_relevant_system(self, name, direction):
        """
        Return True if the given named system is relevant.

        Parameters
        ----------
        name : str
            Name of the System.
        direction : str
            Direction of the search for relevant systems.  'fwd' or 'rev'.

        Returns
        -------
        bool
            True if the given system is relevant.
        """
        if not self._active:
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
        """
        Return True if the given named system is relevant.

        Relevance in this case pertains to all seed/target combinations.

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

        for direction, seeds in self._all_seed_vars.items():
            for seed in seeds:
                if name in self._relevant_systems[seed, direction]:
                    # resolve target dependencies in opposite direction
                    opp = _opposite[direction]
                    for tgt in self._all_seed_vars[opp]:
                        if name in self._relevant_systems[tgt, opp]:
                            return True
        return False

    def system_filter(self, systems, direction=None, relevant=True):
        """
        Filter the given iterator of systems to only include those that are relevant.

        Parameters
        ----------
        systems : iter of Systems
            Iterator over systems.
        direction : str or None
            Direction of the search for relevant variables.  'fwd', 'rev', or None. None is
            only valid if relevance is not active or if doing 'total' relevance, where
            relevance is True if a system is relevant to any pair of of/wrt variables.
        relevant : bool
            If True, return only relevant systems.  If False, return only irrelevant systems.

        Yields
        ------
        System
            Relevant system.
        """
        if self._active:
            if self._force_total:
                relcheck = self.is_total_relevant_system
                for system in systems:
                    if relevant == relcheck(system.pathname):
                        yield system
            else:
                if direction is None:
                    raise RuntimeError("direction must be 'fwd' or 'rev' if relevance is active.")
                relcheck = self.is_relevant_system
                for system in systems:
                    if relevant == relcheck(system.pathname, direction):
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
        relevant : bool
            If True, return only relevant systems.  If False, return only irrelevant systems.

        Yields
        ------
        System
            Relevant system.
        """
        if self._active:
            systems = list(systems)
            #dprint("total all systems:", [s.pathname for s in systems])
            #relsys =  [s.pathname for s in systems if self.is_total_relevant_system(s.pathname)]
            #dprint("total relevant systems:", relsys)
            for system in systems:
                if relevant == self.is_total_relevant_system(system.pathname):
                    yield system
                else:
                    if relevant:
                        dprint("(total)", relevant, "skipping", system.pathname)
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
            self._relevant_vars[key] = _get_set_checker(depnodes - self._all_systems,
                                                        self._all_vars)

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

    def _dump_old(self, out_stream=sys.stdout):
        import pprint
        pprint.pprint(self.old, stream=out_stream)

    def _show_old_relevant_sys(self, relev):
        for dv, dct in relev.items():
            for resp, tup in dct.items():
                vdct = tup[0]
                systems = tup[1]
                print(f"({dv}, {resp}) systems: {sorted(systems)}")
                print(f"({dv}, {resp}) vars: {sorted(vdct['input'].union(vdct['output']))}")


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
