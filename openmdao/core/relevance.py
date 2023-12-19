

class Relevance(object):
    # State is:
    #   - direction  (fwd, rev)
    #   - active voi (dv in fwd, resp in rev)
    #   - target vois (resp in fwd, dv in rev)

    # graph doesn't change

    # maybe ask: return the relevance for a given var/system for a desired
    # set of outputs (fwd) or inputs (rev) and given a specific input variable
    # input in fwd or output in rev

    # storing irrelevant set for each var would likely take up less space,
    # and for a given pair of vars, you can get the intersection of their
    # irrelevant sets to see if they are relevant to each other.  This allows
    # us to compute relevance on the fly for pairs of vars that are not 'built-in'
    # design vars or responses.

    # need special handling for groups because they are relevant if any of their
    # descendants are relevant.
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

    def __init__(self, graph, prob_meta):
        self._graph = graph
        self._all_graph_nodes = set(graph.nodes())
        self._prob_meta = prob_meta
        self._irrelevant_sets = {}  # (varname, direction): set of irrelevant vars
        self.seed_vars = set()  # set of seed vars for the current derivative computation

    def is_relevant(self, name, direction):
        # step 1: if varname irrelevant to seed var(s), return False
        # step 2: if varname is not irrelevant to any target voi, return True
        # step 3: return False
        relseeds = [s for s in self._prob_meta['seed_vars'] 
                    if name not in self._get_irrelevant_nodes(s, direction)]
        if not relseeds:
            return False
        
        for seed in relseeds:
            if name not in self._get_irrelevant_nodes(seed, direction):
                return True
        return False
    
    def _get_irrelevant_nodes(self, varname, direction):
        """
        Return the set of irrelevant variables and components for the given 'wrt' or 'of' variable.

        The irrelevant set is determined lazily and cached for future use.

        Parameters
        ----------
        varname : str
            Name of the variable.  Must be a 'wrt' variable in fwd mode or a 'of' variable
            in rev mode.
        direction : str
            Direction of the derivative.  'fwd' or 'rev'.

        Returns
        -------
        set
            Set of irrelevant variables.
        """
        try:
            return self._irrelevant_sets[(varname, direction)]
        except KeyError:
            key = (varname, direction)
            depnodes = self._dependent_nodes(varname, direction)
            self._irrelevant_sets[key] = self._all_graph_nodes - depnodes
            return self._irrelevant_sets[key]

    def _dependent_nodes(self, start, direction):
        """
        Return set of all downstream nodes starting at the given node.

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
        visited = set()

        stack = [start]
        visited.add(start)

        fnext = self._graph.successors if direction == 'fwd' else self._graph.predecessors

        while stack:
            src = stack.pop()
            for tgt in fnext(src):
                if tgt not in visited:
                    visited.add(tgt)
                    stack.append(tgt)

        return visited
