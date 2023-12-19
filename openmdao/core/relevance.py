

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

    def __init__(self, graph):
        self.graph = graph
        self.irrelevant_sets = {}  # (varname, direction): set of irrelevant vars
        # self._cache = {}  # possibly cache some results for speed???

    def is_relevant_var(self, varname):
        # step 1: if varname irrelevant to active voi, return False
        # step 2: if varname is not irrelevant to any target voi, return True
        # step 3: return False
        pass

    def is_relevant_system(self, system):
        pass
