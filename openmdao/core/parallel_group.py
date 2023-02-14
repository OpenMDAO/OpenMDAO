"""Define the ParallelGroup class."""

import networkx as nx

from openmdao.core.group import Group


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

    def _get_sys_tree(self, tree):
        tree = super()._get_sys_tree(tree)

        if self.comm.size > 1:
            prefix = self.pathname + '.' if self.pathname else ''
            subtree = nx.subgraph(tree, [n for n in tree if n.startswith(prefix)])
            edges = tree.edges()
            for sub in self.comm.allgather(subtree):  # TODO: make this more efficient
                for n, data in sub.nodes(data=True):
                    if n not in tree:
                        tree.add_node(n, **data)
                for u, v, data in sub.edges(data=True):
                    if (u, v) not in edges:
                        tree.add_edge(u, v, **data)

        return tree
