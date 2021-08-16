"""Define the ParallelGroup class."""

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
