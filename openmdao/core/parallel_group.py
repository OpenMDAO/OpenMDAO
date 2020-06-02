"""Define the ParallelGroup class."""

from openmdao.core.group import Group


class ParallelGroup(Group):
    """
    Class used to group systems together to be executed in parallel.
    """

    def __init__(self, **kwargs):
        """
        Set the mpi_proc_allocator option to 'parallel'.

        Parameters
        ----------
        **kwargs : dict
            dict of arguments available here and in all descendants of this
            Group.
        """
        super(ParallelGroup, self).__init__(**kwargs)
        self._mpi_proc_allocator.parallel = True
