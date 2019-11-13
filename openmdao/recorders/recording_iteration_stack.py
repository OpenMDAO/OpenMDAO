"""Management of iteration stack for recording."""
import weakref

from openmdao.utils.mpi import MPI


class _RecIteration(object):
    """
    A class that encapsulates the iteration stack.

    Some tests needed to reset the stack and this avoids issues
    with data left over from other tests.

    Attributes
    ----------
    stack : list
        A list that holds the stack of iteration coordinates.
    prefix : str or None
        Prefix to prepend to iteration coordinates.
    """

    def __init__(self):
        """
        Initialize.
        """
        self.stack = []
        self.prefix = None

    def print_recording_iteration_stack(self):
        """
        Print the record iteration stack.

        Used for debugging.
        """
        print()
        for name, iter_count in reversed(self.stack):
            print('^^^', name, iter_count)
        print(60 * '^')

    def get_formatted_iteration_coordinate(self):
        """
        Format the iteration coordinate into human-readable form.

        'rank0:pyoptsparsedriver|6|root._solve_nonlinear|6|mda._solve_nonlinear|6|mda.d1._solve_nonlinear|45'

        Returns
        -------
        str :
            the iteration coordinate formatted in our proprietary way.
        """
        separator = '|'

        # prefix
        if self.prefix:
            prefix = '%s_' % self.prefix
        else:
            prefix = ''

        if MPI:
            prefix += 'rank%d:' % MPI.COMM_WORLD.rank
        else:
            prefix += 'rank0:'

        # iteration hierarchy
        coord_list = []
        for name, iter_count in self.stack:
            coord_list.append('{}{}{}'.format(name, separator, iter_count))

        return prefix + separator.join(coord_list)


class Recording(object):
    """
    A class that acts as a context manager.

    But with properly-timed values for abs and rel,
    where solvers are concerned.

    Attributes
    ----------
    name : str
        Name of object getting recorded.
    iter_count : int
        Current counter of iterations completed.
    recording_requester : weakref to object
        The object that wants to be recorded.
    stack : list
        Stack containing names and iteration counts.
    abs : float
        Absolute error.
    rel : float
        Relative error.
    _is_solver : bool
        True if recording_requester is a Solver.
    """

    def __init__(self, name, iter_count, recording_requester):
        """
        Initialize Recording.

        Parameters
        ----------
        name : str
            Name of object getting recorded.
        iter_count : int
            Current counter of iterations completed.
        recording_requester : object
            The object that wants to be recorded.
        """
        self.name = name
        self.iter_count = iter_count
        self.recording_requester = weakref.ref(recording_requester)
        self.abs = 0
        self.rel = 0

        from openmdao.solvers.solver import Solver
        self._is_solver = isinstance(recording_requester, Solver)

    def __enter__(self):
        """
        Do things before the code inside the 'with Recording' block.

        Returns
        -------
        self : object
            self
        """
        self.recording_requester()._recording_iter.stack.append((self.name, self.iter_count))
        return self

    def __exit__(self, *args):
        """
        Do things after the code inside the 'with Recording' block.

        Parameters
        ----------
        *args : array
            Solver recording requires extra args.
        """
        requester = self.recording_requester()
        stack = requester._recording_iter.stack

        # Determine if recording is justified.
        for stack_item in stack:
            if stack_item[0] in ('_run_apply', '_compute_totals'):
                do_recording = False
                break
        else:
            if self._is_solver:
                requester.record_iteration(abs=self.abs, rel=self.rel)
            else:
                requester.record_iteration()

        # Enable the following line for stack debugging.
        # print_recording_iteration_stack()

        stack.pop()
