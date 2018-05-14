"""Management of iteration stack for recording."""
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
    """

    def __init__(self):
        """
        Initialize.
        """
        self.stack = []


recording_iteration = _RecIteration()


def print_recording_iteration_stack():
    """
    Print the record iteration stack.

    Used for debugging.
    """
    print()
    for name, iter_count in reversed(recording_iteration.stack):
        print('^^^', name, iter_count)
    print(60 * '^')


def get_formatted_iteration_coordinate():
    """
    Format the iteration coordinate into human-readable form.

    'rank0:pyoptsparsedriver|6|root._solve_nonlinear|6|mda._solve_nonlinear|6|mda.d1._solve_nonlinear|45'

    Returns
    -------
    str :
        the iteration coordinate formatted in our proprietary way.
    """
    separator = '|'
    iteration_coord_list = []

    for name, iter_count in recording_iteration.stack:
        iteration_coord_list.append('{}{}{}'.format(name, separator, iter_count))

    if MPI and MPI.COMM_WORLD.rank > 0:
        rank = MPI.COMM_WORLD.rank
    else:
        rank = 0

    formatted_iteration_coordinate = ':'.join(["rank%d" % rank,
                                               separator.join(iteration_coord_list)])
    return formatted_iteration_coordinate


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
    recording_requester : object
        The object that wants to be recorded.
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
        self.recording_requester = recording_requester
        self.abs = 0
        self.rel = 0

        from openmdao.solvers.solver import Solver
        self._is_solver = isinstance(self.recording_requester, Solver)

    def __enter__(self):
        """
        Do things before the code inside the 'with Recording' block.

        Returns
        -------
        self : object
            self
        """
        recording_iteration.stack.append((self.name, self.iter_count))
        return self

    def __exit__(self, *args):
        """
        Do things after the code inside the 'with Recording' block.

        Parameters
        ----------
        *args : array
            Solver recording requires extra args.
        """
        # Determine if recording is justified.
        do_recording = True

        for stack_item in recording_iteration.stack:
            if stack_item[0] in ('_run_apply', '_compute_totals'):
                do_recording = False
                break

        if do_recording:
            if self._is_solver:
                self.recording_requester.record_iteration(abs=self.abs, rel=self.rel)
            else:
                self.recording_requester.record_iteration()

        self.recording_requester = None

        # Enable the following line for stack debugging.
        # print_recording_iteration_stack()

        recording_iteration.stack.pop()
