import traceback
import textwrap

from openmdao.utils.mpi import MPI
from openmdao.utils.general_utils import make_traceback, env_truthy


class ErrCollector:
    def __init__(self, name):
        self._name = name
        self.errors = []
        self.err_ids = set()

    def __bool__(self):
        return bool(self.errors)

    def __len__(self):
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def get_errors(self):
        return self.errors

    def append(self, error):
        self.errors.append(error)

    def update(self, errors):
        self.errors.extend(errors)

    def collect_error(self, msg, exc_type=None, tback=None, ident=None):
        """
        Save an error message to raise as an exception later.

        Parameters
        ----------
        msg : str
            The connection error message to be saved.
        exc_type : class or None
            The type of exception to be raised if this error is the only one collected.
        tback : traceback or None
            The traceback of a caught exception.
        ident : int
            Identifier of the object responsible for issuing the error.
        """
        if exc_type is None:
            exc_type = RuntimeError

        if tback is None:
            tback = make_traceback()

        # if saved_errors is None it means we have already finished setup and all errors should
        # be raised as exceptions immediately.
        if self.errors is None or env_truthy('OPENMDAO_FAIL_FAST'):
            raise exc_type(msg).with_traceback(tback)

        self.errors.append((ident, msg, exc_type, tback))

    def check_collected_errors(self, comm):
        """
        If any collected errors are found, raise an exception containing all of them.
        """
        if self.errors is None:
            return

        unique_errors = self.get_unique_saved_errors(comm)

        # set the errors to None so that all future calls will immediately raise an exception.
        self.errors = None

        if unique_errors:
            # self.model.display_conn_graph()
            # self.model.display_dataflow_graph()
            final_msg = [f"\nCollected errors for problem '{self._name}':"]
            for _, msg, exc_type, tback in unique_errors:
                final_msg.append(textwrap.indent(msg, '   '))

                # if there's only one error, include its traceback if it exists.
                if len(unique_errors) == 1:
                    if isinstance(tback, str):
                        final_msg.append('Traceback (most recent call last):')
                        final_msg.append(tback)
                    else:
                        raise exc_type('\n'.join(final_msg)).with_traceback(tback)

            raise RuntimeError('\n'.join(final_msg))

    def any_rank_has_saved_errors(self, comm):
        """
        Return True if any rank has saved errors.

        Parameters
        ----------
        comm : MPI.Comm
            The MPI communicator to use.

        Returns
        -------
        bool
            True if any rank has errors.
        """
        if comm.size == 1:
            return bool(self.errors)
        else:
            if MPI and comm is not None and comm.size > 1:
                return comm.allreduce(len(self.errors), op=MPI.SUM) > 0
            else:
                return bool(self.errors)

    def get_unique_saved_errors(self, comm):
        """
        Get a list of unique saved errors.

        Returns
        -------
        list
            List of unique saved errors.
        """
        unique_errors = []
        if self.any_rank_has_saved_errors(comm):
            # traceback won't pickle, so convert to string
            if comm.size > 1:
                saved = [(ident, msg, exc_type, ''.join(traceback.format_tb(tback)))
                            for ident, msg, exc_type, tback in self.errors]
                all_errors = comm.allgather(saved)
            else:
                all_errors = [self.errors]

            seen = set()
            for errors in all_errors:
                for ident, msg, exc_type, tback in errors:
                    if (ident is None and msg not in seen) or ident not in seen:
                        unique_errors.append((ident, msg, exc_type, tback))
                        seen.add(ident)
                        seen.add(msg)

        return unique_errors
