
import time
import os

from openmdao.components.exec_comp import ExecComp
from openmdao.core.analysis_error import AnalysisError
from openmdao.utils.mpi import MPI

class ExecComp4Test(ExecComp):
    """
    A version of ExecComp for benchmarking and testing.

    Parameters
    ----------
    exprs : str or list of str
        The expressions that determine the inputs and outputs of this component.

    nl_delay : float(0.01)
        The sleep time in seconds that will occur when solve_nonlinear is called.

    lin_delay : float(0.01)
        The sleep time in seconds that will occur when apply_linear is called.

    rec_procs : tuple of the form (minprocs, maxprocs)
        Minimum and maximun MPI processes usable by this component.

    fail_rank : int or collection of int (0)
        Rank (if running under MPI) or worker number (if running under
        multiprocessing) where failures will be initiated.

    fails : list or tuple of int
        If the current self.num_nl_solves matches any of these, then this
        component will raise an exception.

    fail_hard : bool(False)
        If True and fails is not empty, this component will raise a
        RuntimeError when a failure is induced. Otherwise, an AnalysisError
        will be raised.
    """
    def __init__(self, exprs, nl_delay=0.01, lin_delay=0.01,
                 req_procs=(1,1), fail_rank=-1, fails=(),
                 fail_hard=False, **kwargs):

        super(ExecComp4Test, self).__init__(exprs, **kwargs)
        self.nl_delay = nl_delay
        self.lin_delay = lin_delay
        self.num_nl_solves = 0
        self.num_compute_partials = 0
        self.req_procs = req_procs
        self.fail_rank = fail_rank
        if isinstance(fail_rank, int):
            self.fail_rank = (self.fail_rank,)
        self.fails = fails
        self.fail_hard = fail_hard

    def compute(self, inputs, outputs):
        """
        Execute this component's assignment statements.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.

        outputs : `Vector`
            `Vector` containing outputs.
        """
        try:
            if self.comm.rank in self.fail_rank and self.num_nl_solves in self.fails:
                if self.fail_hard:
                    raise RuntimeError("OMG, a critical error!")
                else:
                    raise AnalysisError("just an analysis error")
            super(ExecComp4Test, self).compute(inputs, outputs)
            time.sleep(self.nl_delay)
        finally:
            self.num_nl_solves += 1

    def compute_partials(self, inputs, partials):
        """
        Use complex step method to update the given Jacobian.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing parameters. (p)

        partials : `Jacobian`
            Contains sub-jacobians.
        """
        super(ExecComp4Test, self).compute_partials(inputs, partials)
        time.sleep(self.lin_delay)
        self.num_compute_partials += 1
