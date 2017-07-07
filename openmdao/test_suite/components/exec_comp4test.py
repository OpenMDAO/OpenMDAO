from __future__ import print_function

import time

from openmdao.components.exec_comp import ExecComp
from openmdao.utils.mpi import MPI


class ExecComp4Test(ExecComp):
    """
    A version of ExecComp for benchmarking and testing.

    Args
    ----
    exprs : str or list of str
        The expressions that determine the inputs and outputs of this component.

    nl_delay : float(0.01)
        The sleep time in seconds that will occur when solve_nonlinear is called.

    lin_delay : float(0.01)
        The sleep time in seconds that will occur when apply_linear is called.

    trace : bool(False)
        If True, print some mininal trace information while running.

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
                 trace=False, req_procs=(1, 1), fail_rank=0, fails=(),
                 fail_hard=False, **kwargs):

        super(ExecComp4Test, self).__init__(exprs, **kwargs)
        self.nl_delay = nl_delay
        self.lin_delay = lin_delay
        self.trace = trace
        self.num_nl_solves = 0
        self.num_apply_lins = 0
        self.req_procs = req_procs
        self.fail_rank = fail_rank
        if isinstance(fail_rank, int):
            self.fail_rank = (self.fail_rank,)
        self.fails = fails
        self.fail_hard = fail_hard

        # # make a case_rank output so that we can see in tests which rank
        # # ran a case
        # self.add_output('case_rank', 0, pass_by_obj=True)

    def get_req_procs(self):
        return self.req_procs

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
        if MPI:
            myrank = outputs['case_rank'] = MPI.COMM_WORLD.rank
        else:
            myrank = 0

        try:
            if myrank in self.fail_rank and self.num_nl_solves in self.fails:
                if self.fail_hard:
                    raise RuntimeError("OMG, a critical error!")
                else:
                    # raise AnalysisError("just an analysis error")
                    pass
            super(ExecComp4Test, self).compute(inputs, outputs)
            time.sleep(self.nl_delay)
        finally:
            self.num_nl_solves += 1

    def compute_partials(self, inputs, outputs, partials):
        """
        Use complex step method to update the given Jacobian.

        Parameters
        ----------
        inputs : `VecWrapper`
            `VecWrapper` containing parameters. (p)

        outputs : `VecWrapper`
            `VecWrapper` containing outputs and states. (u)

        partials : `Jacobian`
            Contains sub-jacobians.
        """
        self.compute_partials(inputs, outputs, partials)
        time.sleep(self.lin_delay)
        self.num_apply_lins += 1
