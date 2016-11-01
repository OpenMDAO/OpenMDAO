from __future__ import division, print_function
import numpy
from six.moves import range

from solver import NonlinearSolver


class NonlinearBlockJac(NonlinearSolver):

METHOD = 'NL: NLBJ'

def _iter_execute(self):
    system = self._system
    system._transfers[None](system._inputs, system._outputs, 'fwd')
    for subsys in system._subsystems_myproc:
        subsys._solve_nonlinear()
