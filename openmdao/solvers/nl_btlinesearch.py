""" Backtracking line search using the Armijo-Goldstein condition."""

from math import isnan

import numpy as np

from openmdao.util.record_util import update_local_meta, create_local_meta

from openmdao.solvers.solver import NonlinearSolver

class BacktrackingLineSearch(NonlinearSolver):

    SOLVER = 'NL: BK_TKG'

    def __init__(self):
        super(BacktrackingLineSearch, self).__init__()

        opt = self.options
        opt['ilimit'] = 5
        opt.add_option('solve_subsystems', True,
                       desc='Set to True to solve subsystems. You may need '
                            'this for solvers nested under Newton.')
        opt.add_option('rho', 0.5,
                       desc="Backtracking step.")
        opt.add_option('c', 0.5,
                       desc="Slope check trigger.")

    def _iter_execute(self):
        """ Take the gradient calculated by the parent solver and figure out
        how far to go.

        Args
        ----

        alpha_scalar : float
            Initial over-relaxation factor as used in parent solver.

        alpha : ndarray
            Initial over-relaxation factor as used in parent solver, vector
            (so we don't re-allocate).

        base_u : ndarray
            Initial value of unknowns before the Newton step.

        base_norm : float
            Norm of the residual prior to taking the Newton step.

        fnorm : float
            Norm of the residual after taking the Newton step.


        Returns
        --------
        float
            Norm of the final residual
        """

        system = self._system

        maxiter = self.options['ilimit']
        rho = self.options['rho']
        c = self.options['c']
        iprint = self.options['iprint']
        result = system.dumat[None]

        itercount = 0
        ls_alpha = alpha_scalar

        # Further backtacking if needed.
        # The Armijo-Goldstein is basically a slope comparison --actual vs predicted.
        # We don't have an actual gradient, but we have the Newton vector that should
        # take us to zero, and our "runs" are the same, and we can just compare the
        # "rise".
        while itercount < maxiter and (base_norm - fnorm) < c*ls_alpha*base_norm:

            ls_alpha *= rho

            # If our step will violate any upper or lower bounds, then reduce
            # alpha in just that direction so that we only step to that
            # boundary.
            unknowns.vec[:] = base_u
            alpha[:] = ls_alpha
            alpha = unknowns.distance_along_vector_to_limit(alpha, result)

            unknowns.vec += alpha*result.vec
            itercount += 1

            # Metadata update
            update_local_meta(local_meta, (solver.iter_count, itercount))

            # Just evaluate the model with the new points
            if self.options['solve_subsystems']:
                system.children_solve_nonlinear(local_meta)
            system.apply_nonlinear(params, unknowns, resids, local_meta)

            fnorm = resids.norm()
            if iprint == 2:
                self.print_norm(self.print_name, system, itercount,
                                fnorm, fnorm0, indent=1, solver='LS')


        # Final residual print if you only want the last one
        if iprint == 1:
            self.print_norm(self.print_name, system, itercount,
                            fnorm, fnorm0, indent=1, solver='LS')

        if itercount >= maxiter or isnan(fnorm):

            if self.options['err_on_maxiter']:
                msg = "Solve in '{}': BackTracking failed to converge after {} " \
                      "iterations."
                raise AnalysisError(msg.format(system.pathname, maxiter))

            msg = 'FAILED to converge after %d iterations' % itercount
            fail = True
        else:
            msg = 'Converged in %d iterations' % itercount
            fail = False

        if iprint > 0 or (fail and iprint > -1 ):

            self.print_norm(self.print_name, system, itercount,
                            fnorm, fnorm0, msg=msg, indent=1, solver='LS')

        return fnorm
