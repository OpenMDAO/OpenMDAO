import numpy as np
import openmdao.api as om
import unittest

from openmdao.utils.assert_utils import assert_near_equal

# ==============================================================================
# Extension modules
# ==============================================================================

# test 2d problem for the schur complement solver
# R1 = (x1 - x2**3 + 1)**3 - x2**3
# R2 = 2x1 + 3x2 - 5


case_solutions = {'liu': np.array([1.0, 1.0]),
                  'simple1': np.array([1.0, 1.0]),
                  'simple2': np.array([1.0, 1.0]),
                  'simple3': np.array([1.0, 1.0]),
                  'simple4': np.array([1.0, 1.0]),
                  'center1': np.array([0.0, 0.0]),
                  'center2': np.array([0.0, 0.0]),
                  'center3': np.array([0.0, 0.0]),
                  'center4': np.array([0.0, 0.0])}


class TestSchurSolvers(unittest.TestCase):

    def test_result(self):
        # for case in ('liu', 'simple1', 'simple2', 'simple3', 'simple4',
        #              'center1', 'center2', 'center3', 'center4'):
        for case in ('liu', 'center1',):
            # for solver in ('nlbgs', 'hierarchical_newton', 'schur', 'newton_schur',
            #                'schur_BGS', 'newton_BGS', 'schur_schur', 'NLrun_schur'):
            for solver in ('schur_schur',):# 'schur_BGS', 'newton_schur', 'schur_schur',):
                for mode in ('fwd', 'rev',):
                    with self.subTest(f'{case=} {solver=} {mode=}'):
                        _run_test_problem(case=case, solver=solver, mode=mode,
                                        solution=case_solutions[case])


def _run_test_problem(case, solver, mode, solution):

    # create the openmdao components
    class Comp1(om.ImplicitComponent):
        def initialize(self):
            pass

        def setup(self):
            self.add_output("x1")
            self.add_input("x2")

        def setup_partials(self):
            self.declare_partials(of="*", wrt="*", method="cs")

        def apply_nonlinear(self, inputs, outputs, residuals):
            x1 = outputs["x1"]
            x2 = inputs["x2"]
            if case == "liu":
                residuals["x1"] = (x1 - x2**3 + 1) ** 3 - x2**3
            elif case == "simple1":
                residuals["x1"] = (x1 - 1) ** 2 + (x2 - 1)
            elif case == "simple2":
                residuals["x1"] = (x1 - 1) + 5 * (x2 - 1) ** 2
            elif case == "simple3":
                residuals["x1"] = (x1 - 1) + 5 * (x2 - 1) ** 3
            elif case == "simple4":
                residuals["x1"] = (x1 - 1) + 5 * (x2 - 1) ** 3

            elif case == "center1":
                residuals["x1"] = 5 * x1 * 0.5 + 20 * ((x2 * 0.5) ** 3)
            elif case == "center2":
                residuals["x1"] = 5 * x1 * 0.5 + 20 * ((x2 * 0.5) ** 3)

            elif case == "center3":
                residuals["x1"] = 40 * x1 + 20 * ((1 * x2) ** 3)
            elif case == "center4":
                residuals["x1"] = 40 * x1 + 20 * ((1 * x2) ** 3)


    class Comp2(om.ImplicitComponent):
        def initialize(self):
            pass

        def setup(self):
            self.add_output("x2")
            self.add_input("x1")

        def setup_partials(self):
            self.declare_partials(of="*", wrt="*", method="cs")

        def apply_nonlinear(self, inputs, outputs, residuals):
            x1 = inputs["x1"]
            x2 = outputs["x2"]
            # original residual from Liu2015
            if case == "liu":
                # the sub problem they had is a linear system, so a pure Newton has to follow that line, which is why its a difficult problem
                residuals["x2"] = 2 * x1 + 3 * x2 - 5

            # modified residuals that makes the component-jacobian zero
            elif case == "modified_linear":
                # this is a "linear" sub problem. convergence follows x1 = 1 line
                residuals["x2"] = x1 - 1

            elif case == "modified_nonlinear":
                # this is the "nonlinear" sub problem. dependency to x1 is also linear here
                residuals["x2"] = (x1 - 1) ** 2

            elif case == "simple":
                residuals["x2"] = (x2 - 1) ** 2 + (x1 - 1)
            elif case == "simple2":
                residuals["x2"] = (x2 - 1) + 10 * (x1 - 1) ** 2
            elif case == "simple3":
                residuals["x2"] = 10 * (x2 - 1) + 10 * (x1 - 1) ** 3
            elif case == "simple4":
                residuals["x2"] = 10 * (x1 - 1) ** 3

            elif case == "center1":
                residuals["x2"] = x2 * 0.5 + ((x1 * 0.5) ** 3)
            elif case == "center2":
                residuals["x2"] = 1 * ((x1 * 0.5) ** 3)

            elif case == "center3":
                residuals["x2"] = 8 * x2 + 1 * ((1 * x1) ** 3)
            elif case == "center4":
                residuals["x2"] = x1**3


    p = om.Problem()

    comp1 = p.model.add_subsystem("comp1", Comp1())

    if solver == "newton":
        # the parent is doing a newton, so we just run once
        comp1.nonlinear_solver = om.NonlinearRunOnce()
        comp1.linear_solver = om.LinearRunOnce()
    # TODO with ths schur solver we solve the subsystem with its own newton
    elif (
        solver == "nlbgs"
        or solver == "hierarchical_newton"
        or solver == "schur"
        or solver == "newton_schur"
        or solver == "schur_BGS"
        or solver == "newton_BGS"
        or solver == "schur_schur"
        or solver == "NLrun_schur"
    ):
        # parent is doing NLBGS, so we are on our own to solve this subsystem
        comp1.nonlinear_solver = om.NewtonSolver(atol=1e-13, rtol=1e-13, solve_subsystems=False, maxiter=40)
        comp1.linear_solver = om.DirectSolver()

    comp2 = p.model.add_subsystem("comp2", Comp2())

    if (
        solver == "newton"
        or solver == "schur"
        or solver == "newton_schur"
        or solver == "schur_BGS"
        or solver == "newton_BGS"
        or solver == "schur_schur"
        or solver == "NLrun_schur"
    ):
        # the parent is doing a newton, so we just run once
        comp2.nonlinear_solver = om.NonlinearRunOnce()
        comp2.linear_solver = om.LinearRunOnce()
    elif solver == "nlbgs" or solver == "hierarchical_newton":
        # parent is doing NLBGS, so we are on our own to solve this subsystem
        comp2.nonlinear_solver = om.NewtonSolver(atol=1e-15, rtol=1e-15, solve_subsystems=False, maxiter=40)
        comp2.linear_solver = om.DirectSolver()

    p.model.connect("comp1.x1", "comp2.x1")
    p.model.connect("comp2.x2", "comp1.x2")

    if solver == "newton":
        p.model.nonlinear_solver = om.NewtonSolver(atol=1e-12, rtol=1e-12, solve_subsystems=False, maxiter=40)
        p.model.linear_solver = om.DirectSolver()

    elif solver == "nlbgs":
        p.model.nonlinear_solver = om.NonlinearBlockGS(atol=1e-12, rtol=1e-12, maxiter=20, use_apply_nonlinear=True)
        p.model.linear_solver = om.LinearBlockGS()

    elif solver == "broyden":
        p.model.nonlinear_solver = om.BroydenSolver(
            state_vars=["comp2.x2"],
        )
        p.model.linear_solver = om.DirectSolver()
        p.model.nonlinear_solver.linear_solver = om.DirectSolver()

    elif solver == "hierarchical_newton":
        p.model.nonlinear_solver = om.NewtonSolver(atol=1e-12, rtol=1e-12, solve_subsystems=True, maxiter=40)
        p.model.linear_solver = om.DirectSolver()

    elif solver == "schur":
        p.model.nonlinear_solver = om.NonlinearSchurSolver(
            atol=1e-14,
            rtol=1e-14,
            solve_subsystems=True,
            maxiter=60,
            max_sub_solves=60,
            mode_nonlinear=mode,
            bounds={"lower": [-0.25], "upper": [0.25]},
        )
        p.model.linear_solver = om.DirectSolver()

    elif solver == "newton_schur":
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, maxiter=60, max_sub_solves=60
        )
        p.model.linear_solver = om.LinearSchur(mode_linear=mode)

    elif solver == "schur_BGS":
        p.model.nonlinear_solver = om.NonlinearSchurSolver(
            atol=1e-14,
            rtol=1e-14,
            solve_subsystems=True,
            maxiter=60,
            max_sub_solves=60,
            mode_nonlinear=mode,
        )
        p.model.linear_solver = om.LinearBlockGS()
    elif solver == "newton_BGS":
        p.model.nonlinear_solver = om.NewtonSolver(
            atol=1e-14, rtol=1e-14, solve_subsystems=True, maxiter=60, max_sub_solves=60
        )
        p.model.linear_solver = om.LinearBlockGS()
    elif solver == "schur_schur":
        p.model.nonlinear_solver = om.NonlinearSchurSolver(
            atol=1e-14,
            rtol=1e-14,
            solve_subsystems=True,
            maxiter=60,
            max_sub_solves=60,
            mode_nonlinear=mode,
        )
        p.model.linear_solver = om.LinearSchur(mode_linear=mode)
    elif solver == "NLrun_schur":
        p.model.nonlinear_solver = om.NonlinearRunOnce()
        p.model.linear_solver = om.LinearSchur(mode_linear=mode)
    p.setup(mode=mode)

    ###
    ### INITIAL GUESSES
    ###
    delta_init = 0.5
    x_init = np.array(
        [
            [solution[0] - delta_init, solution[1] - delta_init],
            [solution[0] - delta_init, solution[1] + delta_init],
            # [solution[0] + delta_init, solution[1] - delta_init],
            # [solution[0] + delta_init, solution[1] + delta_init],
            # [solution[0], solution[1] - delta_init],
            # [solution[0], solution[1] + delta_init],
            # [solution[0] - delta_init, solution[1]],
            # [solution[0] + delta_init, solution[1]],
            # [solution[0] + 0.6, solution[1] + 0.45],
        ]
    )

    n_xinit = x_init.shape[0]
    x_hist = []  # we will append the history from all cases here

    for ii in range(n_xinit):
        # set the initial values
        p.setup(mode=mode)
        p.model.set_solver_print(level=-1, depth=3)

        p["comp1.x1"] = x_init[ii, 0]
        p["comp2.x2"] = x_init[ii, 1]

        recorder = om.SqliteRecorder(f"solver_{ii}.sql")
        nlsolver = p.model.nonlinear_solver
        nlsolver.add_recorder(recorder)
        nlsolver.recording_options["record_abs_error"] = True
        nlsolver.recording_options["record_rel_error"] = True
        nlsolver.recording_options["record_outputs"] = True
        nlsolver.recording_options["record_inputs"] = True

        p.run_model()

        assert_near_equal([p.get_val('comp1.x1')[0], p.get_val('comp2.x2')[0]],
                          solution,
                          tolerance=1.0E-12)

        p.cleanup()

        # --- Load the recorded data ---
        cr = om.CaseReader(p.get_outputs_dir() / f'solver_{ii}.sql')
        solver_cases = cr.list_cases("root.nonlinear_solver", out_stream=None)
        x1_hist = []
        x2_hist = []

        # if we are doing the NLBGS solver, we also need to add the initial guess bec. it doesnt record it for some reason
        if (
            solver == "nlbgs"
            or solver == "schur"
            or solver == "newton_schur"
            or solver == "schur_BGS"
            or solver == "newton_BGS"
            or solver == "schur_schur"
            or solver == "NLrun_schur"
        ):
            x1_hist.append(x_init[ii, 0])
            x2_hist.append(x_init[ii, 1])

        for case_id in solver_cases:
            # put the convergence history in an array
            x1_hist.append(cr.get_case(case_id).outputs["comp1.x1"][0])
            x2_hist.append(cr.get_case(case_id).outputs["comp2.x2"][0])

        x_hist.append((x1_hist, x2_hist))


    # for ii in range(len(x_hist)):
    #     print(ii, x_hist[ii], "\n\n")
    # quit()

    # check some totals
    # p.check_partials()

    # p.model.list_inputs()
    # p.model.list_outputs(residuals=True)

    ###
    ### create a contour plot of the residual
    ###

    # import matplotlib.pyplot as plt

    # # need to set the solver to nonlinear run once
    # p.model.nonlinear_solver = om.NewtonSolver(atol=1e-12, rtol=1e-12, solve_subsystems=False, maxiter=0)
    # p.model.linear_solver = om.LinearRunOnce()
    # p.model.set_solver_print(level=-1)
    # p.setup(mode=mode)

    # nx1 = 201
    # nx2 = 201
    # contour_delta = 1.0
    # x1 = np.linspace(solution[0] - contour_delta, solution[0] + contour_delta, nx1)
    # x2 = np.linspace(solution[1] - contour_delta, solution[1] + contour_delta, nx2)
    # X1, X2 = np.meshgrid(x1, x2)
    # RES_LOG = np.zeros_like(X1)
    # RES_L2 = np.zeros_like(X1)

    # # loop over x1 and x2 and set the residual array
    # for ii in range(nx1):
    #     for jj in range(nx2):
    #         x1 = X1[ii, jj]
    #         x2 = X2[ii, jj]

    #         # set the values
    #         p["comp1.x1"] = x1
    #         p["comp2.x2"] = x2

    #         # run the model
    #         p.run_model()

    #         # get the residual
    #         r1 = p.model._residuals["comp1.x1"]
    #         r2 = p.model._residuals["comp2.x2"]

    #         RES_LOG[ii, jj] = np.log(np.sqrt(r1**2 + r2**2) + 1)
    #         RES_L2[ii, jj] = np.sqrt((r1**2 + r2**2))
    #         # to just show the residual norm itself, which appears small
    #         # RES[ii, jj] = np.sqrt(r1 ** 2 + r2 ** 2)

    # ###
    # ### PLOTTING PARAMETERS
    # ###
    # plt.rc("text", usetex=True)
    # plt.rc("font", size=22)

    # # cm_dict = niceplots.get_niceColors()
    # cm = [
    #     "Blue",
    #     "Orange"
    # ]

    # ###
    # ### LOG PLOT
    # ###

    # plot_names = ["log", "l2"]

    # for kk, RES in enumerate([RES_LOG, RES_L2]):
    #     fig = plt.figure(figsize=(8, 6))
    #     ax = fig.add_subplot(111)

    #     CS = ax.contourf(X1, X2, RES, 50, alpha=1.0)
    #     ax.contour(X1, X2, RES, 50, alpha=0.5, colors="w", linewidths=0.2)
    #     cbar = fig.colorbar(CS, ax=ax, shrink=0.9)
    #     if kk == 0:
    #         cbar.set_label(r"$\log ( || \mathcal{R}|| + 1)$", rotation=0, labelpad=10)
    #     else:
    #         cbar.set_label(r"$|| \mathcal{R}||_2$", rotation=0, labelpad=10)

    #     cbar.set_ticks([np.min(RES), np.max(RES)])

    #     # loop over all initial guesses
    #     for ii in range(n_xinit):
    #         x1_hist = x_hist[ii][0]
    #         x2_hist = x_hist[ii][1]
    #         # print(ii, x1_hist, x2_hist)

    #         # plot the solver convergence path
    #         ax.plot(
    #             x1_hist,
    #             x2_hist,
    #             label=" ",
    #             linewidth=2,
    #             marker=".",
    #             markersize=6,
    #             markeredgewidth=0.4,
    #             markeredgecolor="w",
    #             c=cm[ii],
    #         )

    #         # for jj in range(len(x1_hist)):
    #         #     # just print the first 4 iters
    #         #     if jj < 4:
    #         #         ax.annotate(jj, (x1_hist[jj], x2_hist[jj]), c=cm[ii])

    #     # put at star at the solution
    #     ax.scatter([solution[0]], [solution[1]], marker="*", color="r", s=120)

    #     ax.set_xlabel(r"$u_1$")
    #     ax.set_ylabel(r"$u_2$", rotation=0, labelpad=10)
    #     ax.set_xticks([solution[0] - contour_delta, solution[0], solution[0] + contour_delta])
    #     ax.set_yticks([solution[1] - contour_delta, solution[1], solution[1] + contour_delta])
    #     # ax.set_title(r"\rm Simple Nonlinear Problem")
    #     plt.axis([solution[0] - contour_delta, solution[0] + contour_delta, solution[1] - contour_delta, solution[1] + contour_delta])
    #     ax.set_aspect("equal")

    #     plt.tight_layout()
    #     fig.suptitle(f'{case}\n{solver}')
    #     plt.savefig(f"{plot_names[kk]}_{case}_{solver}_problem_contour.jpg", dpi=600)
    #     # plt.savefig(f"{case}_problem_contour_{plot_names[kk]}.pdf")


if __name__ == '__main__':
    unittest.main()

    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--case", type=str, default="center3")
    # parser.add_argument("--solver", type=str, default="schur_schur")
    # parser.add_argument("--mode", type=str, default="rev")
    # args = parser.parse_args()

    # _run_test_problem(case=args.case, solver=args.solver, mode=args.mode,
    #                   solution=case_solutions[args.case])