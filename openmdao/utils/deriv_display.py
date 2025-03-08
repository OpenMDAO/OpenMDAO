"""
Functions used for the display of derivatives matrices.
"""

import textwrap
from io import StringIO

import numpy as np

from openmdao.utils.general_utils import add_border
from openmdao.utils.mpi import MPI
from openmdao.visualization.tables.table_builder import generate_table


def _deriv_display(system, err_iter, derivatives, rel_error_tol, abs_error_tol, out_stream,
                   fd_opts, totals=False, show_only_incorrect=False, lcons=None):
    """
    Print derivative error info to out_stream.

    Parameters
    ----------
    system : System
        The system for which derivatives are being displayed.
    err_iter : iterator
        Iterator that yields tuples of the form (key, fd_norm, fd_opts, directional, above_tol,
        inconsistent) for each subjac.
    derivatives : dict
        Dictionary containing derivative information keyed by (of, wrt).
    rel_error_tol : float
        Relative error tolerance.
    abs_error_tol : float
        Absolute error tolerance.
    out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
    fd_opts : dict
        Dictionary containing options for the finite difference.
    totals : bool
        True if derivatives are totals.
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    lcons : list or None
        For total derivatives only, list of outputs that are actually linear constraints.
    sort : bool
        If True, sort subjacobian keys alphabetically.
    """
    from openmdao.core.component import Component

    if out_stream is None:
        return

    # Match header to appropriate type.
    if isinstance(system, Component):
        sys_type = 'Component'
    else:
        sys_type = 'Group'

    sys_name = system.pathname
    sys_class_name = type(system).__name__

    if totals:
        sys_name = 'Full Model'

    num_bad_jacs = 0  # Keep track of number of bad derivative values for each component

    # Need to capture the output of a component's derivative
    # info so that it can be used if that component is the
    # worst subjac. That info is printed at the bottom of all the output
    sys_buffer = StringIO()

    if totals:
        title = "Total Derivatives"
    else:
        title = f"{sys_type}: {sys_class_name} '{sys_name}'"

    print(f"{add_border(title, '-')}\n", file=sys_buffer)
    parts = []

    for key, fd_opts, directional, above_tol, inconsistent in err_iter:

        if above_tol or inconsistent:
            num_bad_jacs += 1

        of, wrt = key
        derivative_info = derivatives[key]

        # Informative output for responses that were declared with an index.
        indices = derivative_info.get('indices')
        if indices is not None:
            of = f'{of} (index size: {indices})'

        # need this check because if directional may be list
        if isinstance(wrt, str):
            wrt = f"'{wrt}'"
        if isinstance(of, str):
            of = f"'{of}'"

        if directional:
            wrt = f"(d){wrt}"

        tol_violations = derivative_info['tol violation']
        abs_errs = derivative_info['abs error']
        rel_errs = derivative_info['rel error']
        vals_at_max_err = derivative_info['vals_at_max_error']
        steps = derivative_info['steps']

        Jfwd = derivative_info.get('J_fwd')
        Jrev = derivative_info.get('J_rev')

        if len(steps) > 1:
            stepstrs = [f", step={step}" for step in steps]
        else:
            stepstrs = [""]

        fd_desc = f"{fd_opts['method']}:{fd_opts['form']}"
        parts.append(f"  {sys_name}: {of} wrt {wrt}")
        if not isinstance(of, tuple) and lcons and of.strip("'") in lcons:
            parts[-1] += " (Linear constraint)"
        parts.append('')

        def tol_violation_str(check_str, desired_str):
            return f'({check_str} - {desired_str}) - (atol + rtol * {desired_str})'

        for i in range(len(tol_violations)):
            if directional:
                if totals and tol_violations[i].forward is not None:
                    err = _format_error(tol_violations[i].forward, 0.0)
                    parts.append(f'    Max Tolerance Violation ([fwd, fd] Dot Product Test)'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      abs error: {abs_errs[i].forward:.6e}')
                    parts.append(f'      rel error: {rel_errs[i].forward:.6e}')
                    parts.append(f'      fwd value: {vals_at_max_err[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {vals_at_max_err[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if ('directional_fd_rev' in derivative_info and
                        derivative_info['directional_fd_rev'][i]):
                    err = _format_error(tol_violations[i].reverse, 0.0)
                    parts.append(f'    Max Tolerance Violation ([rev, fd] Dot Product Test)'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      abs error: {abs_errs[i].reverse:.6e}')
                    parts.append(f'      rel error: {rel_errs[i].reverse:.6e}')
                    fd, rev = derivative_info['directional_fd_rev'][i]
                    parts.append(f'      rev value: {rev:.6e}')
                    parts.append(f'      fd value: {fd:.6e} ({fd_desc}{stepstrs[i]})\n')
            else:
                if tol_violations[i].forward is not None:
                    err = _format_error(tol_violations[i].forward, 0.0)
                    parts.append(f'    Max Tolerance Violation {tol_violation_str("Jfwd", "Jfd")}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      abs error: {abs_errs[i].forward:.6e}')
                    parts.append(f'      rel error: {rel_errs[i].forward:.6e}')
                    parts.append(f'      fwd value: {vals_at_max_err[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {vals_at_max_err[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if tol_violations[i].reverse is not None:
                    err = _format_error(tol_violations[i].reverse, 0.0)
                    parts.append(f'    Max Tolerance Violation {tol_violation_str("Jrev", "Jfd")}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      abs error: {abs_errs[i].reverse:.6e}')
                    parts.append(f'      rel error: {rel_errs[i].reverse:.6e}')
                    parts.append(f'      rev value: {vals_at_max_err[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {vals_at_max_err[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

        if directional:
            if ('directional_fwd_rev' in derivative_info and
                    derivative_info['directional_fwd_rev']):
                err = _format_error(tol_violations[0].fwd_rev, 0.0)
                parts.append(f'    Max Tolerance Violation ([rev, fwd] Dot Product Test) : {err}')
                parts.append(f'      abs error: {abs_errs[0].fwd_rev:.6e}')
                parts.append(f'      rel error: {rel_errs[0].fwd_rev:.6e}')
                fwd, rev = derivative_info['directional_fwd_rev']
                parts.append(f'      rev value: {rev:.6e}')
                parts.append(f'      fwd value: {fwd:.6e}\n')
        elif tol_violations[0].fwd_rev is not None:
            err = _format_error(tol_violations[0].fwd_rev, 0.0)
            parts.append(f'    Max Tolerance Violation {tol_violation_str("Jrev", "Jfwd")}'
                         f' : {err}')
            parts.append(f'      abs error: {abs_errs[0].fwd_rev:.6e}')
            parts.append(f'      rel error: {rel_errs[0].fwd_rev:.6e}')
            parts.append(f'      rev value: {vals_at_max_err[0].fwd_rev[0]:.6e}')
            parts.append(f'      fwd value: {vals_at_max_err[0].fwd_rev[1]:.6e}\n')

        if inconsistent:
            parts.append('\n    * Inconsistent value across ranks *\n')

        comm = system._problem_meta['comm']
        if MPI and comm.size > 1:
            parts.append(f'\n    MPI Rank {comm.rank}\n')

        if 'uncovered_nz' in derivative_info:
            uncovered_nz = list(derivative_info['uncovered_nz'])
            uncovered_threshold = derivative_info['uncovered_threshold']
            rs = np.array([r for r, _ in uncovered_nz], dtype=int)
            cs = np.array([c for _, c in uncovered_nz])
            parts.append(f'    Sparsity excludes {len(uncovered_nz)} entries which'
                         f' appear to be non-zero. (Magnitudes exceed {uncovered_threshold}) *')
            with np.printoptions(linewidth=1000, formatter={'int': lambda i: f'{i}'}):
                parts.append(f'      Rows: {rs}')
                parts.append(f'      Cols: {cs}\n')

        with np.printoptions(linewidth=240):
            # Raw Derivatives
            if tol_violations[0].forward is not None:
                if directional:
                    parts.append('    Directional Derivative (Jfwd)')
                else:
                    parts.append('    Raw Forward Derivative (Jfwd)')
                Jstr = textwrap.indent(str(Jfwd), '    ')
                parts.append(f"{Jstr}\n")

            fdtype = fd_opts['method'].upper()

            if tol_violations[0].reverse is not None:
                if directional:
                    if totals:
                        parts.append('    Directional Derivative (Jrev) Dot Product')
                    else:
                        parts.append('    Directional Derivative (Jrev)')
                else:
                    parts.append('    Raw Reverse Derivative (Jrev)')
                Jstr = textwrap.indent(str(Jrev), '    ')
                parts.append(f"{Jstr}\n")

            try:
                fds = derivative_info['J_fd']
            except KeyError:
                fds = [0.]

            for i in range(len(tol_violations)):
                fd = fds[i]

                Jstr = textwrap.indent(str(fd), '    ')
                if directional:
                    if totals and tol_violations[i].reverse is not None:
                        parts.append(f'    Directional {fdtype} Derivative (Jfd) '
                                     f'Dot Product{stepstrs[i]}\n{Jstr}\n')
                    else:
                        parts.append(f"    Directional {fdtype} Derivative (Jfd)"
                                     f"{stepstrs[i]}\n{Jstr}\n")
                else:
                    parts.append(f"    Raw {fdtype} Derivative (Jfd){stepstrs[i]}"
                                 f"\n{Jstr}\n")

        parts.append(' -' * 30)
        parts.append('')

    sys_buffer.write('\n'.join(parts))

    if not show_only_incorrect or num_bad_jacs > 0:
        out_stream.write(sys_buffer.getvalue())


def _print_tv(tol_violation):
    """
    Enclose the tolerance violation in parentheses if it is negative.

    Parameters
    ----------
    tol_violation : float
        The tolerance violation.

    Returns
    -------
    str
        The formatted tolerance violation.
    """
    if tol_violation < 0:
        return f'({tol_violation:.6e})'
    return f'{tol_violation:.6e}'


def _deriv_display_compact(system, err_iter, derivatives, out_stream, totals=False,
                           show_only_incorrect=False, show_worst=False):
    """
    Print derivative error info to out_stream in a compact tabular format.

    Parameters
    ----------
    system : System
        The system for which derivatives are being displayed.
    err_iter : iterator
        Iterator that yields tuples of the form (key, fd_norm, fd_opts, directional, above_tol,
        inconsistent) for each subjac.
    derivatives : dict
        Dictionary containing derivative information keyed by (of, wrt).
    out_stream : file-like object
            Where to send human readable output.
            Set to None to suppress.
    totals : bool
        True if derivatives are totals.
    show_only_incorrect : bool, optional
        Set to True if output should print only the subjacs found to be incorrect.
    show_worst : bool
        Set to True to show the worst subjac.

    Returns
    -------
    tuple or None
        Tuple contains the worst tolerance violation, corresponding table row, and table header.
    """
    if out_stream is None:
        return

    from openmdao.core.component import Component

    # Match header to appropriate type.
    if isinstance(system, Component):
        sys_type = 'Component'
    else:
        sys_type = 'Group'

    sys_name = system.pathname
    sys_class_name = type(system).__name__
    matrix_free = system.matrix_free and not totals

    if totals:
        sys_name = 'Full Model'

    num_bad_jacs = 0  # Keep track of number of bad derivative values for each component

    # Need to capture the output of a component's derivative
    # info so that it can be used if that component is the
    # worst subjac. That info is printed at the bottom of all the output
    sys_buffer = StringIO()

    if totals:
        title = "Total Derivatives"
    else:
        title = f"{sys_type}: {sys_class_name} '{sys_name}'"

    print(f"{add_border(title, '-')}\n", file=sys_buffer)

    table_data = []
    worst_subjac = None

    for key, _, directional, above_tol, inconsistent in err_iter:

        if above_tol or inconsistent:
            num_bad_jacs += 1

        of, wrt = key
        derivative_info = derivatives[key]

        # Informative output for responses that were declared with an index.
        indices = derivative_info.get('indices')
        if indices is not None:
            of = f'{of} (index size: {indices})'

        if directional:
            wrt = f"(d) {wrt}"

        tol_violations = derivative_info['tol violation']
        vals_at_max_err = derivative_info['vals_at_max_error']
        steps = derivative_info['steps']

        # loop over different fd step sizes
        for tol_violation, abs_val, step in zip(tol_violations, vals_at_max_err, steps):

            err_desc = []
            maxtv = tol_violation.max(use_abs=False)
            if maxtv > 0.:
                err_desc.append(f'{maxtv: .6e}>TOL')
            if inconsistent:
                err_desc.append(' <RANK INCONSISTENT>')
            if 'uncovered_nz' in derivative_info:
                err_desc.append(' <BAD SPARSITY>')
            err_desc = ''.join(err_desc)

            start = [of, wrt, step] if len(steps) > 1 else [of, wrt]

            if totals:
                # use forward even if both fwd and rev are defined
                if tol_violation.forward is not None:
                    calc_abs = _print_tv(tol_violation.forward)
                    calc_abs_val_fd = abs_val.forward[1]
                    calc_abs_val = abs_val.forward[0]
                elif tol_violation.reverse is not None:
                    calc_abs = _print_tv(tol_violation.reverse)
                    calc_abs_val_fd = abs_val.reverse[1]
                    calc_abs_val = abs_val.reverse[0]

                table_data.append(start + [calc_abs_val, calc_abs_val_fd, calc_abs, err_desc])
            else:  # partials
                if matrix_free:
                    table_data.append(start +
                                      [abs_val.forward[0], abs_val.forward[1],
                                       _print_tv(tol_violation.forward),
                                       abs_val.reverse[0], abs_val.reverse[1],
                                       _print_tv(tol_violation.reverse),
                                       abs_val.fwd_rev[0], abs_val.fwd_rev[1],
                                       _print_tv(tol_violation.fwd_rev),
                                       err_desc])
                else:
                    if abs_val.forward is not None:
                        table_data.append(start +
                                          [abs_val.forward[0], abs_val.forward[1],
                                           _print_tv(tol_violation.forward), err_desc])
                    else:
                        table_data.append(start +
                                          [abs_val.reverse[0], abs_val.reverse[1],
                                           _print_tv(tol_violation.reverse), err_desc])

                # See if this subjacobian has the greater error in the derivative computation
                # compared to the other subjacobians so far
                if worst_subjac is None or tol_violation.max(use_abs=False) > worst_subjac[0]:
                    worst_subjac = (tol_violation.max(use_abs=False), table_data[-1])

    headers = []
    if table_data:
        headers = ["'of' variable", "'wrt' variable"]
        if len(steps) > 1:
            headers.append('step')

        column_meta = {}

        if matrix_free:
            column_meta[4] = {'align': 'right'}
            column_meta[7] = {'align': 'right'}
            column_meta[10] = {'align': 'right'}
            headers.extend(['fwd val', 'fd val', '(fwd-fd) - (a + r*fd)',
                            'rev val', 'fd val', '(rev-fd) - (a + r*fd)',
                            'fwd val', 'rev val', '(fwd-rev) - (a + r*rev)',
                            'error desc'])
        else:
            column_meta[4] = {'align': 'right'}
            headers.extend(['calc val', 'fd val', '(calc-fd) - (a + r*fd)',
                            'error desc'])

        _print_deriv_table(table_data, headers, sys_buffer, col_meta=column_meta)

        if worst_subjac is not None and worst_subjac[0] <= 0:
            worst_subjac = None

        if show_worst and worst_subjac is not None:
            if worst_subjac[0] > 0:
                print(f"\nWorst Sub-Jacobian (tolerance violation): {worst_subjac[0]}\n",
                      file=sys_buffer)
                _print_deriv_table([worst_subjac[1]], headers, sys_buffer, col_meta=column_meta)

    if not show_only_incorrect or num_bad_jacs > 0:
        out_stream.write(sys_buffer.getvalue())

    if worst_subjac is None:
        return None

    return worst_subjac + (headers, column_meta)


def _format_error(error, tol):
    """
    Format the error, flagging if necessary.

    Parameters
    ----------
    error : float
        The error.
    tol : float
        Tolerance above which errors are flagged

    Returns
    -------
    str
        Formatted and possibly flagged error.
    """
    if np.isnan(error) or error < tol:
        return f'({error:.6e})'
    return f'{error:.6e} *'


def _print_deriv_table(table_data, headers, out_stream, tablefmt='grid', col_meta=None):
    """
    Print a table of derivatives.

    Parameters
    ----------
    table_data : list
        List of lists containing the table data.
    headers : list
        List of column headers.
    out_stream : file-like object
        Where to send human readable output.
        Set to None to suppress.
    tablefmt : str
        The table format to use.
    col_meta : dict
        Dict containing metadata keyed by column index.
    """
    if table_data and out_stream is not None:
        num_col_meta = {'format': '{: .6e}'}
        column_meta = [{}, {}]
        column_meta.extend([num_col_meta.copy() for _ in range(len(headers) - 3)])
        column_meta.append({})
        if col_meta:
            for i, meta in col_meta.items():
                column_meta[i].update(meta)

        print(generate_table(table_data, headers=headers, tablefmt=tablefmt,
                             column_meta=column_meta, missing_val='n/a'), file=out_stream)
