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
        Iterator that yields tuples of the form (key, fd_norm, fd_opts, directional, above_abs,
        above_rel, inconsistent) for each subjac.
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

    for key, fd_opts, directional, above_abs, above_rel, inconsistent in err_iter:

        if above_abs or above_rel or inconsistent:
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

        abs_errs = derivative_info['abs error']
        rel_errs = derivative_info['rel error']
        abs_vals = derivative_info['vals_at_max_abs']
        rel_vals = derivative_info['vals_at_max_rel']
        denom_idxs = derivative_info['denom_idx']
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

        for i in range(len(abs_errs)):
            # Absolute Errors
            if directional:
                if totals and abs_errs[i].forward is not None:
                    err = _format_error(abs_errs[i].forward, abs_error_tol)
                    parts.append(f'    Max Absolute Error (Jfwd - Jfd){stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {abs_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {abs_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if ('directional_fd_rev' in derivative_info and
                        derivative_info['directional_fd_rev'][i]):
                    err = _format_error(abs_errs[i].reverse, abs_error_tol)
                    parts.append('    Max Absolute Error ([rev, fd] Dot Product Test)'
                                 f'{stepstrs[i]} : {err}')
                    fd, rev = derivative_info['directional_fd_rev'][i]
                    parts.append(f'      rev value: {rev:.6e}')
                    parts.append(f'      fd value: {fd:.6e} ({fd_desc}{stepstrs[i]})\n')
            else:
                if abs_errs[i].forward is not None:
                    err = _format_error(abs_errs[i].forward, abs_error_tol)
                    parts.append(f'    Max Absolute Error (Jfwd - Jfd){stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {abs_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {abs_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if abs_errs[i].reverse is not None:
                    err = _format_error(abs_errs[i].reverse, abs_error_tol)
                    parts.append(f'    Max Absolute Error (Jrev - Jfd){stepstrs[i]} : {err}')
                    parts.append(f'      rev value: {abs_vals[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {abs_vals[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

        if directional:
            if ('directional_fwd_rev' in derivative_info and
                    derivative_info['directional_fwd_rev']):
                err = _format_error(abs_errs[0].fwd_rev, abs_error_tol)
                parts.append(f'    Max Absolute Error ([rev, fwd] Dot Product Test) : {err}')
                fwd, rev = derivative_info['directional_fwd_rev']
                parts.append(f'      rev value: {rev:.6e}')
                parts.append(f'      fwd value: {fwd:.6e}\n')
        elif abs_errs[0].fwd_rev is not None:
            err = _format_error(abs_errs[0].fwd_rev, abs_error_tol)
            parts.append(f'    Max Absolute Error (Jrev - Jfwd) : {err}')
            parts.append(f'      rev value: {abs_vals[0].fwd_rev[0]:.6e}')
            parts.append(f'      fwd value: {abs_vals[0].fwd_rev[1]:.6e}\n')

        divname = {
            'fwd': ['Jfwd', 'Jfd'],
            'rev': ['Jrev', 'Jfd'],
            'fwd_rev': ['Jrev', 'Jfwd']
        }

        for i in range(len(abs_errs)):
            didxs = denom_idxs[i]
            divname_fwd = divname['fwd'][didxs['fwd']]
            divname_rev = divname['rev'][didxs['rev']]
            divname_fwd_rev = divname['fwd_rev'][didxs['fwd_rev']]

            # Relative Errors
            if directional:
                if totals and rel_errs[i].forward is not None:
                    err = _format_error(rel_errs[i].forward, rel_error_tol)
                    parts.append(f'    Max Relative Error (Jfwd - Jfd) / {divname_fwd}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {rel_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if ('directional_fd_rev' in derivative_info and
                        derivative_info['directional_fd_rev'][i]):
                    err = _format_error(rel_errs[i].reverse, rel_error_tol)
                    parts.append(f'    Max Relative Error ([rev, fd] Dot Product Test) '
                                 f'/ {divname_rev}{stepstrs[i]} : {err}')
                    parts.append(f'      rev value: {rel_vals[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')
            else:
                if rel_errs[i].forward is not None:
                    err = _format_error(rel_errs[i].forward, rel_error_tol)
                    parts.append(f'    Max Relative Error (Jfwd - Jfd) / {divname_fwd}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      fwd value: {rel_vals[i].forward[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].forward[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

                if rel_errs[i].reverse is not None:
                    err = _format_error(rel_errs[i].reverse, rel_error_tol)
                    parts.append(f'    Max Relative Error (Jrev - Jfd) / {divname_rev}'
                                 f'{stepstrs[i]} : {err}')
                    parts.append(f'      rev value: {rel_vals[i].reverse[0]:.6e}')
                    parts.append(f'      fd value: {rel_vals[i].reverse[1]:.6e} '
                                 f'({fd_desc}{stepstrs[i]})\n')

        if rel_errs[0].fwd_rev is not None:
            if directional:
                err = _format_error(rel_errs[0].fwd_rev, rel_error_tol)
                parts.append(f'    Max Relative Error ([rev, fwd] Dot Product Test) / '
                             f'{divname_fwd_rev} : {err}')
                rev, fwd = derivative_info['directional_fwd_rev']
                parts.append(f'      rev value: {rev:.6e}')
                parts.append(f'      fwd value: {fwd:.6e}\n')
            else:
                err = _format_error(rel_errs[0].fwd_rev, rel_error_tol)
                parts.append(f'    Max Relative Error (Jrev - Jfwd) / {divname_fwd_rev} : '
                             f'{err}')
                parts.append(f'      rev value: {rel_vals[0].fwd_rev[0]:.6e}')
                parts.append(f'      fwd value: {rel_vals[0].fwd_rev[1]:.6e}\n')

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
            if abs_errs[0].forward is not None:
                if directional:
                    parts.append('    Directional Derivative (Jfwd)')
                else:
                    parts.append('    Raw Forward Derivative (Jfwd)')
                Jstr = textwrap.indent(str(Jfwd), '    ')
                parts.append(f"{Jstr}\n")

            fdtype = fd_opts['method'].upper()

            if abs_errs[0].reverse is not None:
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

            for i in range(len(abs_errs)):
                fd = fds[i]

                Jstr = textwrap.indent(str(fd), '    ')
                if directional:
                    if totals and abs_errs[i].reverse is not None:
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


def _deriv_display_compact(system, err_iter, derivatives, out_stream, totals=False,
                           show_only_incorrect=False, show_worst=False):
    """
    Print derivative error info to out_stream in a compact tabular format.

    Parameters
    ----------
    system : System
        The system for which derivatives are being displayed.
    err_iter : iterator
        Iterator that yields tuples of the form (key, fd_norm, fd_opts, directional, above_abs,
        above_rel, inconsistent) for each subjac.
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
        Tuple contains the worst relative error, corresponding table row, and table header.
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

    for key, _, directional, above_abs, above_rel, inconsistent in err_iter:

        if above_abs or above_rel or inconsistent:
            num_bad_jacs += 1

        of, wrt = key
        derivative_info = derivatives[key]

        # Informative output for responses that were declared with an index.
        indices = derivative_info.get('indices')
        if indices is not None:
            of = f'{of} (index size: {indices})'

        if directional:
            wrt = f"(d) {wrt}"

        err_desc = []
        if above_abs:
            err_desc.append(' >ABS_TOL')
        if above_rel:
            err_desc.append(' >REL_TOL')
        if inconsistent:
            err_desc.append(' <RANK INCONSISTENT>')
        if 'uncovered_nz' in derivative_info:
            err_desc.append(' <BAD SPARSITY>')
        err_desc = ''.join(err_desc)

        abs_errs = derivative_info['abs error']
        rel_errs = derivative_info['rel error']
        abs_vals = derivative_info['vals_at_max_abs']
        rel_vals = derivative_info['vals_at_max_rel']
        steps = derivative_info['steps']

        # loop over different fd step sizes
        for abs_err, rel_err, abs_val, rel_val, step in zip(abs_errs, rel_errs,
                                                            abs_vals, rel_vals,
                                                            steps):

            # use forward even if both fwd and rev are defined
            if abs_err.forward is not None:
                calc_abs = abs_err.forward
                calc_rel = rel_err.forward
                calc_abs_val_fd = abs_val.forward[1]
                calc_rel_val_fd = rel_val.forward[1]
                calc_abs_val = abs_val.forward[0]
                calc_rel_val = rel_val.forward[0]
            elif abs_err.reverse is not None:
                calc_abs = abs_err.reverse
                calc_rel = rel_err.reverse
                calc_abs_val_fd = abs_val.reverse[1]
                calc_rel_val_fd = rel_val.reverse[1]
                calc_abs_val = abs_val.reverse[0]
                calc_rel_val = rel_val.reverse[0]

            start = [of, wrt, step] if len(steps) > 1 else [of, wrt]

            if totals:
                table_data.append(start +
                                  [calc_abs_val, calc_abs_val_fd, calc_abs,
                                   calc_rel_val, calc_rel_val_fd, calc_rel,
                                   err_desc])
            else:  # partials
                if matrix_free:
                    table_data.append(start +
                                      [abs_val.forward[0], abs_val.forward[1],
                                       abs_err.forward,
                                       abs_val.reverse[0], abs_val.reverse[1],
                                       abs_err.reverse,
                                       abs_val.fwd_rev[0], abs_val.fwd_rev[1],
                                       abs_err.fwd_rev,
                                       rel_val.forward[0], rel_val.forward[1],
                                       rel_err.forward,
                                       rel_val.reverse[0], rel_val.reverse[1],
                                       rel_err.reverse,
                                       rel_val.fwd_rev[0], rel_val.fwd_rev[1],
                                       rel_err.fwd_rev,
                                       err_desc])
                else:
                    if abs_val.forward is not None:
                        table_data.append(start +
                                          [abs_val.forward[0], abs_val.forward[1],
                                           abs_err.forward,
                                           rel_val.forward[0], rel_val.forward[1],
                                           rel_err.forward,
                                           err_desc])
                    else:
                        table_data.append(start +
                                          [abs_val.reverse[0], abs_val.reverse[1],
                                           abs_err.reverse,
                                           rel_val.reverse[0], rel_val.reverse[1],
                                           rel_err.reverse,
                                           err_desc])

                    assert abs_err.fwd_rev is None
                    assert rel_err.fwd_rev is None

                # See if this subjacobian has the greater error in the derivative computation
                # compared to the other subjacobians so far
                if worst_subjac is None or rel_err.max() > worst_subjac[0]:
                    worst_subjac = (rel_err.max(), table_data[-1])

    headers = []
    if table_data:
        headers = ["'of' variable", "'wrt' variable"]
        if len(steps) > 1:
            headers.append('step')

        if matrix_free:
            headers.extend(['a(fwd val)', 'a(fd val)', 'a(fwd-fd)',
                            'a(rev val)', 'a(rchk val)', 'a(rev-fd)',
                            'a(fwd val)', 'a(rev val)', 'a(fwd-rev)',
                            'r(fwd val)', 'r(fd val)', 'r(fwd-fd)',
                            'r(rev val)', 'r(rchk val)', 'r(rev-fd)',
                            'r(fwd val)', 'r(rev val)', 'r(fwd-rev)',
                            'error desc'])
        else:
            headers.extend(['a(calc val)', 'a(fd val)', 'a(calc-fd)',
                            'r(calc val)', 'r(fd val)', 'r(calc-fd)',
                            'error desc'])

        _print_deriv_table(table_data, headers, sys_buffer)

        if show_worst and worst_subjac is not None:
            print(f"\nWorst Sub-Jacobian (relative error): {worst_subjac[0]}\n",
                  file=sys_buffer)
            _print_deriv_table([worst_subjac[1]], headers, sys_buffer)

    if not show_only_incorrect or num_bad_jacs > 0:
        out_stream.write(sys_buffer.getvalue())

    if worst_subjac is None:
        return None

    return worst_subjac + (headers,)


def _format_error(error, tol):
    """
    Format the error, flagging if necessary.

    Parameters
    ----------
    error : float
        The absolute or relative error.
    tol : float
        Tolerance above which errors are flagged

    Returns
    -------
    str
        Formatted and possibly flagged error.
    """
    if np.isnan(error) or error < tol:
        return f'{error:.6e}'
    return f'{error:.6e} *'


def _print_deriv_table(table_data, headers, out_stream, tablefmt='grid'):
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
    """
    if table_data and out_stream is not None:
        num_col_meta = {'format': '{: 1.4e}'}
        column_meta = [{}, {}]
        column_meta.extend([num_col_meta.copy() for _ in range(len(headers) - 3)])
        column_meta.append({})
        print(generate_table(table_data, headers=headers, tablefmt=tablefmt,
                             column_meta=column_meta, missing_val='n/a'), file=out_stream)
