"""
Routines to compute coloring for use with simultaneous derivatives.
"""
import os
import sys
import time
import warnings
import json
import pickle
import inspect
import traceback
from collections import OrderedDict, defaultdict
from itertools import combinations, chain
from distutils.version import LooseVersion
from contextlib import contextmanager
from pprint import pprint
from itertools import groupby

import numpy as np
from scipy.sparse.compressed import get_index_dtype

from openmdao.jacobians.jacobian import Jacobian
from openmdao.utils.array_utils import array_viz
from openmdao.utils.general_utils import simple_warning, prom2ivc_src_dict
import openmdao.utils.hooks as hooks
from openmdao.utils.mpi import MPI
from openmdao.utils.file_utils import _load_and_exec


CITATIONS = """
@article{Coleman+VermaSISC1998,
  author  = {Thomas F. Coleman and Arun Verma},
  title   = {The Efficient Computation of Sparse Jacobian Matrices Using Automatic Differentiation},
  journal = {SIAM Journal of Scientific Computing},
  year    = 1998,
  number  = 4,
  pages   = {1210-1233},
  month   = 7,
  volume  = 19
}
"""

# If this is True, then IF simul coloring is specified, use it.
# If False, don't use it regardless.
# The command line total_coloring command makes this False when generating a
# new coloring.
_use_total_sparsity = True

# If this is True, then IF partial/semi-total coloring is specified, use it.
# If False, don't use it regardless.
# The command line partial_coloring command makes this False when generating a
# new partial/semi-total coloring.
_use_partial_sparsity = True

# If True, ignore use_fixed_coloring if the coloring passed to it is _STD_COLORING_FNAME.
# This is used when the 'openmdao partial_coloring' or 'openmdao total_coloring' commands
# are running, because the intent there is to generate new coloring files regardless of
# whether use_fixed_coloring was called.
_force_dyn_coloring = False

# used as an indicator that we should automatically name coloring file based on class module
# path or system pathname
_STD_COLORING_FNAME = object()


# default values related to the computation of a sparsity matrix
_DEF_COMP_SPARSITY_ARGS = {
    'tol': 1e-25,     # use this tolerance to determine what's a zero when determining sparsity
    'orders': None,   # num orders += around 'tol' for the tolerance sweep when determining sparsity
    'num_full_jacs': 3,      # number of full jacobians to generate before computing sparsity
    'perturb_size': 1e-9,    # size of input/output perturbation during generation of sparsity
    'min_improve_pct': 5.,   # don't use coloring unless at least 5% decrease in number of solves
    'show_summary': True,    # if True, print a short summary of the coloring
    'show_sparsity': False,  # if True, show a plot of the sparsity
}


# A dict containing colorings that have been generated during the current execution.
# When a dynamic coloring is specified for a particular class and per_instance is False,
# this dict can be checked for an existing class version of the coloring that can be used
# for that instance.
_CLASS_COLORINGS = {}


# numpy versions before 1.12 don't use the 'axis' arg passed to count_nonzero and always
# return an int instead of an array of ints, so create our own function for those versions.
if LooseVersion(np.__version__) >= LooseVersion("1.12"):
    _count_nonzeros = np.count_nonzero
else:
    def _count_nonzeros(arr, axis=None):
        if axis == 1:  # rows
            count = np.empty(arr.shape[0], dtype=int)
            for row in range(arr.shape[0]):
                count[row] = np.count_nonzero(arr[row])
        elif axis == 0:  # cols
            count = np.empty(arr.shape[1], dtype=int)
            for col in range(arr.shape[1]):
                count[col] = np.count_nonzero(arr[:, col])
        else:
            return np.count_nonzero(arr)

        return count


class Coloring(object):
    """
    Container for all information relevant to a coloring.

    Attributes
    ----------
    _shape : tuple of int (nrows, ncols)
        Tuple describing the shape of the sparsity matrix.
    _nzrows : ndarray of int
        Row indices of nonzero entries in the full jac sparsity matrix.
    _nzcols : ndarray of int
        Column indices of nonzero entries in the full jac sparsity matrix.
    _pct_nonzero : float
        If known, percentage of nonzero vs total array entries.
    _fwd : tuple (col_lists, row_maps) or None
        Contains lists of grouped columns and nonzero rows for each column for forward coloring.
    _rev : tuple (col_lists, row_maps) or None
        Contains lists of grouped columns and nonzero rows for each column for reverse coloring.
    _col_vars : list of str or None
        Names of variables corresponding to columns.
    _col_var_sizes : ndarray or None
        Sizes of column variables.
    _row_vars : list of str or None
        Names of variables corresponding to rows.
    _row_var_sizes : ndarray or None
        Sizes of row variables.
    _meta : dict
        Dictionary of metadata used to create the coloring.
    _names_array : ndarray or None:
        Names of total jacobian rows or columns.
    _local_array : ndarray or None:
        Indices of total jacobian rows or columns.
    """

    def __init__(self, sparsity, row_vars=None, row_var_sizes=None, col_vars=None,
                 col_var_sizes=None):
        """
        Initialize data structures.

        Parameters
        ----------
        sparsity : ndarray
            Full jacobian sparsity matrix (dense bool form).
        row_vars : list of str or None
            Names of variables corresponding to rows.
        row_var_sizes : ndarray or None
            Sizes of row variables.
        col_vars : list of str or None
            Names of variables corresponding to columns.
        col_var_sizes : ndarray or None
            Sizes of column variables.
        """
        # store the nonzero row and column indices if jac sparsity is provided
        self._nzrows, self._nzcols = np.nonzero(sparsity)
        self._shape = sparsity.shape
        self._pct_nonzero = np.count_nonzero(sparsity) / (self._shape[0] * self._shape[1]) * 100

        self._row_vars = row_vars
        self._row_var_sizes = row_var_sizes
        self._col_vars = col_vars
        self._col_var_sizes = col_var_sizes

        self._fwd = None
        self._rev = None
        self._meta = {}

        self._names_array = None
        self._local_array = None

    def color_iter(self, direction):
        """
        Given a direction, yield an iterator over column (or row) groups.

        Parameters
        ----------
        direction : str
            Derivative direction ('fwd' or 'rev').

        Yields
        ------
        list of int
            Lists of column indices (in fwd mode) or row indices (in rev mode).
        """
        if direction == 'fwd':
            colors = self._fwd[0]
        elif direction == 'rev':
            colors = self._rev[0]
        else:
            raise RuntimeError("Invalid direction '%s' in color_iter" % direction)

        for i in colors[0]:
            yield [i]

        for i in range(1, len(colors)):
            yield colors[i]

    def color_nonzero_iter(self, direction):
        """
        Given a direction, yield an iterator over (columns, nz_rows) or (rows, nz_columns).

        Parameters
        ----------
        direction : str
            Indicates which coloring subdict ('fwd' or 'rev') to use.

        Yields
        ------
        (column or row groups, nonzero row or column lists)
            Yields a list of columns/rows and their associated nonzero rows/columns for each
            color.
        """
        nz_rows = self.get_row_col_map(direction)
        for col_chunk in self.color_iter(direction):
            yield col_chunk, [nz_rows[c] for c in col_chunk]

    def get_row_col_map(self, direction):
        """
        Return mapping of nonzero rows to each column (fwd) or nonzeros columns to each row (rev).

        Parameters
        ----------
        direction : str
            Indicator of forward mode ('fwd') or reverse mode ('rev').

        Returns
        -------
        list of lists of int
            List where each entry contains list of nonzero rows/cols for the index corresponding
            to each column/row.
        """
        if direction == 'fwd':
            return self._fwd[1]
        elif direction == 'rev':
            return self._rev[1]
        else:
            raise RuntimeError("Invalid direction '%s' in get_row_col_map" % direction)

    def modes(self):
        """
        Return a tuple containing the modes included in this coloring.

        Returns
        -------
        tuple
            Tuple containing some subset of ('fwd', 'rev').
        """
        if self._fwd and self._rev:
            return ('fwd', 'rev')
        elif self._fwd:
            return ('fwd',)
        elif self._rev:
            return ('rev',)
        return ()

    def _solves_info(self):
        """
        Return info about the number of colors given the current coloring scheme.

        Returns
        -------
        float
            Total size (minimum chosen based on which mode is better).
        float
            Total solves.
        float
            Number of forward solves.
        float
            Number of reverse solves.
        float
            Percent improvment.
        """
        rev_size = self._shape[0]  # nrows
        fwd_size = self._shape[1]  # ncols

        tot_solves = self.total_solves()

        fwd_solves = rev_solves = 0
        if tot_solves == 0:  # no coloring found
            tot_solves = tot_size = min([rev_size, fwd_size])
            pct = 0.
        else:
            fwd_lists = self._fwd[0] if self._fwd else []
            rev_lists = self._rev[0] if self._rev else []

            if self._meta.get('bidirectional'):
                tot_size = min(fwd_size, rev_size)
            elif fwd_lists and not rev_lists:
                tot_size = fwd_size
            elif rev_lists and not fwd_lists:
                tot_size = rev_size
            else:
                tot_size = min(fwd_size, rev_size)

            if fwd_lists:
                fwd_solves = len(fwd_lists[0]) + len(fwd_lists) - 1

            if rev_lists:
                rev_solves = len(rev_lists[0]) + len(rev_lists) - 1

            if tot_size <= 0:
                pct = 0.
            else:
                pct = ((tot_size - tot_solves) / tot_size * 100)

        if tot_size < 0:
            tot_size = '?'

        return tot_size, tot_solves, fwd_solves, rev_solves, pct

    def total_solves(self, do_fwd=True, do_rev=True):
        """
        Return total number of solves required based on the given coloring info.

        Parameters
        ----------
        do_fwd : bool
            If True, add fwd colors to total.
        do_rev : bool
            If True, add rev colors to total.

        Returns
        -------
        int
            Total number of solves required to compute the jacobian.
        """
        total = 0

        # lists[0] are the uncolored columns or rows, which are solved individually so
        # we add all of them, along with the number of remaining lists, where each
        # sublist is a bunch of columns or rows that are solved together, to get the total colors
        # (which equals the total number of linear solves).
        if do_fwd and self._fwd:
            row_lists, _ = self._fwd
            total += len(row_lists[0]) + len(row_lists) - 1
        if do_rev and self._rev:
            col_lists, _ = self._rev
            total += len(col_lists[0]) + len(col_lists) - 1

        return total

    @staticmethod
    def load(fname):
        """
        Read the coloring object from the given pickle file.

        Parameters
        ----------
        fname : str
            Name of file to read from.

        Returns
        -------
        Coloring
            See docstring for Coloring class.
        """
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def save(self, fname):
        """
        Write the coloring object to the given stream, creating intermediate dirs if needed.

        Parameters
        ----------
        fname : str
            File to save to.
        """
        if isinstance(fname, str):
            color_dir = os.path.dirname(os.path.abspath(fname))
            if not os.path.exists(color_dir):
                os.makedirs(color_dir)
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError("Can't save coloring.  Expected a string for fname but got a %s" %
                            type(fname).__name__)

    def _check_config_total(self, driver):
        """
        Check the config of this total Coloring vs. the existing driver config.

        Parameters
        ----------
        driver : Driver
            Current driver object.
        """
        of_names, of_sizes = _get_response_info(driver)
        wrt_names, wrt_sizes = _get_desvar_info(driver)

        self._config_check_msgs(of_names, of_sizes, wrt_names, wrt_sizes, driver)

    def _check_config_partial(self, system):
        """
        Check the config of this partial (or semi-total) Coloring vs. the existing model config.

        Parameters
        ----------
        system : System
            System being colored.
        """
        # check the contents (vars and sizes) of the input and output vectors of system
        info = {'coloring': None, 'wrt_patterns': self._meta['wrt_patterns']}
        system._update_wrt_matches(info)
        if system.pathname:
            wrt_matches = set(['.'.join((system.pathname, n))
                              for n in info['wrt_matches_prom']])
            # for partial and semi-total derivs, convert to promoted names
            ordered_of_info = system._jac_var_info_abs2prom(system._jacobian_of_iter())
            ordered_wrt_info = \
                system._jac_var_info_abs2prom(system._jacobian_wrt_iter(wrt_matches))
        else:
            ordered_of_info = list(system._jacobian_of_iter())
            ordered_wrt_info = list(system._jacobian_wrt_iter(info['wrt_matches']))

        of_names = [t[0] for t in ordered_of_info]
        wrt_names = [t[0] for t in ordered_wrt_info]

        of_sizes = [t[2] - t[1] for t in ordered_of_info]
        wrt_sizes = [t[2] - t[1] for t in ordered_wrt_info]

        self._config_check_msgs(of_names, of_sizes, wrt_names, wrt_sizes, system)

    def _config_check_msgs(self, of_names, of_sizes, wrt_names, wrt_sizes, obj):
        msg_suffix = ("Make sure you don't have different problems that have the same coloring "
                      "directory. Set the coloring directory by setting the value of "
                      "problem.options['coloring_dir'].")

        msg = ["%s: Current coloring configuration does not match the "
               "configuration of the current model." % obj.msginfo]

        if of_names != self._row_vars:
            of_diff = set(of_names) - set(self._row_vars)
            if of_diff:
                msg.append('   The following row vars were added: %s.' % sorted(of_diff))
            else:
                of_diff = set(self._row_vars) - set(of_names)
                if of_diff:
                    msg.append('   The following row vars were removed: %s.' % sorted(of_diff))
                else:
                    msg.append('   The row vars have changed order.')

        if wrt_names != self._col_vars:
            wrt_diff = set(wrt_names) - set(self._col_vars)
            if wrt_diff:
                msg.append('   The following column vars were added: %s.' % sorted(wrt_diff))
            else:
                wrt_diff = set(self._col_vars) - set(wrt_names)
                if wrt_diff:
                    msg.append('   The following column vars were removed: %s.' % sorted(wrt_diff))
                else:
                    msg.append('   The column vars have changed order.')

        # check sizes
        changed_sizes = []
        if of_names == self._row_vars:
            for i, (my_sz, sz) in enumerate(zip(self._row_var_sizes, of_sizes)):
                if my_sz != sz:
                    changed_sizes.append(of_names[i])

        if wrt_names == self._col_vars:
            for i, (my_sz, sz) in enumerate(zip(self._col_var_sizes, wrt_sizes)):
                if my_sz != sz:
                    changed_sizes.append(wrt_names[i])

        if changed_sizes:
            msg.append('   The following variables have changed sizes: %s.' % sorted(changed_sizes))

        if len(msg) > 1:
            msg.append(msg_suffix)
            raise RuntimeError('\n'.join(msg))

    def __repr__(self):
        """
        Return a short summary representation of this coloring.

        Returns
        -------
        str
            Brief summary.
        """
        shape = self._shape

        if self._fwd and self._rev:
            direction = 'bidirectional'
        elif self._fwd:
            direction = 'fwd'
        else:
            direction = 'rev'

        return (
            f"Coloring (direction: {direction}, ncolors: {self.total_solves()}, shape: {shape}"
            f", pct nonzero: {self._pct_nonzero:.2f}, tol: {self._meta.get('good_tol')}"
        )

    def summary(self):
        """
        Print a summary of this coloring.

        Parameters
        ----------
        stream : file-like
            Where the output will go.
        """
        nrows = self._shape[0] if self._shape else -1
        ncols = self._shape[1] if self._shape else -1

        print("\nJacobian shape: (%d, %d)  (%5.2f%% nonzero)" % (nrows, ncols, self._pct_nonzero))
        if self._fwd is None and self._rev is None:
            tot_size = min(nrows, ncols)
            if tot_size < 0:
                tot_size = '?'
            print("Simultaneous derivatives can't improve on the total number of solves "
                  "required (%s) for this configuration" % tot_size)
        else:
            tot_size, tot_colors, fwd_solves, rev_solves, pct = self._solves_info()

            print("FWD solves: %d   REV solves: %d" % (fwd_solves, rev_solves))
            print("Total colors vs. total size: %d vs %s  (%.1f%% improvement)" %
                  (tot_colors, tot_size, pct))

        meta = self._meta
        print()
        good_tol = meta.get('good_tol')
        if good_tol is not None:
            print("Sparsity computed using tolerance: %g" % meta['good_tol'])
            if meta['n_tested'] > 1:
                print("Most common number of nonzero entries (%d of %d) repeated %d times out "
                      "of %d tolerances tested.\n" % (meta['J_size'] - meta['zero_entries'],
                                                      meta['J_size'],
                                                      meta['nz_matches'], meta['n_tested']))

        sparsity_time = meta.get('sparsity_time')
        if sparsity_time is not None:
            print("Time to compute sparsity: %f sec." % sparsity_time)

        coloring_time = meta.get('coloring_time')
        if coloring_time is not None:
            print("Time to compute coloring: %f sec." % coloring_time)

    def display_txt(self):
        """
        Print the structure of a boolean array with coloring info for each nonzero value.

        Forward mode colored nonzeros are denoted by 'f', reverse mode nonzeros by 'r',
        overlapping nonzeros by 'O' and uncolored nonzeros by 'x'.  Zeros are denoted by '.'.
        Note that x's and O's should never appear unless there is a bug in the coloring
        algorithm.

        If names and sizes of row and column vars are known, print the name of the row var
        alongside each row and print the names of the column vars, aligned with each column,
        at the bottom.
        """
        shape = self._shape
        nrows, ncols = shape

        # array of chars the same size as dense jacobian
        charr = np.full(shape, '.', dtype=str)

        # mark all nonzero entries as 'x' initially, so the 'x' will be left
        # if not covered with an 'f' or an 'r'
        charr[self._nzrows, self._nzcols] = 'x'

        if self._fwd:
            full_rows = np.arange(nrows, dtype=int)
            col2row = self._fwd[1]
            for grp in self._fwd[0]:
                for c in grp:
                    rows = col2row[c]
                    if rows is None:
                        rows = full_rows
                    charr[rows, c] = 'f'

        has_overlap = False
        if self._rev:
            full_cols = np.arange(ncols, dtype=int)
            row2col = self._rev[1]
            for grp in self._rev[0]:
                for r in grp:
                    cols = row2col[r]
                    if cols is None:
                        cols = full_cols
                    for c in cols:
                        # check for any overlapping entries (ones having both a fwd and rev color)
                        # NOTE: this should never happen unless there's a bug in the coloring!
                        if charr[r, c] == 'f':
                            charr[r, c] = 'O'  # mark entry as overlapping
                            has_overlap = True
                        else:
                            charr[r, c] = 'r'

        if (self._row_vars is None or self._row_var_sizes is None or self._col_vars is None or
                self._col_var_sizes is None):
            # we don't have var name/size info, so just show the unadorned matrix
            for r in range(nrows):
                for c in range(ncols):
                    print(charr[r, c], end='')
                print(' %d' % r)
        else:
            # we have var name/size info, so mark rows/cols with their respective variable names
            rowstart = rowend = 0
            for rv, rvsize in zip(self._row_vars, self._row_var_sizes):
                rowend += rvsize
                for r in range(rowstart, rowend):
                    colstart = colend = 0
                    for _, cvsize in zip(self._col_vars, self._col_var_sizes):
                        colend += cvsize
                        for c in range(colstart, colend):
                            print(charr[r, c], end='')
                        colstart = colend
                    print(' %d  %s' % (r, rv))  # include row variable with each row
                rowstart = rowend

            # now print the column vars below the matrix, with each one spaced over to line up
            # with the appropriate starting column of the matrix ('|' marks the start of each var)
            start = 0
            for name, size in zip(self._col_vars, self._col_var_sizes):
                tab = ' ' * start
                print('%s|%s' % (tab, name))
                start += size

        if has_overlap:
            raise RuntimeError("Internal coloring bug: jacobian has entries where fwd and rev "
                               "colorings overlap!")

    def display(self):
        """
        Display a plot of the sparsity pattern, showing grouping by color.
        """
        try:
            from matplotlib import pyplot, axes, cm
        except ImportError:
            print("matplotlib is not installed so the coloring viewer is not available. The ascii "
                  "based coloring viewer can be accessed by calling display_txt() on the Coloring "
                  "object or by using 'openmdao view_coloring --textview <your_coloring_file>' "
                  "from the command line.")
            return

        nrows, ncols = self._shape
        aspect_ratio = ncols / nrows
        J = np.ones((nrows, ncols, 3), dtype=float)

        tot_size, tot_colors, fwd_solves, rev_solves, pct = self._solves_info()

        size = 10
        if nrows > ncols:
            mult = nrows / size
            ysize = nrows / mult
            xsize = ysize * aspect_ratio
        else:
            mult = ncols / size
            xsize = ncols / mult
            ysize = xsize / aspect_ratio

        xsize = max(1, int(xsize))
        ysize = max(1, int(ysize))

        fig = pyplot.figure(figsize=(xsize, ysize))  # in inches
        ax = pyplot.gca()

        # hide tic marks/labels
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

        if self._row_vars is not None and self._col_vars is not None:
            # map row/col to corresponding var names
            entry_xnames = np.zeros(ncols, dtype=int)
            entry_ynames = np.zeros(nrows, dtype=int)
            entry_xcolors = np.zeros(ncols, dtype=int)
            entry_ycolors = np.zeros(nrows, dtype=int)

            # pick two colors for our checkerboard pattern
            sjcolors = [cm.get_cmap('Greys')(0.3), cm.get_cmap('Greys')(0.4)]

            colstart = colend = 0
            for i, cvsize in enumerate(self._col_var_sizes):
                colend += cvsize
                entry_xnames[colstart:colend] = i
                colstart = colend

            # we have var name/size info, so mark rows/cols with their respective variable names
            rowstart = rowend = 0
            for ridx, rvsize in enumerate(self._row_var_sizes):
                rowend += rvsize
                entry_ynames[rowstart:rowend] = ridx

                colstart = colend = 0
                for cidx, cvsize in enumerate(self._col_var_sizes):
                    colend += cvsize
                    # display grid that breaks up the Jacobian into subjacs by variable pairs.
                    # using (ridx+cidx)%2 will give us a nice checkerboard pattern
                    J[rowstart:rowend, colstart:colend] = sjcolors[(ridx + cidx) % 2][:3]
                    colstart = colend

                rowstart = rowend

            def on_press(event):
                if event.inaxes == ax:
                    ix = int(event.xdata)
                    iy = int(event.ydata)

                    if event.xdata - ix >= .5:
                        ix += 1
                    ix = max(0, ix)
                    ix = min(ncols, ix)

                    if event.ydata - iy >= .5:
                        iy += 1
                    iy = max(0, iy)
                    iy = min(nrows, iy)

                    # if J[iy, ix] is not one of the background 'checkerboard' colors, then it
                    # must be either forward or reverse colored.
                    if np.all(J[iy, ix] == sjcolors[0][:3]) or np.all(J[iy, ix] == sjcolors[1][:3]):
                        color_str = ''
                    else:
                        # display the color number because sometimes certain colormap colors look
                        # too similar to the eye.
                        if entry_xcolors[ix] != 0:
                            color_str = 'Color: %d (fwd)' % entry_xcolors[ix]
                        else:
                            color_str = 'Color: %d (rev)' % entry_ycolors[iy]

                    # because we have potentially really long pathnames, we just print
                    # the 'of' and 'wrt' variables to the console instead of trying to display
                    # them on the plot.
                    print('\nJ[%d, %d]  %s' % (iy, ix, color_str),
                          '\nOF:', self._row_vars[entry_ynames[iy]],
                          '\nWRT:', self._col_vars[entry_xnames[ix]])

            def on_resize(event):
                fig.tight_layout()

            # set up event handling
            fig.canvas.mpl_connect('button_press_event', on_press)
            fig.canvas.mpl_connect('resize_event', on_resize)

        color_arrays = []
        if self._fwd:
            # winter is a blue/green color map
            cmap = cm.get_cmap('winter')

            icol = 1
            full_rows = np.arange(nrows, dtype=int)
            col2row = self._fwd[1]
            for i, grp in enumerate(self._fwd[0]):
                for c in grp:
                    rows = col2row[c]
                    if rows is None:
                        rows = full_rows
                    idx = icol / fwd_solves
                    for r in rows:
                        J[r, c][:] = cmap(idx)[:3]
                    if i == 0:  # group 0 are uncolored (each col has different color)
                        icol += 1
                    entry_xcolors[c] = icol
                icol += 1

        if self._rev:
            # autumn_r is a red/yellow color map
            cmap = cm.get_cmap('autumn_r')

            icol = 1
            full_cols = np.arange(ncols, dtype=int)
            row2col = self._rev[1]
            for i, grp in enumerate(self._rev[0]):
                for r in grp:
                    cols = row2col[r]
                    if cols is None:
                        cols = full_cols
                    idx = icol / rev_solves
                    for c in cols:
                        J[r, c][:] = cmap(idx)[:3]
                    if i == 0:  # group 0 are uncolored (each col has different color)
                        icol += 1
                    entry_ycolors[r] = icol
                icol += 1

        typ = self._meta['type'].upper()[0] + self._meta['type'][1:]

        ax.set_title("%s Jacobian Coloring (%d x %d)\n%d fwd colors, %d rev colors "
                     "(%.1f%% improvement)" %
                     (typ, self._shape[0], self._shape[1], fwd_solves, rev_solves, pct))

        pyplot.imshow(J, interpolation="none")
        fig.tight_layout()

        pyplot.show()

    def get_dense_sparsity(self):
        """
        Return a dense bool array representing the full sparsity.

        Returns
        -------
        ndarray
            Dense sparsity matrix.
        """
        J = np.zeros(self._shape, dtype=bool)
        J[self._nzrows, self._nzcols] = True
        return J

    def get_subjac_sparsity(self):
        """
        Compute the sparsity structure of each subjacobian based on the full jac sparsity.

        If row/col variables and sizes are not known, returns None.

        Returns
        -------
        dict or None
            Mapping of (of, wrt) keys to their corresponding (nzrows, nzcols, shape).
        """
        if self._row_vars and self._col_vars and self._row_var_sizes and self._col_var_sizes:
            J = self.get_dense_sparsity()
            return _jac2subjac_sparsity(J, self._row_vars, self._col_vars,
                                        self._row_var_sizes, self._col_var_sizes)

    def _subjac_sparsity_iter(self):
        subjac_sparsity = self.get_subjac_sparsity()

        if subjac_sparsity is None:
            raise RuntimeError("Coloring doesn't have enough info to compute subjac sparsity.")

        ostart = oend = 0
        for of, sub in subjac_sparsity.items():
            istart = iend = 0
            for i, (wrt, tup) in enumerate(sub.items()):
                nzrows, nzcols, shape = tup
                iend += shape[1]
                if i == 0:
                    oend += shape[0]
                if nzrows.size > 0:
                    yield (of, wrt, list(nzrows), list(nzcols), ostart, oend, istart, iend)
                istart = iend
            ostart = oend

    def get_declare_partials_calls(self):
        """
        Return a string containing declare_partials() calls based on the subjac sparsity.

        Returns
        -------
        str
            A string containing a declare_partials() call for each nonzero subjac. This
            string may be cut and pasted into a component's setup() method.
        """
        lines = []
        for of, wrt, nzrows, nzcols, _, _, _, _ in self._subjac_sparsity_iter():
            lines.append("    self.declare_partials(of='%s', wrt='%s', rows=%s, cols=%s)" %
                         (of, wrt, nzrows, nzcols))
        return '\n'.join(lines)

    def get_row_var_coloring(self, varname):
        """
        Return the number of fwd and rev solves needed for a particular row variable.

        Parameters
        ----------
        varname : str
            Name of the row variable.

        Returns
        -------
        int
            Number of forward solves needed for the given variable.
        int
            Number of reverse solves needed for the given variable.
        """
        fwd_solves = 0
        rev_solves = 0
        if self._row_vars and self._col_vars and self._row_var_sizes and self._col_var_sizes:
            row_slice = col_slice = slice(None)
            start = end = 0
            for name, size in zip(self._row_vars, self._row_var_sizes):
                end += size
                if name == varname:
                    row_slice = slice(start, end)
                    break
                start = end
            else:
                raise RuntimeError("Can't find variable '%s' in coloring." % varname)

            J = np.zeros(self._shape, dtype=bool)
            subJ = J[row_slice, col_slice]

            if self._fwd:
                uncolored = [[c] for c in self._fwd[0][0]]

                colored = self._fwd[0][1:]
                nzrows = self._fwd[1]
                for color_group in chain(uncolored, colored):
                    subJ[:, :] = False
                    # if any color in the group has nonzeros in our variable, add a solve
                    for c in color_group:
                        J[nzrows[c], c] = True

                    if np.any(subJ):
                        fwd_solves += 1

            if self._rev:
                uncolored = [[r] for r in self._rev[0][0]]
                colored = self._rev[0][1:]
                for color_group in chain(uncolored, colored):
                    subJ[:, :] = False
                    J[color_group, :] = True

                    if np.any(subJ):
                        rev_solves += 1

        return fwd_solves, rev_solves

    def _local_indices(self, inds, mode):

        if self._names_array is None and self._local_array is None:
            col_names = self._col_vars
            col_sizes = self._col_var_sizes
            row_names = self._row_vars
            row_sizes = self._row_var_sizes

            if mode == 'fwd':
                col_info = zip(col_names, col_sizes)
            else:
                col_info = zip(row_names, row_sizes)

            names = []
            indices = []
            for i, j in col_info:
                names.append(np.repeat(i, j))
                indices.append(np.arange(j))

            self._names_array = np.concatenate(names)
            self._local_array = np.concatenate(indices)

        if isinstance(inds, list):
            var_name_and_sub_indices = [(key, [x[1] for x in group]) for key, group in groupby(
                zip(self._names_array[inds], self._local_array[inds]), key=lambda x: x[0])]
        else:
            var_name_and_sub_indices = [(self._names_array[inds], self._local_array[inds])]

        return var_name_and_sub_indices


def _order_by_ID(col_matrix):
    """
    Return columns in order of incidence degree (ID).

    ID is the number of already colored neighbors (neighbors are dependent columns).

    Parameters
    ----------
    col_matrix : ndarray
        Boolean array of column dependencies.

    Yields
    ------
    int
        Column index.
    """
    degrees = _count_nonzeros(col_matrix, axis=0)
    ncols = degrees.size

    if ncols == 0:
        return

    # use max degree column as a starting point instead of just choosing a random column
    # since all have incidence degree of 0 when we start.
    start = degrees.argmax()
    yield start

    colored_degrees = np.zeros(degrees.size, dtype=get_index_dtype(maxval=degrees[start]))
    colored_degrees[col_matrix[start]] += 1
    colored_degrees[start] = -ncols  # ensure that this col will never have max degree again

    for i in range(ncols - 1):
        col = colored_degrees.argmax()
        colored_degrees[col_matrix[col]] += 1
        colored_degrees[col] = -ncols  # ensure that this col will never have max degree again
        yield col


def _J2col_matrix(J):
    """
    Convert boolean jacobian sparsity matrix to a column adjacency matrix.

    Parameters
    ----------
    J : ndarray
        Boolean jacobian sparsity matrix.

    Returns
    -------
    ndarray
        Column adjacency matrix.
    """
    nrows, ncols = J.shape

    col_matrix = np.zeros((ncols, ncols), dtype=bool)

    # mark col_matrix entries as True when nonzero row entries make them dependent
    for row in range(nrows):
        nzro = np.nonzero(J[row])[0]
        for col in nzro:
            col_matrix[col, nzro] = True

    # zero out diagonal (column is not adjacent to itself)
    np.fill_diagonal(col_matrix, False)

    return col_matrix


def _Jc2col_matrix_direct(J, Jc):
    """
    Convert a partitioned jacobian sparsity matrix to a column adjacency matrix.

    This creates the column adjacency matrix used for direct jacobian determination
    as described in Coleman, T.F., Verma, A. (1998) The efficient Computation of Sparse Jacobian
    Matrices Using Automatic Differentiation. SIAM Journal on Scientific Computing, 19(4),
    1210-1233.

    Parameters
    ----------
    J : ndarray
        Boolean jacobian sparsity matrix.
    Jc : ndarray
        Boolean sparsity matrix of a partition of J.

    Returns
    -------
    ndarray
        Column adjacency matrix.
    """
    assert J.shape == Jc.shape

    nrows, ncols = J.shape

    col_matrix = np.zeros((ncols, ncols), dtype=bool)

    col_keep = _count_nonzeros(Jc, axis=0) > 0

    # mark col_matrix[col1, col2] as True when Jc[row, col1] is True OR Jc[row, col2] is True
    for row in range(nrows):
        nzro = np.nonzero(J[row] & col_keep)[0]
        for col1, col2 in combinations(nzro, 2):
            if Jc[row, col1] or Jc[row, col2]:
                col_matrix[col1, col2] = True
                col_matrix[col2, col1] = True

    # zero out diagonal (column is not adjacent to itself)
    np.fill_diagonal(col_matrix, False)

    return col_matrix


def _get_full_disjoint_cols(J):
    """
    Find sets of disjoint columns in J and their corresponding rows using a col adjacency matrix.

    Parameters
    ----------
    J : ndarray
        The total jacobian.

    Returns
    -------
    list
        List of lists of disjoint columns
    """
    return _get_full_disjoint_col_matrix_cols(_J2col_matrix(J))


def _get_full_disjoint_col_matrix_cols(col_matrix):
    """
    Find sets of disjoint columns in a column intersection matrix.

    Parameters
    ----------
    col_matrix : ndarray
        Column intersection matrix

    Returns
    -------
    list
        List of lists of disjoint columns
    """
    color_groups = []
    _, ncols = col_matrix.shape

    # -1 indicates that a column has not been colored
    colors = np.full(ncols, -1, dtype=get_index_dtype(maxval=ncols))

    for col in _order_by_ID(col_matrix):
        neighbor_colors = set(colors[col_matrix[col]])
        for color, grp in enumerate(color_groups):
            if color not in neighbor_colors:
                grp.append(col)
                colors[col] = color
                break
        else:
            colors[col] = len(color_groups)
            color_groups.append([col])

    return color_groups


def _color_partition(J, Jpart):
    """
    Compute a single directional fwd coloring using partition Jpart.

    This routine is used to compute a fwd coloring on Jc and a rev coloring on Jr.T.

    Parameters
    ----------
    J : ndarray
        Dense jacobian sparsity matrix
    Jpart : ndarray
        Partition of the jacobian sparsity matrix.

    Returns
    -------
    list
        List of color groups.  First group is uncolored.
    list
        List of nonzero rows for each column.
    """
    ncols = Jpart.shape[1]
    col_nonzeros = _count_nonzeros(Jpart, axis=0)
    row_nonzeros = _count_nonzeros(Jpart, axis=1)
    col_keep = col_nonzeros > 0
    row_keep = row_nonzeros > 0

    # use this to map indices back to the full J indices.
    idxmap = np.arange(ncols, dtype=int)[col_keep]

    intersection_mat = _Jc2col_matrix_direct(J, Jpart)
    intersection_mat = intersection_mat[col_keep]
    intersection_mat = intersection_mat[:, col_keep]

    col_groups = _get_full_disjoint_col_matrix_cols(intersection_mat)

    for i, group in enumerate(col_groups):
        col_groups[i] = sorted([idxmap[c] for c in group if col_keep[idxmap[c]]])
    col_groups = _split_groups(col_groups)

    col2row = [None] * ncols
    for col in idxmap:
        col2row[col] = [r for r in np.nonzero(Jpart[:, col])[0] if row_keep[r]]

    return [col_groups, col2row]


def MNCO_bidir(J):
    """
    Compute bidirectional coloring using Minimum Nonzero Count Order (MNCO).

    Based on the algorithm found in Coleman, T.F., Verma, A. (1998) The efficient Computation
    of Sparse Jacobian Matrices Using Automatic Differentiation. SIAM Journal on Scientific
    Computing, 19(4), 1210-1233.

    Parameters
    ----------
    J : ndarray
        Dense Jacobian sparsity matrix (boolean)

    Returns
    -------
    Coloring
        See docstring for Coloring class.
    """
    start_time = time.time()

    nrows, ncols = J.shape

    coloring = Coloring(sparsity=J)

    M_col_nonzeros = _count_nonzeros(J, axis=0)
    M_row_nonzeros = _count_nonzeros(J, axis=1)

    M_rows, M_cols = coloring._nzrows, coloring._nzcols

    Jc_rows = [None] * nrows
    Jr_cols = [None] * ncols

    row_i = col_i = 0

    # partition J into Jc and Jr
    # Jc is colored by column and those columns will be solved in fwd mode
    # Jr is colored by row and those rows will be solved in reverse mode
    # We build Jc from bottom up (by row) and Jr from right to left (by column).

    # get index of row with fewest nonzeros and col with fewest nonzeros
    r = M_row_nonzeros.argmin()
    c = M_col_nonzeros.argmin()

    nnz_r = M_row_nonzeros[r]
    nnz_c = M_col_nonzeros[c]

    Jc_nz_max = 0   # max row nonzeros in Jc
    Jr_nz_max = 0   # max col nonzeros in Jr

    while M_rows.size + M_cols.size > 0:
        # what the algorithm is doing is basically minimizing the total of the max number of nonzero
        # columns in Jc + the max number of nonzero rows in Jr, so it's basically minimizing
        # the upper bound of the number of colors that will be needed.

        # we differ from the algorithm in the paper here slightly because we add ncols and nrows to
        # different sides of the inequality in order to prevent bad colorings when we have
        # matrices that have many more rows than columns or many more columns than rows.
        if ncols + Jr_nz_max + max(Jc_nz_max, nnz_r) < (nrows + Jc_nz_max + max(Jr_nz_max, nnz_c)):
            Jc_rows[r] = M_cols[M_rows == r]
            Jc_nz_max = max(nnz_r, Jc_nz_max)

            M_row_nonzeros[r] = ncols + 1  # make sure we don't pick this one again
            M_col_nonzeros[Jc_rows[r]] -= 1

            keep = M_rows != r
            r = M_row_nonzeros.argmin()
            c = M_col_nonzeros.argmin()
            nnz_r = M_row_nonzeros[r]

            row_i += 1
        else:
            Jr_cols[c] = M_rows[M_cols == c]
            Jr_nz_max = max(nnz_c, Jr_nz_max)

            M_col_nonzeros[c] = nrows + 1  # make sure we don't pick this one again
            M_row_nonzeros[Jr_cols[c]] -= 1

            keep = M_cols != c
            r = M_row_nonzeros.argmin()
            c = M_col_nonzeros.argmin()
            nnz_c = M_col_nonzeros[c]

            col_i += 1

        M_rows = M_rows[keep]
        M_cols = M_cols[keep]

    nnz_Jc = nnz_Jr = 0
    jac = np.zeros(J.shape, dtype=bool)

    if row_i > 0:
        Jc = jac
        # build Jc and do fwd coloring on it
        for i, cols in enumerate(Jc_rows):
            if cols is not None:
                Jc[i][cols] = True
                nnz_Jc += len(cols)

        coloring._fwd = _color_partition(J, Jc)
        jac[:] = False  # reset for use with Jr

    if col_i > 0:
        Jr = jac
        # build Jr and do rev coloring
        for i, rows in enumerate(Jr_cols):
            if rows is not None:
                Jr[rows, i] = True
                nnz_Jr += len(rows)

        coloring._rev = _color_partition(J.T, Jr.T)

    if np.count_nonzero(J) != nnz_Jc + nnz_Jr:
        raise RuntimeError("Nonzero mismatch for J vs. Jc and Jr")

    # check_coloring(J, coloring)

    coloring._meta['coloring_time'] = time.time() - start_time
    coloring._meta['bidirectional'] = True

    return coloring


def _tol_sweep(arr, tol=_DEF_COMP_SPARSITY_ARGS['tol'], orders=_DEF_COMP_SPARSITY_ARGS['orders']):
    """
    Find best tolerance 'around' tol to choose nonzero values of arr.

    Sweeps over tolerances +- 'orders' orders of magnitude around tol and picks the most
    stable one (one corresponding to the most repeated number of nonzero entries).

    The array 'arr' must not contain negative numbers.

    Parameters
    ----------
    arr : ndarray
        The array requiring computation of nonzero values.
    tol : float
        Tolerance.  We'll sweep above and below this by 'orders' of magnitude.
    orders : int
        Number of orders of magnitude for one direction of our sweep.

    Returns
    -------
    dict
        Info about the tolerance and how it was determined.
    """
    if orders is None:   # skip the sweep. Just use the tolerance given.
        good_tol = tol
        nz_matches = n_tested = 1
    else:
        nzeros = []
        itol = tol * 10.**orders
        smallest = tol / 10.**orders
        n_tested = 0
        while itol >= smallest:
            if itol < 1.:
                rows, cols = np.nonzero(arr > itol)
                if nzeros and nzeros[-1][1] == len(rows):
                    nzeros[-1][0].append(itol)
                else:
                    nzeros.append(([itol], len(rows)))
                n_tested += 1
            itol *= .1

        # pick lowest tolerance corresponding to the most repeated number of 'zero' entries
        sorted_items = sorted(nzeros, key=lambda x: len(x[0]), reverse=True)
        nz_matches = len(sorted_items[0][0])

        if nz_matches <= 1:
            lst = []
            for itols, nz in sorted_items:
                entry = ", ".join(['%3.1g' % tol for tol in itols])
                if len(itols) > 1:
                    entry = "[%s]" % entry
                lst.append("(%s, %d)" % (entry, nz))

            raise RuntimeError("Could not find more than 1 tolerance to match any number of "
                               "nonzeros. This indicates that your tolerance sweep of +- %d "
                               "orders, starting from %s is not big enough.  To get a 'stable' "
                               "sparsity pattern, try re-running with a larger tolerance sweep.\n"
                               "Nonzeros found for each tolerance: [%s]" %
                               (orders, tol, ", ".join(lst)))

        good_tol = sorted_items[0][0][-1]

    info = {
        'tol': tol,
        'orders': orders,
        'good_tol': good_tol,
        'nz_matches': nz_matches,
        'n_tested': n_tested,
        'zero_entries': arr[arr <= good_tol].size,
        'J_size': arr.size,
    }

    return info


@contextmanager
def _compute_total_coloring_context(top):
    """
    Context manager for computing total jac sparsity for simultaneous coloring.

    Parameters
    ----------
    top : System
        Top of the system hierarchy where coloring will be done.
    """
    np.random.seed(41)  # set seed for consistency

    for system in top.system_iter(recurse=True, include_self=True):
        if system.matrix_free:
            raise RuntimeError("%s: simultaneous coloring does not currently work with matrix free "
                               "components." % system.pathname)

        jac = system._assembled_jac
        if jac is None:
            jac = system._jacobian
        if jac is not None:
            jac._randomize = True

    yield

    for system in top.system_iter(recurse=True, include_self=True):
        jac = system._assembled_jac
        if jac is None:
            jac = system._jacobian
        if jac is not None:
            jac._randomize = False


def _get_bool_total_jac(prob, num_full_jacs=_DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                        tol=_DEF_COMP_SPARSITY_ARGS['tol'],
                        orders=_DEF_COMP_SPARSITY_ARGS['orders'], setup=False, run_model=False,
                        of=None, wrt=None, use_abs_names=True):
    """
    Return a boolean version of the total jacobian.

    The jacobian is computed by calculating a total jacobian using _compute_totals 'num_full_jacs'
    times and adding the absolute values of those together, then dividing by 'num_full_jacs',
    then converting to a boolean array, specifying all entries below a tolerance as False and all
    others as True.  Prior to calling _compute_totals, all of the partial jacobians in the
    model are modified so that when any of their subjacobians are assigned a value, that
    value is populated with positive random numbers in the range [1.0, 2.0).

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    num_full_jacs : int
        Number of times to repeat total jacobian computation.
    tol : float
        Starting tolerance on values in jacobian.  Actual tolerance is computed based on
        consistent numbers of zero entries over a sweep of tolerances.  Anything smaller in
        magnitude than the computed tolerance will be set to 0.0.
    orders : int
        Number of orders of magnitude for up and down tolerance sweep (default is 5).
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.
    of : iter of str or None
        Names of response variables.
    wrt : iter of str or None
        Names of design variables.

    Returns
    -------
    ndarray
        A boolean composite of 'num_full_jacs' total jacobians.
    """
    # clear out any old simul coloring info
    driver = prob.driver
    driver._res_jacs = {}

    if setup:
        prob.setup(mode=prob._mode)

    if run_model:
        prob.run_model(reset_iter_counts=False)

    if of is None or wrt is None:
        desvars = prom2ivc_src_dict(driver._designvars)
        driver_wrt = list(desvars)
        driver_of = driver._get_ordered_nl_responses()
        if not driver_wrt or not driver_of:
            raise RuntimeError("When computing total jacobian sparsity, either 'of' and 'wrt' "
                               "must be provided or design_vars/constraints/objective must be "
                               "added to the driver.")
        wrt = driver_wrt
        of = driver_of
        use_driver = True
    else:
        use_driver = False

    with _compute_total_coloring_context(prob.model):
        start_time = time.time()
        fullJ = None
        for i in range(num_full_jacs):
            if use_driver:
                J = prob.driver._compute_totals(of=of, wrt=wrt, return_format='array',
                                                use_abs_names=use_abs_names)
            else:
                J = prob.compute_totals(of=of, wrt=wrt, return_format='array',
                                        use_abs_names=use_abs_names)
            if fullJ is None:
                fullJ = np.abs(J)
            else:
                fullJ += np.abs(J)
        elapsed = time.time() - start_time

    fullJ *= (1.0 / np.max(fullJ))

    info = _tol_sweep(fullJ, tol, orders)
    info['num_full_jacs'] = num_full_jacs
    info['sparsity_time'] = elapsed
    info['type'] = 'total'

    print("Full total jacobian was computed %d times, taking %f seconds." % (num_full_jacs,
                                                                             elapsed))
    print("Total jacobian shape:", fullJ.shape, "\n")

    boolJ = np.zeros(fullJ.shape, dtype=bool)
    boolJ[fullJ > info['good_tol']] = True

    return boolJ, info


def _jac2subjac_sparsity(J, ofs, wrts, of_sizes, wrt_sizes):
    """
    Given a boolean jacobian and variable names and sizes, compute subjac sparsity.

    Parameters
    ----------
    J : ndarray
        Boolean jacobian.
    ofs : list of str
        List of variables corresponding to rows.
    wrts : list of str
        List of variables corresponding to columns.
    of_sizes : ndarray of int
        Sizes of ofs variables.
    wrt_sizes : ndarray of int
        Sizes of wrts variables.

    Returns
    -------
    OrderedDict
        Nested OrderedDict of form sparsity[of][wrt] = (rows, cols, shape)
    """
    sparsity = OrderedDict()
    row_start = row_end = 0

    for of, of_size in zip(ofs, of_sizes):
        sparsity[of] = OrderedDict()
        row_end += of_size
        col_start = col_end = 0
        for wrt, wrt_size in zip(wrts, wrt_sizes):
            col_end += wrt_size

            # save sparsity structure as  (rows, cols, shape)
            irows, icols = np.nonzero(J[row_start:row_end, col_start:col_end])
            sparsity[of][wrt] = (irows, icols, (of_size, wrt_size))

            col_start = col_end

        row_start = row_end

    return sparsity


def _write_sparsity(sparsity, stream):
    """
    Write the sparsity structure to the given stream.

    Parameters
    ----------
    sparsity : dict
        Nested dict of subjac sparsity for each total derivative.
    stream : file-like
        Output stream.
    """
    stream.write("{\n")

    last_res_idx = len(sparsity) - 1
    for i, out in enumerate(sorted(sparsity)):
        out_dict = sparsity[out]
        stream.write('"%s": {\n' % out)
        last_dv_idx = len(out_dict) - 1
        for j, inp in enumerate(sorted(out_dict)):
            subjac = out_dict[inp]
            rows, cols, shape = subjac
            if len(rows) > 15:
                stream.write('   "%s": [\n' % inp)
                stream.write('        %s,\n' % rows)
                stream.write('        %s,\n' % cols)
                stream.write('        %s]' % list(shape))
            else:
                stream.write('   "%s": [%s, %s, %s]' % (inp, rows, cols, list(shape)))
            if j == last_dv_idx:
                stream.write('\n')
            else:
                stream.write(',\n')
        stream.write("}")
        if i == last_res_idx:
            stream.write('\n')
        else:
            stream.write(',\n')

    stream.write("}\n")


def _get_desvar_info(driver, names=None, use_abs_names=True):
    desvars = prom2ivc_src_dict(driver._designvars)

    if names is None:
        abs_names = list(desvars)
        return abs_names, [desvars[n]['size'] for n in abs_names]

    model = driver._problem().model
    abs2meta = model._var_allprocs_abs2meta

    if use_abs_names:
        abs_names = names
    else:
        prom2abs = model._var_allprocs_prom2abs_list['output']
        abs_names = [prom2abs[n][0] for n in names]

    # if a variable happens to be a design var, use that size
    sizes = []
    for n in abs_names:
        if n in desvars:
            sizes.append(desvars[n]['size'])
        else:
            sizes.append(abs2meta[n]['global_size'])

    return abs_names, sizes


def _get_response_info(driver, names=None, use_abs_names=True):
    responses = driver._responses
    if names is None:
        abs_names = driver._get_ordered_nl_responses()
        return abs_names, [responses[n]['size'] for n in abs_names]

    model = driver._problem().model
    abs2meta = model._var_allprocs_abs2meta

    if use_abs_names:
        abs_names = names
    else:
        prom2abs = model._var_allprocs_prom2abs_list['output']
        abs_names = [prom2abs[n][0] for n in names]

    # if a variable happens to be a response var, use that size
    sizes = []
    for n in abs_names:
        if n in responses:
            sizes.append(responses[n]['size'])
        else:
            sizes.append(abs2meta[n]['global_size'])

    return abs_names, sizes


def get_tot_jac_sparsity(problem, mode='fwd',
                         num_full_jacs=_DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                         tol=_DEF_COMP_SPARSITY_ARGS['tol'],
                         setup=False, run_model=False):
    """
    Compute derivative sparsity for the given problem.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    num_full_jacs : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance used to determine if an array entry is nonzero.
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.

    Returns
    -------
    dict
        A nested dict specifying subjac sparsity for each total deriv, e.g., sparsity[resp][dv].
    ndarray
        Boolean sparsity matrix.
    """
    driver = problem.driver

    J, _ = _get_bool_total_jac(problem, num_full_jacs=num_full_jacs, tol=tol, setup=setup,
                               run_model=run_model)

    ofs, of_sizes = _get_response_info(driver)
    wrts, wrt_sizes = _get_desvar_sizes(driver)

    sparsity = _jac2subjac_sparsity(J, ofs, wrts, of_sizes, wrt_sizes)

    driver._total_jac = None

    return sparsity, J


def _split_groups(groups):
    uncolored = [grp[0] for grp in groups if len(grp) == 1]
    groups = [grp for grp in groups if len(grp) > 1]

    # the first lists entry corresponds to all uncolored columns (columns that are not
    # disjoint wrt any other columns).  The other entries are groups of columns that do not
    # share any nonzero row entries in common.
    clists = [uncolored]
    clists.extend(groups)

    return clists


def _compute_coloring(J, mode):
    """
    Compute a good coloring in a specified dominant direction.

    Parameters
    ----------
    J : ndarray
        The boolean total jacobian.
    mode : str
        The direction for solving for total derivatives.  Must be 'fwd', 'rev' or 'auto'.
        If 'auto', use bidirectional coloring.

    Returns
    -------
    Coloring
        See Coloring class docstring.
    """
    start_time = time.time()
    nrows, ncols = J.shape

    if mode == 'auto':  # use bidirectional coloring
        coloring = MNCO_bidir(J)
        fwdcoloring = _compute_coloring(J, 'fwd')
        if coloring.total_solves() >= fwdcoloring.total_solves():
            coloring = fwdcoloring
            coloring._meta['fallback'] = True
        revcoloring = _compute_coloring(J, 'rev')
        if coloring.total_solves() > revcoloring.total_solves():
            coloring = revcoloring
            coloring._meta['fallback'] = True
        return coloring

    rev = mode == 'rev'

    coloring = Coloring(sparsity=J)

    if rev:
        J = J.T

    col_groups = _split_groups(_get_full_disjoint_cols(J))

    full_slice = slice(None)
    col2rows = [full_slice] * J.shape[1]  # will contain list of nonzero rows for each column
    for lst in col_groups:
        for col in lst:
            col2rows[col] = np.nonzero(J[:, col])[0]

    if rev:
        coloring._rev = (col_groups, col2rows)
    else:  # fwd
        coloring._fwd = (col_groups, col2rows)

    coloring._meta['coloring_time'] = time.time() - start_time

    return coloring


def compute_total_coloring(problem, mode=None, of=None, wrt=None,
                           num_full_jacs=_DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                           tol=_DEF_COMP_SPARSITY_ARGS['tol'],
                           orders=_DEF_COMP_SPARSITY_ARGS['orders'],
                           setup=False, run_model=False, fname=None, use_abs_names=False):
    """
    Compute simultaneous derivative colorings for the total jacobian of the given problem.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    mode : str or None
        The direction for computing derivatives.  If None, use problem._mode.
    of : iter of str or None
        Names of the 'response' variables.
    wrt : iter of str or None
        Names of the 'design' variables.
    num_full_jacs : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance used to determine if an array entry is nonzero.
    orders : int
        Number of orders above and below the tolerance to check during the tolerance sweep.
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.
    fname : filename or None
        File where output coloring info will be written. If None, no info will be written.
    use_abs_names : bool
        If True, use absolute naming for of and wrt variables.

    Returns
    -------
    Coloring
        See docstring for Coloring class.
    """
    driver = problem.driver

    abs_ofs, of_sizes = _get_response_info(driver, of, use_abs_names)
    abs_wrts, wrt_sizes = _get_desvar_info(driver, wrt, use_abs_names)

    model = problem.model

    if mode is None:
        if model._approx_schemes:
            mode = 'fwd'
        else:
            mode = problem._orig_mode
    if mode != problem._orig_mode and mode != problem._mode:
        raise RuntimeError("given mode (%s) does not agree with Problem mode (%s)" %
                           (mode, problem._mode))

    if model._approx_schemes:  # need to use total approx coloring
        if len(abs_ofs) != len(driver._responses):
            raise NotImplementedError("Currently there is no support for approx coloring when "
                                      "linear constraint derivatives are computed separately "
                                      "from nonlinear ones.")
        _initialize_model_approx(model, driver, abs_ofs, abs_wrts)
        if model._coloring_info['coloring'] is None:
            kwargs = {n: v for n, v in model._coloring_info.items()
                      if n in _DEF_COMP_SPARSITY_ARGS and v is not None}
            kwargs['method'] = list(model._approx_schemes)[0]
            model.declare_coloring(**kwargs)
        if run_model:
            problem.run_model()
        coloring = model._compute_approx_coloring(wrt_patterns='*',
                                                  method=list(model._approx_schemes)[0],
                                                  num_full_jacs=num_full_jacs, tol=tol,
                                                  orders=orders)[0]
    else:
        J, sparsity_info = _get_bool_total_jac(problem, num_full_jacs=num_full_jacs, tol=tol,
                                               orders=orders, setup=setup,
                                               run_model=run_model, of=abs_ofs, wrt=abs_wrts,
                                               use_abs_names=True)
        coloring = _compute_coloring(J, mode)
        if coloring is not None:
            coloring._row_vars = abs_ofs
            coloring._row_var_sizes = of_sizes
            coloring._col_vars = abs_wrts
            coloring._col_var_sizes = wrt_sizes

            # save metadata we used to create the coloring
            coloring._meta.update(sparsity_info)

            driver._total_jac = None

            system = problem.model
            if fname is not None:
                if ((system._full_comm is not None and system._full_comm.rank == 0) or
                        (system._full_comm is None and system.comm.rank == 0)):
                    coloring.save(fname)

    return coloring


def dynamic_total_coloring(driver, run_model=True, fname=None):
    """
    Compute simultaneous deriv coloring during runtime.

    Parameters
    ----------
    driver : <Driver>
        The driver performing the optimization.
    run_model : bool
        If True, call run_model before computing coloring.
    fname : str or None
        Name of file where coloring will be saved.

    Returns
    -------
    Coloring
        The computed coloring.
    """
    problem = driver._problem()
    if not problem.model._use_derivatives:
        simple_warning("Derivatives have been turned off. Skipping dynamic simul coloring.")
        return

    driver._total_jac = None

    problem.driver._coloring_info['coloring'] = None

    num_full_jacs = driver._coloring_info.get('num_full_jacs',
                                              _DEF_COMP_SPARSITY_ARGS['num_full_jacs'])
    tol = driver._coloring_info.get('tol', _DEF_COMP_SPARSITY_ARGS['tol'])
    orders = driver._coloring_info.get('orders', _DEF_COMP_SPARSITY_ARGS['orders'])

    coloring = compute_total_coloring(problem, num_full_jacs=num_full_jacs, tol=tol, orders=orders,
                                      setup=False, run_model=run_model, fname=fname,
                                      use_abs_names=True)

    if coloring is not None:
        if driver._coloring_info['show_sparsity']:
            coloring.display_txt()
        if driver._coloring_info['show_summary']:
            coloring.summary()

        driver._coloring_info['coloring'] = coloring
        driver._setup_simul_coloring()
        driver._setup_tot_jac_sparsity(coloring)

    return coloring


def _total_coloring_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao total_coloring' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file (pickle format)')
    parser.add_argument('-n', action='store', dest='num_jacs', type=int,
                        help='number of times to repeat derivative computation when '
                        'computing sparsity')
    parser.add_argument('--orders', action='store', dest='orders', type=int,
                        help='Number of orders (+/-) used in the tolerance sweep.')
    parser.add_argument('-t', '--tol', action='store', dest='tolerance', type=float,
                        help='tolerance used to determine if a jacobian entry is nonzero')
    parser.add_argument('--view', action='store_true', dest='show_sparsity',
                        help="Display a visualization of the final jacobian used to "
                        "compute the coloring.")
    parser.add_argument('--textview', action='store_true', dest='show_sparsity_text',
                        help="Display a text-based visualization of the colored jacobian.")
    parser.add_argument('--profile', action='store_true', dest='profile',
                        help="Do profiling on the coloring process.")


def _total_coloring_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao total_coloring'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    from openmdao.core.problem import Problem
    from openmdao.devtools.debug import profiling
    from openmdao.utils.general_utils import do_nothing_context

    global _use_total_sparsity

    _use_total_sparsity = False

    def _total_coloring(prob):
        if prob.model._use_derivatives:
            hooks._unregister_hook('final_setup', 'Problem')  # avoid recursive loop
            if options.outfile:
                outfile = os.path.abspath(options.outfile)
            else:
                outfile = os.path.join(prob.options['coloring_dir'], 'total_coloring.pkl')

            color_info = prob.driver._coloring_info
            if options.tolerance is None:
                options.tolerance = color_info['tol']
            if options.orders is None:
                options.orders = color_info['orders']
            if options.num_jacs is None:
                options.num_jacs = color_info['num_full_jacs']

            with profiling('coloring_profile.out') if options.profile else do_nothing_context():
                coloring = compute_total_coloring(prob,
                                                  num_full_jacs=options.num_jacs,
                                                  tol=options.tolerance,
                                                  orders=options.orders,
                                                  setup=False, run_model=True, fname=outfile,
                                                  use_abs_names=True)

            if coloring is not None:
                if options.show_sparsity_text:
                    coloring.display_txt()
                if options.show_sparsity:
                    coloring.display()
                coloring.summary()
        else:
            print("Derivatives are turned off.  Cannot compute simul coloring.")
        exit()

    hooks._register_hook('final_setup', 'Problem', post=_total_coloring)

    _load_and_exec(options.file[0], user_args)


def _partial_coloring_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao partial_color' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('--no_recurse', action='store_true', dest='norecurse',
                        help='Do not recurse from the provided system down.')
    parser.add_argument('--system', action='store', dest='system', default='',
                        help='pathname of system to color or to start recursing from.')
    parser.add_argument('-c', '--class', action='append', dest='classes', default=[],
                        help='compute a coloring for instances of the given class. '
                        'This option may be be used multiple times to specify multiple classes. '
                        'Class name can optionally contain the full module path.')
    parser.add_argument('--compute_decl_partials', action='store_true', dest='compute_decls',
                        help="Display declare_partials() calls required to specify computed "
                        "sparsity.")
    parser.add_argument('--method', action='store', dest='method',
                        help='approximation method ("fd" or "cs").')
    parser.add_argument('--step', action='store', dest='step',
                        help='approximation step size.')
    parser.add_argument('--form', action='store', dest='form',
                        help='approximation form ("forward", "backward", "central"). Only applies '
                        'to "fd" method.')
    parser.add_argument('--perturbation', action='store', dest='perturb_size', default=1e-3,
                        type=float, help='random perturbation size used when computing sparsity.')
    parser.add_argument('-n', action='store', dest='num_full_jacs', default=3, type=int,
                        help='number of times to repeat derivative computation when '
                        'computing sparsity')
    parser.add_argument('--tol', action='store', dest='tol', default=1.e-15, type=float,
                        help='tolerance used to determine if a jacobian entry is nonzero')
    parser.add_argument('--per_instance', action='store', dest='per_instance',
                        help='Generate a coloring file per instance, rather than a coloring file '
                        'per class.')
    parser.add_argument('--view', action='store_true', dest='show_sparsity',
                        help="Display a visualization of the colored jacobian.")
    parser.add_argument('--textview', action='store_true', dest='show_sparsity_text',
                        help="Display a text-based visualization of the colored jacobian.")
    parser.add_argument('--profile', action='store_true', dest='profile',
                        help="Do profiling on the coloring process.")


def _get_partial_coloring_kwargs(system, options):
    if options.system != '' and options.classes:
        raise RuntimeError("Can't specify --system and --class together.")

    kwargs = {}
    names = ('method', 'form', 'step', 'num_full_jacs', 'perturb_size', 'tol')
    for name in names:
        if getattr(options, name) is not None:
            kwargs[name] = getattr(options, name)

    recurse = not options.norecurse
    if recurse and not system._subsystems_allprocs:
        recurse = False
    kwargs['recurse'] = recurse

    per_instance = getattr(options, 'per_instance')
    kwargs['per_instance'] = (per_instance is None or
                              per_instance.lower() not in ['false', '0', 'no'])

    return kwargs


def _partial_coloring_cmd(options, user_args):
    """
    Return the hook function for 'openmdao partial_color'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.

    """
    from openmdao.core.problem import Problem
    from openmdao.core.component import Component
    from openmdao.devtools.debug import profiling
    from openmdao.utils.general_utils import do_nothing_context

    global _use_partial_sparsity, _force_dyn_coloring

    _use_partial_sparsity = False
    _force_dyn_coloring = True

    def _show(system, options, coloring):
        if options.show_sparsity_text and not coloring._meta.get('show_sparsity'):
            coloring.display_txt()
            print('\n')

        if options.show_sparsity and not coloring._meta.get('show_sparsity'):
            coloring.display()
            print('\n')

        if not coloring._meta.get('show_summary'):
            print("\nApprox coloring for '%s' (class %s)" % (system.pathname,
                                                             type(system).__name__))
            coloring.summary()
            print('\n')

        if options.compute_decls and isinstance(system, Component):
            print('\n    # add the following lines to class %s to declare sparsity' %
                  type(system).__name__)
            print(coloring.get_declare_partials_calls())

    def _partial_coloring(prob):
        if prob.model._use_derivatives:
            hooks._unregister_hook('final_setup', 'Problem')  # avoid recursive loop

            prob.run_model()  # get a consistent starting values for inputs and outputs

            with profiling('coloring_profile.out') if options.profile else do_nothing_context():
                if options.system == '':
                    system = prob.model
                    _initialize_model_approx(system, prob.driver)
                else:
                    system = prob.model._get_subsystem(options.system)
                if system is None:
                    raise RuntimeError("Can't find system with pathname '%s'." % options.system)

                kwargs = _get_partial_coloring_kwargs(system, options)

                if options.classes:
                    to_find = set(options.classes)
                    found = set()
                    kwargs['recurse'] = False
                    for s in system.system_iter(include_self=True, recurse=True):
                        for c in options.classes:
                            klass = s.__class__.__name__
                            mod = s.__class__.__module__
                            if c == klass or c == '.'.join([mod, klass]):
                                if c in to_find:
                                    found.add(c)
                                try:
                                    coloring = s._compute_approx_coloring(**kwargs)[0]
                                except Exception:
                                    tb = traceback.format_exc()
                                    print("The following error occurred while attempting to "
                                          "compute coloring for %s:\n %s" % (s.pathname, tb))
                                else:
                                    if coloring is not None:
                                        _show(s, options, coloring)
                                if options.norecurse:
                                    break
                    else:
                        if to_find - found:
                            raise RuntimeError("Failed to find any instance of classes %s" %
                                               sorted(to_find - found))
                else:
                    colorings = system._compute_approx_coloring(**kwargs)
                    if not colorings:
                        print("No coloring found.")
                    else:
                        for c in colorings:
                            if c is not None:
                                path = c._meta['pathname']
                                s = prob.model._get_subsystem(path) if path else prob.model
                                _show(s, options, c)
        else:
            print("Derivatives are turned off.  Cannot compute simul coloring.")
        exit()

    hooks._register_hook('final_setup', 'Problem', post=_partial_coloring)

    _load_and_exec(options.file[0], user_args)


def _view_coloring_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao view_coloring' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='coloring file.')
    parser.add_argument('--view', action='store_true', dest='show_sparsity',
                        help="Display a visualization of the colored jacobian.")
    parser.add_argument('--textview', action='store_true', dest='show_sparsity_text',
                        help="Display a text-based visualization of the colored jacobian.")
    parser.add_argument('-s', action='store_true', dest='subjac_sparsity',
                        help="Display sparsity patterns for subjacs.")
    parser.add_argument('-m', action='store_true', dest='show_meta',
                        help="Display coloring metadata.")
    parser.add_argument('-v', '--var', action='store', dest='color_var',
                        help='show the coloring (number of fwd and rev solves needed) '
                        'for a particular variable.')


def _view_coloring_exec(options, user_args):
    """
    Execute the 'openmdao view_coloring' command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Command line options after '--' (if any).  Passed to user script.
    """
    coloring = Coloring.load(options.file[0])
    if options.show_sparsity_text:
        coloring.display_txt()

    if options.show_sparsity:
        coloring.display()

    if options.subjac_sparsity:
        print("\nSubjacobian sparsity:")
        for tup in coloring._subjac_sparsity_iter():
            print("(%s, %s)\n   rows=%s\n   cols=%s" % tup[:4])
        print()

    if options.color_var is not None:
        fwd, rev = coloring.get_row_var_coloring(options.color_var)
        print("\nVar: %s  (fwd solves: %d,  rev solves: %d)\n" % (options.color_var, fwd, rev))

    if options.show_meta:
        print("\nColoring metadata:")
        pprint(coloring._meta)

    coloring.summary()


def _initialize_model_approx(model, driver, of=None, wrt=None):
    """
    Set up internal data structures needed for computing approx totals.
    """
    design_vars = driver._designvars

    if of is None:
        of = driver._get_ordered_nl_responses()
    if wrt is None:
        wrt = list(design_vars)

    # Initialization based on driver (or user) -requested "of" and "wrt".
    if (not model._owns_approx_jac or model._owns_approx_of is None or
            model._owns_approx_of != of or model._owns_approx_wrt is None or
            model._owns_approx_wrt != wrt):
        model._owns_approx_of = of
        model._owns_approx_wrt = wrt

        # Support for indices defined on driver vars.
        if MPI and model.comm.size > 1:
            of_idx = model._owns_approx_of_idx
            driver_resp = driver._dist_driver_vars
            for key, val in driver._responses.items():
                if val['indices'] is not None:
                    if val['distributed'] and key in driver_resp:
                        of_idx[key] = driver_resp[key][0]
                    else:
                        of_idx[key] = val['indices']
        else:
            model._owns_approx_of_idx = {
                key: val['indices'] for key, val in driver._responses.items()
                if val['indices'] is not None
            }
        model._owns_approx_wrt_idx = {
            key: val['indices'] for key, val in design_vars.items()
            if val['indices'] is not None
        }


def _get_coloring_meta(coloring=None):
    if coloring is None:
        dct = _DEF_COMP_SPARSITY_ARGS.copy()
        dct['coloring'] = None
        dct['dynamic'] = False
        dct['static'] = None
        return dct

    return coloring._meta.copy()
