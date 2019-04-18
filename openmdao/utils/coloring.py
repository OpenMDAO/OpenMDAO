"""
Routines to compute coloring for use with simultaneous derivatives.
"""
from __future__ import division, print_function

import os
import sys
import time
import warnings
import json
import pickle
from collections import OrderedDict, defaultdict
from itertools import combinations
from distutils.version import LooseVersion
from contextlib import contextmanager

from six import iteritems, string_types
from six.moves import range

import numpy as np
from scipy.sparse.compressed import get_index_dtype
from scipy.sparse import coo_matrix

from openmdao.jacobians.jacobian import Jacobian
from openmdao.matrices.matrix import sparse_types
from openmdao.utils.array_utils import array_viz
from openmdao.utils.general_utils import simple_warning
from openmdao.utils.mpi import MPI
from openmdao.approximation_schemes.approximation_scheme import _initialize_model_approx


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

# If this is True, then IF simul coloring/sparsity is specified, use it.
# If False, don't use it regardless.
# The command line total_coloring and sparsity commands make this False when generating a
# new coloring and/or sparsity.
_use_sparsity = True


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
    _shape : tuple of int (nrows, ncols) or None
        Tuple describing the shape of the sparsity matrix.
    _nzrows : ndarray of int or None
        Row indices of nonzero entries in the full jac sparsity matrix.
    _nzcols : ndarray of int or None
        Column indices of nonzero entries in the full jac sparsity matrix.
    _coloring_time : float or None
        If known, the time it took to compute the coloring.
    _sparsity_time : float or None
        If known, the time it took to compute the sparsity.
    _pct_nonzero : float or None
        If known, percentage of nonzero vs total array entries.
    _fwd : tuple (col_lists, row_maps)
        Contains lists of grouped columns and nonzero rows for each column for forward coloring.
    _rev : tuple (col_lists, row_maps)
        Contains lists of grouped columns and nonzero rows for each column for reverse coloring.
    _col_vars : list of str or None
        Names of variables corresponding to columns.
    _col_var_sizes : ndarray or None
        Sizes of column variables.
    _row_vars : list of str or None
        Names of variables corresponding to rows.
    _row_var_sizes : ndarray or None
        Sizes of row variables.
    _static : bool
        If True, this coloring was not generated dynamically during the current session.
    _meta : dict
        Dictionary of metadata used to create the coloring.
    _writers : dict
        Mapping of file extension to a tuple (func, open_str), where func writes
        the coloring in a specific format, and open_str is the string indicating
        if the file should be ascii ('w') or binary ('wb').
    """

    def __init__(self, sparsity=None, row_vars=None, row_var_sizes=None, col_vars=None,
                 col_var_sizes=None):
        """
        Initialize data structures.

        Parameters
        ----------
        sparsity : ndarray or None
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
        if sparsity is not None:
            self._nzrows, self._nzcols = np.nonzero(sparsity)
            self._shape = sparsity.shape
            self._pct_nonzero = np.count_nonzero(sparsity) / (self._shape[0] * self._shape[1]) * 100
        else:
            self._nzrows = self._nzcols = self._shape = self._pct_nonzero = None

        self._row_vars = row_vars
        self._row_var_sizes = row_var_sizes
        self._col_vars = col_vars
        self._col_var_sizes = col_var_sizes

        self._coloring_time = None
        self._sparsity_time = None
        self._static = True
        self._fwd = None
        self._rev = None
        self._meta = {}
        self._writers = {
            'json': (self._write_json, 'w'),
            'pkl': (self._write_pickle, 'wb'),
        }

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
        rev_size = self._shape[0] if self._shape else -1  # nrows
        fwd_size = self._shape[1] if self._shape else -1  # ncols

        tot_colors = self.total_solves()

        fwd_solves = rev_solves = 0
        if tot_colors == 0:  # no coloring found
            tot_colors = tot_size = min([rev_size, fwd_size])
            pct = 0.
        else:
            fwd_lists = self._fwd[0] if self._fwd else []
            rev_lists = self._rev[0] if self._rev else []

            if fwd_lists and not rev_lists:
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
                pct = ((tot_size - tot_colors) / tot_size * 100)

        if tot_size < 0:
            tot_size = '?'

        return tot_size, tot_colors, fwd_solves, rev_solves, pct

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
        Read the coloring object from the given file.

        The format is determined by the file extension.

        Parameters
        ----------
        fname : str
            Name of file to read from.

        Returns
        -------
        Coloring
            See docstring for Coloring class.
        """
        name, ext = fname.rsplit('.', 1)
        try:
            loader = _loaders[ext]
        except KeyError:
            raise RuntimeError("Can't find a coloring loader for extension '%s'." % ext)

        return loader(fname)

    @staticmethod
    def _load_pickle(fname):
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

    @staticmethod
    def _load_json(fname):
        """
        Read the coloring object from the given json file.

        Parameters
        ----------
        fname : str
            Name of file to read from.

        Returns
        -------
        Coloring
            See docstring for Coloring class.
        """
        with open(fname, 'r') as f:
            return _json2coloring(json.load(f))

    def save(self, stream):
        """
        Write the coloring object to the given stream.

        Parameters
        ----------
        stream : file-like or str
            File to save to.
        """
        if stream is None:
            return   # don't try to save

        if isinstance(stream, string_types):
            fname = stream
            name, fmt = fname.rsplit('.', 1)

            try:
                writer, otype = self._writers[fmt]
            except KeyError:
                raise RuntimeError("No writer available for format '%s'", fmt)

            with open(fname, otype) as f:
                writer(f)
        else:
            self._write_pickle(stream)

    def _write_pickle(self, stream):
        """
        Write the coloring to the given stream in pickle format.

        Parameters
        ----------
        stream : file-like
            Output stream.
        """
        pickle.dump(self, stream)

    def _write_json(self, stream):
        """
        Write the coloring to the given stream in json format.

        Parameters
        ----------
        stream : file-like
            Output stream.
        """
        tty = stream.isatty()
        none = 'null'
        sparsity = self.get_subjac_sparsity()
        modes = [self._fwd, self._rev]

        stream.write("{\n")
        for m, mode in enumerate(modes):
            if mode is None:
                continue

            name = 'column' if mode is self._fwd else 'row'
            mode_name = 'fwd' if mode is self._fwd else 'rev'
            lists, nonzero_entries = mode

            if m > 0:
                stream.write(",\n")

            stream.write('"%s": [[\n' % mode_name)
            last_idx = len(lists) - 1
            for i, lst in enumerate(lists):
                stream.write("   %s" % lst)
                if i < last_idx:
                    stream.write(",")

                if tty:
                    if i == 0:
                        stream.write("   # uncolored %ss" % name)
                    else:
                        stream.write("   # color %d" % i)

                stream.write("\n")

            stream.write("],\n[\n")
            last_idx = len(nonzero_entries) - 1
            for i, nonzeros in enumerate(nonzero_entries):
                if isinstance(nonzeros, list):
                    # convert to list to make json serializable
                    stream.write("   %s" % nonzeros)
                else:  # a full slice
                    stream.write("   %s" % none)

                if i < last_idx:
                    stream.write(",")

                if tty:
                    stream.write("   # %s %d" % (name, i))

                stream.write("\n")

            stream.write("]]")

        if sparsity:
            stream.write(',\n"sparsity": ')
            _write_sparsity(sparsity, stream)
        else:
            stream.write(',\n"sparsity": %s' % none)

        stream.write("\n}")

    def __repr__(self):
        """
        Return a short summary representation of this coloring.

        Returns
        -------
        str
            Brief summary.
        """
        if self._shape:
            shape = self._shape
        else:
            shape = '?'
        if self._fwd and self._rev:
            direction = 'bidirectional'
        elif self._fwd:
            direction = 'fwd'
        else:
            direction = 'rev'
        return 'Coloring (direction: %s, ncolors: %d, shape: %s' % (direction, self.total_solves(),
                                                                    shape)

    def summary(self, stream=sys.stdout):
        """
        Print a summary of this coloring.

        Parameters
        ----------
        stream : file-like
            Where the output will go.
        """
        stream.write('\n')
        if self._sparsity_time is not None:
            stream.write("Time to compute sparsity: %f sec.\n" % self._sparsity_time)
        if self._coloring_time is not None:
            stream.write("Time to compute coloring: %f sec.\n" % self._coloring_time)

        nrows = self._shape[0] if self._shape else -1
        ncols = self._shape[1] if self._shape else -1

        if self._pct_nonzero is None:
            stream.write("Jacobian shape: (%d, %d)\n" % (nrows, ncols))
        else:
            stream.write("Jacobian shape: (%d, %d)  (%5.2f%% nonzero)\n" % (nrows, ncols,
                                                                            self._pct_nonzero))

        if self._fwd is None and self._rev is None:
            tot_size = min(nrows, ncols)
            if tot_size < 0:
                tot_size = '?'
            stream.write("\nSimultaneous derivatives can't improve on the total number of solves "
                         "required (%s) for this configuration\n" % tot_size)
        else:
            tot_size, tot_colors, fwd_solves, rev_solves, pct = self._solves_info()

            stream.write("\nFWD solves: %d   REV solves: %d" % (fwd_solves, rev_solves))
            stream.write("\n\nTotal colors vs. total size: %d vs %s  (%.1f%% improvement)\n" %
                         (tot_colors, tot_size, pct))

    def display(self, stream=sys.stdout):
        """
        Display the structure of a boolean array with coloring info for each nonzero value.

        Forward mode colored nonzeros are denoted by 'f', reverse mode nonzeros by 'r',
        overlapping nonzeros by 'O' and uncolored nonzeros by 'x'.  Zeros are denoted by '.'.

        If names and sizes of row and column vars are known, print the name of the row var
        alongside each row and print the names of the column vars, aligned with each column,
        at the bottom.

        Parameters
        ----------
        stream : file-like
            Stream where output will be written.
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
                    stream.write(charr[r, c])
                stream.write(' %d\n' % r)
        else:
            # we have var name/size info, so mark rows/cols with their respective variable names
            rowstart = rowend = 0
            for rv, rvsize in zip(self._row_vars, self._row_var_sizes):
                rowend += rvsize
                for r in range(rowstart, rowend):
                    colstart = colend = 0
                    for cv, cvsize in zip(self._col_vars, self._col_var_sizes):
                        colend += cvsize
                        for c in range(colstart, colend):
                            stream.write(charr[r, c])
                        colstart = colend
                    stream.write(' %d  %s\n' % (r, rv))  # include row variable with each row
                rowstart = rowend

            # now print the column vars below the matrix, with each one spaced over to line up
            # with the appropriate starting column of the matrix ('|' marks the start of each var)
            start = 0
            for name, size in zip(self._col_vars, self._col_var_sizes):
                tab = ' ' * start
                stream.write('%s|%s\n' % (tab, name))
                start += size

        if has_overlap:
            raise RuntimeError("Internal coloring bug: jacobian has entries where fwd and rev "
                               "colorings overlap!")

    def get_subjac_sparsity(self):
        """
        Compute the sparsity structure of each subjacobian based on the full jac sparsity.

        If row/col variables and sizes are not known, returns None.

        Returns
        -------
        dict or None
            Mapping of (of, wrt) keys to thier corresponding (nzrows, nzcols, shape).
        """
        if self._row_vars and self._col_vars and self._row_var_sizes and self._col_var_sizes:
            J = np.zeros(self._shape, dtype=bool)
            J[self._nzrows, self._nzcols] = True
            return _jac2subjac_sparsity(J, self._row_vars, self._col_vars,
                                        self._row_var_sizes, self._col_var_sizes)

    def get_declare_partials_calls(self):
        """
        Return a string containing declare_partials() calls based on the subjac sparsity.

        Returns
        -------
        str
            A string containing a declare_partials() call for each nonzero subjac. This
            string may be cut and pasted into a component's setup() method.
        """
        subjac_sparsity = self.get_subjac_sparsity()

        if subjac_sparsity is None:
            raise RuntimeError("Coloring doesn't have enough info to compute subjac sparsity.")

        lines = []
        for of, sub in iteritems(subjac_sparsity):
            for wrt, tup in iteritems(sub):
                nzrows, nzcols, shape = tup
                if nzrows.size > 0:
                    lines.append("    self.declare_partials(of='%s', wrt='%s', rows=%s, cols=%s)" %
                                 (of, wrt, list(nzrows), list(nzcols)))
        return '\n'.join(lines)


_loaders = {
    'json': Coloring._load_json,
    'pkl': Coloring._load_pickle,
}


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
    # We build Jc from bottom up and Jr from right to left.
    r = M_row_nonzeros.argmin()
    c = M_col_nonzeros.argmin()

    nnz_r = M_row_nonzeros[r]
    nnz_c = M_col_nonzeros[c]

    Jc_nz_max = 0   # max row nonzeros in Jc
    Jr_nz_max = 0   # max col nonzeros in Jr

    while M_rows.size + M_cols.size > 0:
        if Jr_nz_max + max(Jc_nz_max, nnz_r) < (Jc_nz_max + max(Jr_nz_max, nnz_c)):
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

    # _check_coloring(J, coloring)

    coloring._coloring_time = time.time() - start_time

    return coloring


def _tol_sweep(arr, tol=1e-15, orders=20):
    """
    Find best tolerance 'around' tol to choose nonzero values of arr.

    Sweeps over tolerances +- 'orders' orders of magnitude around tol and picks the most
    stable one (one corresponding to the most repeated number of nonzero entries).

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
    float
        Chosen tolerance.
    int
        Number of repeated nonzero counts for the given tolerance.
    int
        Number of tolerances tested in the sweep.
    int
        Number of zero entries at chosen tolerance.
    """
    nzeros = defaultdict(list)
    itol = tol * 10.**orders
    smallest = tol / 10.**orders
    n_tested = 0
    while itol >= smallest:
        if itol < 1.:
            num_zero = arr[arr <= itol].size
            nzeros[num_zero].append(itol)
            n_tested += 1
        itol /= 10.

    # pick lowest tolerance corresponding to the most repeated number of 'zero' entries
    sorted_items = sorted(nzeros.items(), key=lambda x: len(x[1]), reverse=True)
    n_matching = len(sorted_items[0][1])

    if n_matching <= 1:
        raise RuntimeError("Could not find more than 1 tolerance to match any number of nonzeros. "
                           "This indicates that your tolerance sweep of +- %d orders, starting "
                           "from %s is not big enough.  To get a 'stable' sparsity pattern, "
                           "try re-running with a larger tolerance sweep.\nNonzeros found for "
                           "each tolerance: %s" % (orders, tol, sorted_items))

    good_tol = sorted_items[0][1][-1]

    return good_tol, n_matching, n_tested, sorted_items[0][0]


@contextmanager
def _compute_total_coloring_context(top):
    """
    Context manager for computing total jac sparsity for simultaneous coloring.

    Parameters
    ----------
    top : System
        Top of the system hierarchy where coloring will be done.
    """
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


def _get_bool_total_jac(prob, repeats=3, tol=1e-15, orders=20, setup=False, run_model=False):
    """
    Return a boolean version of the total jacobian.

    The jacobian is computed by calculating a total jacobian using _compute_totals 'repeats'
    times and adding the absolute values of those together, then dividing by 'repeats',
    then converting to a boolean array, specifying all entries below a tolerance as False and all
    others as True.  Prior to calling _compute_totals, all of the partial jacobians in the
    model are modified so that when any of their subjacobians are assigned a value, that
    value is populated with positive random numbers in the range [1.0, 2.0).

    Parameters
    ----------
    prob : Problem
        The Problem being analyzed.
    repeats : int
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

    Returns
    -------
    ndarray
        A boolean composite of 'repeats' total jacobians.
    """
    # clear out any old simul coloring info
    prob.driver._total_coloring = None
    prob.driver._res_jacs = {}

    if setup:
        prob.setup(mode=prob._mode)

    if run_model:
        prob.run_model(reset_iter_counts=False)

    with _compute_total_coloring_context(prob.model):
        start_time = time.time()
        fullJ = None
        for i in range(repeats):
            J = prob.driver._compute_totals(return_format='array')
            if fullJ is None:
                fullJ = np.abs(J)
            else:
                fullJ += np.abs(J)
        elapsed = time.time() - start_time

    fullJ /= repeats

    good_tol, nz_matches, n_tested, zero_entries = _tol_sweep(fullJ, tol, orders)

    print("\nUsing tolerance: %g" % good_tol)
    print("Most common number of zero entries (%d of %d) repeated %d times out of %d tolerances "
          "tested.\n" % (zero_entries, fullJ.size, nz_matches, n_tested))
    print("Full total jacobian was computed %d times, taking %f seconds." % (repeats, elapsed))
    print("Total jacobian shape:", fullJ.shape, "\n")

    boolJ = np.zeros(fullJ.shape, dtype=bool)
    boolJ[fullJ > good_tol] = True

    return boolJ, elapsed


def _jac2subjac_sparsity(J, ofs, wrts, of_sizes, wrt_sizes, cvt2json=False):
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
    cvt2json : bool
        If True, perform some data conversions to make data serializable to json.

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

            if cvt2json:
                # convert to make JSON serializable
                irows = [int(i) for i in irows]
                icols = [int(i) for i in icols]

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

    stream.write("}")


def _json2coloring(coloring):
    """
    Convert all of the None entries in rowcol_map to full slices.

    Parameters
    ----------
    coloring : dict
        Dict of coloring metadata.

    Returns
    -------
    dict
        Dict of coloring metadata.
    """
    full_slice = slice(None)
    for mode in ('fwd', 'rev'):
        if mode in coloring:
            rcmap = coloring[mode][1]
            for i, entry in enumerate(rcmap):
                if entry is None:
                    rcmap[i] = full_slice

    return coloring


def _get_desvar_sizes(driver, names):
    desvars = driver._designvars
    return [desvars[n]['size'] for n in names]


def _get_response_sizes(driver, names):
    responses = driver._responses
    return [responses[n]['size'] for n in names]


def get_tot_jac_sparsity(problem, mode='fwd', repeats=1, tol=1.e-15,
                         setup=False, run_model=False):
    """
    Compute derivative sparsity for the given problem.

    Parameters
    ----------
    problem : Problem
        The Problem being analyzed.
    mode : str
        Derivative direction.
    repeats : int
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

    J, _ = _get_bool_total_jac(problem, repeats=repeats, tol=tol, setup=setup, run_model=run_model)

    ofs = driver._get_ordered_nl_responses()
    wrts = list(driver._designvars)
    of_sizes = _get_response_sizes(driver, ofs)
    wrt_sizes = _get_desvar_sizes(driver, wrts)

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

    if mode == 'auto':  # use bidirectional coloring
        return MNCO_bidir(J)

    rev = mode == 'rev'
    nrows, ncols = J.shape

    coloring = Coloring(sparsity=J)

    if rev:
        J = J.T
    col_groups = _split_groups(_get_full_disjoint_cols(J))

    full_slice = slice(None)
    col2rows = [full_slice] * J.shape[1]  # will contain list of nonzero rows for each column
    for lst in col_groups:
        for col in lst:
            col2rows[col] = np.nonzero(J[:, col])[0]

    if mode == 'fwd':
        coloring._fwd = (col_groups, col2rows)
    else:
        coloring._rev = (col_groups, col2rows)

    coloring._coloring_time = time.time() - start_time

    return coloring


def compute_total_coloring(problem, mode=None, repeats=1, tol=1.e-15, orders=20, setup=False,
                           run_model=False, bool_jac=None, fname=None):
    """
    Compute simultaneous derivative colorings for the total jacobian of the given problem.

    Parameters
    ----------
    problem : Problem or None
        The Problem being analyzed.
    mode : str or None
        The direction for computing derivatives.  If None, use problem._mode.
    repeats : int
        Number of times to repeat total jacobian computation.
    tol : float
        Tolerance used to determine if an array entry is nonzero.
    orders : int
        Number of orders above and below the tolerance to check during the tolerance sweep.
    setup : bool
        If True, run setup before calling compute_totals.
    run_model : bool
        If True, run run_model before calling compute_totals.
    bool_jac : ndarray
        If problem is not supplied, a previously computed boolean jacobian can be used.
    fname : filename or None
        File where output coloring info will be written. If None, no info will be written.

    Returns
    -------
    Coloring
        See docstring for Coloring class.
    """
    sparsity = None

    if problem is not None:
        driver = problem.driver
        ofs = driver._get_ordered_nl_responses()
        wrts = list(driver._designvars)
        of_sizes = _get_response_sizes(driver, ofs)
        wrt_sizes = _get_desvar_sizes(driver, wrts)

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
            if len(ofs) != len(driver._responses):
                raise NotImplementedError("Currently there is no support for approx coloring when "
                                          "linear constraint derivatives are computed separately "
                                          "from nonlinear ones.")
            _initialize_model_approx(model, driver, ofs, wrts)
            coloring = model.compute_approx_coloring(wrt='*', method=list(model._approx_schemes)[0],
                                                     repeats=repeats)
        else:
            J, sparsity_time = _get_bool_total_jac(problem, repeats=repeats, tol=tol,
                                                   orders=orders, setup=setup,
                                                   run_model=run_model)
            coloring = _compute_coloring(J, mode)
            coloring._row_vars = ofs
            coloring._row_var_sizes = of_sizes
            coloring._col_vars = wrts
            coloring._col_var_sizes = wrt_sizes
            coloring._sparsity_time = sparsity_time

        driver._total_jac = None

    elif bool_jac is not None:
        J = bool_jac
        time_sparsity = 0.
        if mode is None:
            mode = 'auto'
        driver = None
        coloring = _compute_coloring(J, mode)
    else:
        raise RuntimeError("You must supply either problem or bool_jac to "
                           "compute_total_coloring().")

    if fname is not None:
        coloring.save(fname)

    return coloring


def dynamic_sparsity(driver):
    """
    Compute deriv sparsity during runtime.

    Parameters
    ----------
    driver : <Driver>
        The driver performing the optimization.
    """
    problem = driver._problem
    if not problem.model._use_derivatives:
        simple_warning("Derivatives have been turned off. Skipping dynamic sparsity computation.")
        return

    driver._total_jac = None
    repeats = driver.options['dynamic_derivs_repeats']

    # save the total_sparsity.json file for later inspection
    sparsity, _ = get_tot_jac_sparsity(problem, mode=problem._mode, repeats=repeats)

    with open("total_sparsity.json", "w") as f:
        _write_sparsity(sparsity, f)

    driver.set_total_jac_sparsity(sparsity)
    driver._setup_tot_jac_sparsity()


def dynamic_total_coloring(driver, run_model=True):
    """
    Compute simultaneous deriv coloring during runtime.

    Parameters
    ----------
    driver : <Driver>
        The driver performing the optimization.
    run_model : bool
        If True, call run_model before computing coloring.
    """
    problem = driver._problem
    if not problem.model._use_derivatives:
        simple_warning("Derivatives have been turned off. Skipping dynamic simul coloring.")
        return

    driver._total_jac = None

    if driver._total_coloring is not None:
        if driver._total_coloring._static:
            raise RuntimeError("A total coloring was already set and would be overridden by the "
                               "requested dynamic coloring.")
        else:
            # this is not the first time run_driver has been executed.  Go ahead and regen the
            # coloring just in case something has changed since the first call, but reset anything
            # that will raise exceptions about multiple definition.
            driver._total_jac_sparsity = None  # prevent complaints about redefining the sparsity

    # if model is using approx derivs, we must ensure that we're starting from a valid point
    if problem.model._approx_schemes:
        problem.run_model()

    problem.driver._total_coloring = None
    problem.driver._res_jacs = {}

    coloring = compute_total_coloring(problem,
                                      repeats=driver.options['dynamic_derivs_repeats'],
                                      tol=1.e-15,
                                      setup=False, run_model=run_model,
                                      fname='total_coloring.pkl')

    coloring.summary()

    coloring._static = False
    driver.set_coloring_spec(coloring)
    driver._setup_simul_coloring()


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
    parser.add_argument('-n', action='store', dest='num_jacs', default=3, type=int,
                        help='number of times to repeat derivative computation when '
                        'computing sparsity')
    parser.add_argument('-t', '--tol', action='store', dest='tolerance', default=1.e-15, type=float,
                        help='tolerance used to determine if a jacobian entry is nonzero')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final jacobian used to "
                        "compute the coloring.")
    parser.add_argument('--activate', action='store_true', dest='activate',
                        help="Activate the computed coloring and continue running the script.")
    parser.add_argument('--no-sparsity', action='store_true', dest='no_sparsity',
                        help="Exclude the sparsity structure from the coloring data structure.")
    parser.add_argument('--profile', action='store_true', dest='profile',
                        help="Do profiling on the coloring process.")


def _total_coloring_cmd(options):
    """
    Return the post_setup hook function for 'openmdao total_coloring'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    from openmdao.core.problem import Problem
    from openmdao.devtools.debug import profiling
    from openmdao.utils.general_utils import do_nothing_context

    global _use_sparsity

    _use_sparsity = False

    def _total_coloring(prob):
        global _use_sparsity
        if prob.model._use_derivatives:
            Problem._post_setup_func = None  # avoid recursive loop
            do_sparsity = not options.no_sparsity

            with profiling('coloring_profile.out') if options.profile else do_nothing_context():
                coloring = compute_total_coloring(prob,
                                                  repeats=options.num_jacs, tol=options.tolerance,
                                                  setup=False, run_model=True,
                                                  fname=options.outfile)

            if options.show_jac:
                coloring.display()
            coloring.summary()
            if options.activate:
                prob.driver.set_coloring_spec(coloring)
                prob.driver._setup_simul_coloring()
                if do_sparsity:
                    prob.driver._setup_tot_jac_sparsity()
        else:
            print("Derivatives are turned off.  Cannot compute simul coloring.")
            exit()
        if options.activate:
            _use_sparsity = True
        else:
            exit()
    return _total_coloring


def get_coloring_fname(system, directory=None, per_instance=False):
    """
    Return the full pathname to a coloring file, generating a default name if necessary.

    Parameters
    ----------
    system : System
        The System having its coloring saved or loaded.
    directory : str or None
        Pathname of the directory where the coloring file will be saved
    per_instance : bool
        If True, the file will be named for each instance of a given class.
        Otherwise, the file will be named based on the class name.

    Returns
    -------
    str
        Full pathname of the coloring file.
    """
    if directory is None:
        return  # the file won't be saved if we return None here

    directory = os.path.abspath(directory)

    if per_instance:
        if system.pathname:
            return os.path.join(directory,
                                'coloring_' + system.pathname.replace('.', '_') + '.pkl')
        else:
            return os.path.join(directory, 'total_coloring.pkl')
    else:
        fn = '_'.join([system.__class__.__module__.replace('.', '_'), system.__class__.__name__])
        return os.path.join(directory, fn + '.pkl')


def _partial_coloring_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao partial_color' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('--dir', action='store', dest='directory',
                        help='Directory where coloring files are saved.')
    parser.add_argument('-r', '--recurse', action='store_true', dest='recurse',
                        help='Recurse from the provided system down.')
    parser.add_argument('--system', action='store', dest='system', default='',
                        help='pathname of system to color or to start recursing from if --recurse'
                        ' is set.')
    parser.add_argument('-c', '--class', action='append', dest='classes', default=[],
                        help='compute a coloring for instances of the given class. '
                        'This option may be be used multiple times to specify multiple classes. '
                        'Class name can optionally contain the full module path. '
                        'Not compatible with the --recurse option.')
    parser.add_argument('--first_only', action='store_true', dest='first_only',
                        help="If using the --class option, only generate coloring for the first "
                        "instance found for each class.")
    parser.add_argument('--activate', action='store_true', dest='activate',
                        help="Activate the computed coloring(s) and continue running the script.")
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
    parser.add_argument('--sparsity_tol', action='store', dest='tol', default=1e-15, type=float,
                        help='tolerance used to determine nonzero entries when computing sparsity.')
    parser.add_argument('-n', action='store', dest='repeats', default=3, type=int,
                        help='number of times to repeat derivative computation when '
                        'computing sparsity')
    parser.add_argument('--tol', action='store', dest='tolerance', default=1.e-15, type=float,
                        help='tolerance used to determine if a jacobian entry is nonzero')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final jacobian used to "
                        "compute the coloring.")
    parser.add_argument('--profile', action='store_true', dest='profile',
                        help="Do profiling on the coloring process.")


def _get_partial_coloring_kwargs(options):
    if options.system != '' and options.classes:
        raise RuntimeError("Can't specify --system and --class together.")
    if options.classes:
        if options.recurse:
            raise RuntimeError("Can't specify --class if --recurse option is set.")

    kwargs = {}
    names = ('method', 'form', 'step', 'repeats', 'perturb_size', 'tol', 'directory',
             'recurse')
    for name in names:
        if getattr(options, name):
            kwargs[name] = getattr(options, name)

    return kwargs


def _partial_coloring_cmd(options):
    """
    Return the post_setup hook function for 'openmdao partial_color'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    from openmdao.core.problem import Problem
    from openmdao.core.component import Component
    from openmdao.devtools.debug import profiling
    from openmdao.utils.general_utils import do_nothing_context

    global _use_sparsity

    _use_sparsity = False

    def _partial_coloring(prob):
        global _use_sparsity
        if prob.model._use_derivatives:
            Problem._post_setup_func = None  # avoid recursive loop

            prob.run_model()  # get a consistent starting values for inputs and outputs

            with profiling('coloring_profile.out') if options.profile else do_nothing_context():
                if options.system == '':
                    system = prob.model
                    _initialize_model_approx(system, prob.driver)
                else:
                    system = prob.model.get_subsystem(options.system)
                if system is None:
                    raise RuntimeError("Can't find system with pathname '%s'." % options.system)

                kwargs = _get_partial_coloring_kwargs(options)
                if 'directory' not in kwargs:
                    kwargs['directory'] = \
                        os.path.join(os.path.abspath(os.path.dirname(options.file[0])),
                                     'coloring_files')

                if options.classes:
                    to_find = set(options.classes)
                    for s in system.system_iter(include_self=True, recurse=True):
                        for c in options.classes:
                            klass = s.__class__.__name__
                            mod = s.__class__.__module__
                            if c == klass or c == '.'.join([mod, klass]):
                                if c in to_find:
                                    to_find.remove(c)
                                coloring = s.compute_approx_coloring(**kwargs)
                                print("Approx coloring for '%s' (class %s)\n" % (s.pathname, klass))
                                if options.show_jac:
                                    coloring.display()
                                coloring.summary()
                                print('\n')
                                if options.compute_decls and isinstance(s, Component):
                                    print('    # add the following lines to class %s to declare '
                                          'sparsity' % klass)
                                    print(coloring.get_declare_partials_calls())
                                if options.activate:
                                    s.set_coloring_spec(coloring)
                                    s._setup_static_approx_coloring()
                                break
                        if not to_find and options.first_only:
                            break
                    else:
                        if to_find:
                            raise RuntimeError("Failed to find any instance of classes %s" %
                                               sorted(to_find))
                else:
                    coloring = system.compute_approx_coloring(**kwargs)

                    print("Approx coloring for '%s' (class %s)\n" % (s.pathname, klass))
                    if options.show_jac:
                        coloring.display()
                    coloring.summary()
                    print('\n')

                    if options.activate:
                        system.set_coloring_spec(coloring)
                        system._setup_static_approx_coloring()
        else:
            print("Derivatives are turned off.  Cannot compute simul coloring.")
        if options.activate:
            # instead of exiting, keep running using the computed coloring(s)
            _use_sparsity = True
        else:
            exit()
    return _partial_coloring


def _sparsity_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao sparsity' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('-o', action='store', dest='outfile', help='output file (json format).')
    parser.add_argument('-n', action='store', dest='num_jacs', default=3, type=int,
                        help='number of times to repeat total derivative computation.')
    parser.add_argument('-t', action='store', dest='tolerance', default=1.e-15, type=float,
                        help='tolerance used to determine if a total jacobian entry is nonzero.')
    parser.add_argument('-j', '--jac', action='store_true', dest='show_jac',
                        help="Display a visualization of the final total jacobian used to "
                        "compute the sparsity.")


def _sparsity_cmd(options):
    """
    Return the post_setup hook function for 'openmdao total_sparsity'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.

    Returns
    -------
    function
        The post-setup hook function.
    """
    from openmdao.core.problem import Problem
    global _use_sparsity

    _use_sparsity = False

    def _sparsity(prob):
        Problem._post_setup_func = None  # avoid recursive loop
        sparsity, J = get_tot_jac_sparsity(prob, repeats=options.num_jacs, tol=options.tolerance,
                                           mode=prob._mode, setup=True, run_model=True)

        if options.outfile is None:
            outfile = sys.stdout
        else:
            outfile = open(options.outfile, 'w')
        _write_sparsity(sparsity, outfile)

        if options.show_jac:
            print("\n")
            ofs = prob.driver._get_ordered_nl_responses()
            wrts = list(prob.driver._designvars)
            array_viz(J, prob, ofs, wrts)

        exit(0)
    return _sparsity


def _check_coloring(J, coloring):
    """
    Raise an exception if any problems are found with the coloring info.

    Parameters
    ----------
    J : ndarray
        Jacobian sparsity matrix.
    coloring : dict
        Metadata required for coloring.
    """
    # check for any overlapping nonzeros
    fwd_coloring = coloring.get('fwd')
    rev_coloring = coloring.get('rev')

    if fwd_coloring is not None:
        col_idx_groups = fwd_coloring[0]
        all_cols = set()
        for grp in col_idx_groups:
            all_cols.update(grp)

        col2row = fwd_coloring[1]
        fwd_nzs = []
        for col, rows in enumerate(col2row):
            if rows is not None:
                fwd_nzs.extend([(r, col) for r in rows])
            elif col in all_cols:  # full solve
                full_nzs = [(r, col) for r in np.nonzero(J[:, col])[0]]
                fwd_nzs.extend(full_nzs)

        fwd_nz_set = set(fwd_nzs)
        if len(fwd_nzs) != len(fwd_nz_set):
            raise RuntimeError("Duplicated nonzeros found in fwd coloring.")

        # find any nonzeros that are left out of any of our columns
        for c in all_cols:
            rnz = col2row[c]
            Jnz = np.nonzero(J[:, c])[0]
            missing = set(Jnz).difference(rnz)
            print("missing for col %d" % c, missing)

    if rev_coloring is not None:
        row_idx_groups = rev_coloring[0]
        all_rows = set()
        for grp in row_idx_groups:
            all_rows.update(grp)

        row2col = rev_coloring[1]
        rev_nzs = []
        for row, cols in enumerate(row2col):
            if cols is not None:
                rev_nzs.extend([(row, c) for c in cols])
            elif row in all_rows:  # full solve
                rev_nzs.extend([(row, c) for c in np.nonzero(J[row])[0]])

        rev_nz_set = set(rev_nzs)
        if len(rev_nzs) != len(rev_nz_set):
            raise RuntimeError("Duplicated nonzeros found in rev coloring.")

        # find any nonzeros that are left out of any of our columns
        for r in all_rows:
            cnz = row2col[r]
            Jnz = np.nonzero(J[r])[0]
            missing = set(Jnz).difference(cnz)
            print("missing for row %d" % r, missing)

    if fwd_coloring is not None and rev_coloring is not None:
        common = fwd_nz_set.intersection(rev_nzs)
        if common:
            raise RuntimeError("Coloring has overlapping nonzeros: %s" % list(common))

        computed_nzs = len(fwd_nz_set) + len(rev_nz_set)
        nzs = np.count_nonzero(J)
        if computed_nzs != nzs:
            raise RuntimeError("Colored nonzeros (%d) != nonzeros in J (%d)" % (computed_nzs, nzs))
