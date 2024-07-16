"""
Routines to compute coloring for use with simultaneous derivatives.
"""
import datetime
import io
import os
import time
import pickle
import sys
import tempfile
import traceback
import pathlib
import webbrowser
from itertools import combinations, groupby
from contextlib import contextmanager
from pprint import pprint
from packaging.version import Version


import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix

from openmdao.core.constants import INT_DTYPE, _DEFAULT_OUT_STREAM
from openmdao.utils.general_utils import _src_name_iter, _src_or_alias_item_iter, \
    _convert_auto_ivc_to_conn_name, pattern_filter
import openmdao.utils.hooks as hooks
from openmdao.utils.file_utils import _load_and_exec
from openmdao.utils.om_warnings import issue_warning, OMDeprecationWarning, DerivativesWarning
from openmdao.utils.reports_system import register_report
from openmdao.devtools.memory import mem_usage
from openmdao.utils.name_maps import rel_name2abs_name

try:
    import matplotlib as mpl
    from matplotlib import pyplot

    if Version(mpl.__version__) < Version("3.6"):
        from matplotlib import cm
except ImportError:
    mpl = None

try:
    from bokeh.models import CategoricalColorMapper, ColumnDataSource, CustomJSHover, \
        Div, HoverTool, PreText
    from bokeh.layouts import column
    from bokeh.palettes import Blues256, Reds256, gray, interp_palette
    from bokeh.plotting import figure
    import bokeh.resources as bokeh_resources
    from bokeh.transform import transform
    import bokeh.io
except ImportError:
    bokeh_resources = None


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

_default_coloring_imagefile = 'coloring.png'

# default values related to the computation of a sparsity matrix
_DEF_COMP_SPARSITY_ARGS = {
    'tol': 1e-25,     # use this tolerance to determine what's a zero when determining sparsity
    'orders': None,   # num orders += around 'tol' for the tolerance sweep when determining sparsity
    'num_full_jacs': 3,      # number of full jacobians to generate before computing sparsity
    'perturb_size': 1e-9,    # size of input/output perturbation during generation of sparsity
    'min_improve_pct': 5.,   # don't use coloring unless at least 5% decrease in number of solves
    'show_summary': True,    # if True, print a short summary of the coloring
    'show_sparsity': False,  # if True, show a plot of the sparsity
    'use_scaling': False,    # if True, use driver scaling when computing sparsity
                             # (total coloring only)
}

_COLORING_VERSION = '1.0'


# A dict containing colorings that have been generated during the current execution.
# When a dynamic coloring is specified for a particular class and per_instance is False,
# this dict can be checked for an existing class version of the coloring that can be used
# for that instance.
_CLASS_COLORINGS = {}


class ColoringMeta(object):
    """
    Container for all metadata relevant to a coloring.

    Parameters
    ----------
    num_full_jacs : int
        Number of full jacobians to generate while computing sparsity.
    tol : float
        Use this tolerance to determine what's a zero when determining sparsity.
    orders : int or None
        Number of orders += around 'tol' for the tolerance sweep when determining sparsity.  If
        None, no tolerance sweep will be performed and whatever 'tol' is specified will be used.
    min_improve_pct : float
        Don't use coloring unless at least min_improve_pct percentage decrease in number of solves.
    show_summary : bool
        If True, print a short summary of the coloring. Defaults to True.
    show_sparsity : bool
        If True, show a plot of the sparsity. Defaults to False.
    dynamic : bool
        True if dynamic coloring is being used.
    static : Coloring, str, or None
        If a Coloring object, just use that.  If a filename, load the coloring from that file.
        If None, do not attempt to use a static coloring.
    perturb_size : float
        Size of input/output perturbation during generation of sparsity.
    use_scaling : bool
        If True, use driver scaling when computing sparsity.
    msginfo : str
        Prefix for warning/error messages.

    Attributes
    ----------
    num_full_jacs : int
        Number of full jacobians to generate while computing sparsity.
    tol : float
        Use this tolerance to determine what's a zero when determining sparsity.
    orders : int or None
        Number of orders += around 'tol' for the tolerance sweep when determining sparsity.  If
        None, no tolerance sweep will be performed and whatever 'tol' is specified will be used.
    min_improve_pct : float
        Don't use coloring unless at least min_improve_pct percentage decrease in number of solves.
    show_summary : bool
        If True, print a short summary of the coloring. Defaults to True.
    show_sparsity : bool
        If True, show a plot of the sparsity. Defaults to False.
    dynamic : bool
        True if dynamic coloring is being used.
    static : Coloring, str, or None
        If a Coloring object, just use that.  If a filename, load the coloring from that file.
        If None, do not attempt to use a static coloring.
    perturb_size : float
        Size of input/output perturbation during generation of sparsity.
    use_scaling : bool
        If True, use driver scaling when computing sparsity.
    msginfo : str
        Prefix for warning/error messages.
    _coloring : Coloring or None
        The coloring object.
    _failed : bool
        If True, coloring was already generated but failed.
    _approx : bool
        If True, this is an approx coloring.
    """

    _meta_names = {'num_full_jacs', 'tol', 'orders', 'min_improve_pct', 'show_summary',
                   'show_sparsity', 'dynamic', 'perturb_size', 'use_scaling', 'msginfo'}

    def __init__(self, num_full_jacs=3, tol=1e-25, orders=None, min_improve_pct=5.,
                 show_summary=True, show_sparsity=False, dynamic=False, static=None,
                 perturb_size=1e-9, use_scaling=False, msginfo=''):
        """
        Initialize data structures.
        """
        self.num_full_jacs = num_full_jacs
        self.tol = tol
        self.orders = orders
        self.min_improve_pct = min_improve_pct
        self.show_summary = show_summary
        self.show_sparsity = show_sparsity
        self.dynamic = dynamic
        self.static = static
        self.perturb_size = perturb_size
        self.use_scaling = use_scaling
        self.msginfo = msginfo
        self._coloring = None
        self._failed = False
        self._approx = False

    def do_compute_coloring(self):
        """
        Return True if coloring should be computed.

        Returns
        -------
        bool
            True if coloring should be computed.
        """
        return self.coloring is None and not self._failed

    def update(self, dct):
        """
        Update the metadata.

        Parameters
        ----------
        dct : dict
            Dictionary of metadata.
        """
        for name, val in dct.items():
            if name in self._meta_names:
                setattr(self, name, val)

    def display(self):
        """
        Display information about the coloring.
        """
        if self.coloring is None:
            if self.show_summary or self.show_sparsity:
                print("No coloring was computed successfully.")
        else:
            if self.show_summary:
                self.coloring.summary()
            if self.show_sparsity:
                self.coloring.display_bokeh(show=True)

    def __iter__(self):
        """
        Iterate over the metadata.

        Yields
        ------
        (str, object)
            Tuple containing the name and value of each metadata item.
        """
        for name in self._meta_names:
            yield name, getattr(self, name)

    def __getitem__(self, name):
        """
        Get the value of the named metadata.

        Parameters
        ----------
        name : str
            Name of the metadata.

        Returns
        -------
        object
            Value of the named metadata.
        """
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(name)

    def __setitem__(self, name, value):
        """
        Set the value of the named metadata.

        Parameters
        ----------
        name : str
            Name of the metadata.
        value : object
            Value of the metadata.
        """
        if name in self.__dict__ or name == 'coloring':
            setattr(self, name, value)
        else:
            raise KeyError(name)

    def get(self, name, default=None):
        """
        Get the value of the named metadata.

        Parameters
        ----------
        name : str
            Name of the metadata.
        default : object or None
            The value to return if the named metadata is not found.

        Returns
        -------
        object
            Value of the named metadata.
        """
        try:
            return getattr(self, name)
        except AttributeError:
            return default

    @property
    def coloring(self):
        """
        Return the coloring.

        Returns
        -------
        Coloring or None
            The coloring.
        """
        return self._coloring

    @coloring.setter
    def coloring(self, coloring):
        """
        Set the coloring.

        Parameters
        ----------
        coloring : Coloring or None
            The coloring.
        """
        self.set_coloring(coloring)

    def set_coloring(self, coloring, msginfo=''):
        """
        Set the coloring.

        Parameters
        ----------
        coloring : Coloring or None
            The coloring.
        msginfo : str
            Prefix for warning/error messages.
        """
        if coloring is None:
            self._coloring = None
            self._failed = False
        elif self._pct_improvement_good(coloring, msginfo):
            self._coloring = coloring
            self._failed = False
        else:
            # if the improvement wasn't large enough, don't use coloring
            self.coloring = None
            self._failed = True

    def reset_coloring(self):
        """
        Reset the coloring to None.
        """
        self._coloring = None
        self._failed = False

    def _pct_improvement_good(self, coloring, msginfo=''):
        """
        Return True if the percentage improvement is greater than the minimum allowed.

        Parameters
        ----------
        coloring : Coloring
            The coloring.
        msginfo : str
            Prefix for warning/error messages.

        Returns
        -------
        bool
            True if the percentage improvement is greater than the minimum allowed.
        """
        if coloring is None:
            return False

        pct = coloring._solves_info()[-1]
        if self.min_improve_pct <= pct:
            return True
        else:
            msg = f"Coloring was deactivated.  Improvement of {pct:.1f}% was less than min " \
                  f"allowed ({self.min_improve_pct:.1f}%)."
            issue_warning(msg, prefix=msginfo, category=DerivativesWarning)
            return False

    def copy(self):
        """
        Return a new object with metadata copied from this object.

        Returns
        -------
        ColoringMeta
            Copy of the metadata.
        """
        return type(self)(**dict(self))


class Partial_ColoringMeta(ColoringMeta):
    """
    Container for all metadata relevant to a partial coloring.

    Parameters
    ----------
    wrt_patterns : list/tuple of str or str
        Patterns used to match wrt variables.
    method : str
        Finite differencing method ('fd' or 'cs').
    form : str
        Form of the derivatives ('forward', 'backward', or 'central').  Only used if method is 'fd'.
    step : float
        Step size for 'fd', or 'cs'.
    per_instance : bool
        Assume each instance can have a different coloring, so coloring will not be saved as
        a class coloring.
    perturb_size : float
        Size of input/output perturbation during generation of sparsity.
    num_full_jacs : int
        Number of full jacobians to generate while computing sparsity.
    tol : float
        Use this tolerance to determine what's a zero when determining sparsity.
    orders : int or None
        Number of orders += around 'tol' for the tolerance sweep when determining sparsity.  If
        None, no tolerance sweep will be performed and whatever 'tol' is specified will be used.
    min_improve_pct : float
        Don't use coloring unless at least min_improve_pct percentage decrease in number of solves.
    show_summary : bool
        If True, print a short summary of the coloring. Defaults to True.
    show_sparsity : bool
        If True, show a plot of the sparsity. Defaults to False.
    dynamic : bool
        True if dynamic coloring is being used.
    static : Coloring, str, or None
        If a Coloring object, just use that.  If a filename, load the coloring from that file.
        If None, do not attempt to use a static coloring.
    msginfo : str
        Prefix for warning/error messages.

    Attributes
    ----------
    wrt_patterns : list/tuple of str or str
        Patterns used to match wrt variables.
    method : str
        Finite differencing method ('fd' or 'cs').
    form : str
        Form of the derivatives ('forward', 'backward', or 'central').  Only used if method is 'fd'.
    step : float
        Step size for 'fd', or 'cs'.
    per_instance : bool
        Assume each instance can have a different coloring, so coloring will not be saved as
        a class coloring.
    fname : str or None
        Filename where coloring is stored.
    wrt_matches : set of str or None
        Where matched wrt names are stored.
    """

    _meta_names = {'wrt_patterns', 'per_instance', 'method', 'form', 'step'}
    _meta_names.update(ColoringMeta._meta_names)

    def __init__(self, wrt_patterns=('*',), method='fd', form=None, step=None, per_instance=True,
                 perturb_size=1e-9, num_full_jacs=3, tol=1e-25, orders=None, min_improve_pct=5.,
                 show_summary=True, show_sparsity=False, dynamic=False, static=None, msginfo=''):
        """
        Initialize data structures.
        """
        super().__init__(num_full_jacs=num_full_jacs, tol=tol, orders=orders,
                         min_improve_pct=min_improve_pct, show_summary=show_summary,
                         show_sparsity=show_sparsity, dynamic=dynamic, static=static,
                         perturb_size=perturb_size, msginfo=msginfo)
        if wrt_patterns is None:
            wrt_patterns = ()
        elif isinstance(wrt_patterns, str):
            wrt_patterns = (wrt_patterns,)
        else:
            wrt_patterns = tuple(wrt_patterns)
        self.wrt_patterns = wrt_patterns
        self.method = method
        self.form = form
        self.step = step
        self.per_instance = per_instance
        self.fname = None
        self.wrt_matches = None
        self._approx = True

    @property
    def wrt_patterns(self):
        """
        Return the wrt patterns.

        Returns
        -------
        list of tuple or None
            Patterns used to match wrt variables.
        """
        return self._wrt_patterns

    @wrt_patterns.setter
    def wrt_patterns(self, patterns):
        """
        Set the wrt patterns.

        Parameters
        ----------
        patterns : list of str or None
            Patterns used to match wrt variables.
        """
        if isinstance(patterns, str):
            self._wrt_patterns = (patterns,)
        elif patterns is None:
            self.wrt_patterns = ()
        else:
            newpats = []
            for pattern in patterns:
                if isinstance(pattern, str):
                    newpats.append(pattern)
                else:
                    raise RuntimeError("Patterns in wrt_patterns must be strings, but found "
                                       f"{pattern} instead.")
            self._wrt_patterns = tuple(newpats)

    def _update_wrt_matches(self, system):
        """
        Determine the list of wrt variables that match the wildcard(s) given in declare_coloring.

        Parameters
        ----------
        system : System
            System being colored.

        Returns
        -------
        set of str or None
            Matched absolute wrt variable names or None if all wrt variables match.
        """
        if '*' in self._wrt_patterns:
            self.wrt_matches = None  # None means match everything
            return

        self.wrt_matches = set(rel_name2abs_name(system, n) for n in
                               pattern_filter(self.wrt_patterns, system._promoted_wrt_iter()))

        # error if nothing matched
        if not self.wrt_matches:
            raise ValueError("{}: Invalid 'wrt' variable(s) specified for colored approx partial "
                             "options: {}.".format(self.msginfo, self.wrt_patterns))

        return self.wrt_matches

    def reset_coloring(self):
        """
        Reset coloring and fname metadata.
        """
        super().reset_coloring()
        if not self.per_instance:
            _CLASS_COLORINGS[self.get_coloring_fname()] = None

    def update(self, dct):
        """
        Update the metadata.

        Parameters
        ----------
        dct : dict
            Dictionary of metadata.
        """
        for name, val in dct.items():
            if name in self._meta_names:
                setattr(self, name, val)


class Coloring(object):
    """
    Container for all information relevant to a coloring.

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
    _abs2prom : {'input': dict, 'output': dict}
        Dictionary mapping absolute names to promoted names.
    """

    def __init__(self, sparsity, row_vars=None, row_var_sizes=None, col_vars=None,
                 col_var_sizes=None):
        """
        Initialize data structures.
        """
        self._shape = sparsity.shape

        if isinstance(sparsity, np.ndarray):
            self._nzrows, self._nzcols = np.nonzero(sparsity)
            self._pct_nonzero = np.count_nonzero(sparsity) / (self._shape[0] * self._shape[1]) * 100
        else:  # sparse
            coo = sparsity.tocoo()
            self._nzrows = coo.row
            self._nzcols = coo.col
            self._pct_nonzero = coo.row.size / (self._shape[0] * self._shape[1]) * 100

        self._row_vars = row_vars
        self._row_var_sizes = row_var_sizes
        self._col_vars = col_vars
        self._col_var_sizes = col_var_sizes

        self._fwd = None
        self._rev = None

        self._meta = {
            'version': _COLORING_VERSION,
            'source': '',
        }

        self._names_array = {'fwd': None, 'rev': None}
        self._local_array = {'fwd': None, 'rev': None}

        self._abs2prom = None

    def get_renamed_copy(self, row_translate, col_translate):
        """
        Return a new Coloring object with the variables renamed.

        Parameters
        ----------
        row_translate : dict
            Dictionary mapping old row names to new row names.
        col_translate : dict
            Dictionary mapping old column names to new column names.

        Returns
        -------
        Coloring
            New Coloring object with the variables renamed.
        """
        row_vars = [row_translate[v] for v in self._row_vars]
        col_vars = [col_translate[v] for v in self._col_vars]
        c = Coloring(self.sparsity, row_vars, self._row_var_sizes, col_vars, self._col_var_sizes)
        c._fwd = self._fwd
        c._rev = self._rev
        c._meta = self._meta.copy()
        c._names_array = self._names_array
        c._local_array = self._local_array
        c._abs2prom = self._abs2prom

        return c

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

        yield from colors

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

    def colored_jac_iter(self, compressed_j, direction, trans=None):
        """
        Yield nonzero parts of columns (fwd) or rows (rev) of a colored jacobian.

        Parameters
        ----------
        compressed_j : ndarray
            The compressed jacobian.
        direction : str
            Derivative computation direction ('fwd' or 'rev').
        trans : ndarray
            Index array to translate from compressed jac in function context to openmdao jac.

        Yields
        ------
        ndarray
            Nonzero part of current jacobian column or row.
        ndarray
            Indices into full jacobian column or row where nonzero values should be placed.
        int
            Index into the full jacobian of the current column or row.
        """
        if direction == 'fwd':
            for i, (nzs, nzparts) in enumerate(self.color_nonzero_iter(direction)):
                for jac_icol, nzpart in zip(nzs, nzparts):
                    if trans is not None:
                        jac_icol = trans[jac_icol]
                    yield compressed_j[nzpart, i], nzpart, jac_icol
        else:  # rev
            for i, (nzs, nzparts) in enumerate(self.color_nonzero_iter(direction)):
                for jac_irow, nzpart in zip(nzs, nzparts):
                    yield compressed_j[i, nzpart], nzpart, jac_irow

    def expand_jac(self, compressed_j, direction):
        """
        Expand the given compressed jacobian into a full jacobian.

        Parameters
        ----------
        compressed_j : ndarray
            The compressed jacobian.
        direction : str
            Derivative computation direction ('fwd' or 'rev').

        Returns
        -------
        ndarray
            The full jacobian.
        """
        if direction == 'fwd':
            J = np.zeros(self._shape)
            for col, nzpart, icol in self.colored_jac_iter(compressed_j, direction):
                J[nzpart, icol] = col
            return J
        else:  # rev
            J = np.zeros(self._shape)
            for row, nzpart, irow in self.colored_jac_iter(compressed_j, direction):
                J[irow, nzpart] = row
            return J

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
                fwd_solves = len(fwd_lists)

            if rev_lists:
                rev_solves = len(rev_lists)

            if tot_size <= 0:
                pct = 0.
            else:
                pct = ((tot_size - tot_solves) / tot_size * 100)

        if tot_size < 0:
            tot_size = '?'

        return tot_size, tot_solves, fwd_solves, rev_solves, pct

    def total_solves(self, fwd=True, rev=True):
        """
        Return total number of solves required based on the given coloring info.

        Parameters
        ----------
        fwd : bool
            If True, add fwd colors to total.
        rev : bool
            If True, add rev colors to total.

        Returns
        -------
        int
            Total number of solves required to compute the jacobian.
        """
        total = 0

        if fwd and self._fwd:
            total += len(self._fwd[0])
        if rev and self._rev:
            total += len(self._rev[0])

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
            bad = False
            try:
                coloring = pickle.load(f)
            except pickle.UnpicklingError:
                bad = True
            else:
                bad = not isinstance(coloring, Coloring)
            if bad:
                raise RuntimeError(f"File '{fname}' is not a valid coloring file.")

        if 'version' not in coloring._meta:
            # old format, have to update color groups
            if coloring._fwd:
                old = coloring._fwd[0]
                newgrps = [[c] for c in old[0]]
                newgrps.extend(old[1:])
                coloring._fwd = (newgrps, coloring._fwd[1])
            if coloring._rev:
                old = coloring._rev[0]
                newgrps = [[c] for c in old[0]]
                newgrps.extend(old[1:])
                coloring._rev = (newgrps, coloring._rev[1])

        if 'timestamp' not in coloring._meta:
            file_mtime = pathlib.Path(fname).stat().st_mtime
            coloring._meta['timestamp'] = datetime.datetime.\
                fromtimestamp(file_mtime).strftime("%Y-%m-%d %H:%M:%S")

        if 'source' not in coloring._meta:
            coloring._meta['source'] = pathlib.Path(fname).absolute()

        return coloring

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
                try:
                    os.makedirs(color_dir)
                except FileExistsError:  # multiple systems could attempt this at the same time
                    pass
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError("Can't save coloring.  Expected a string for fname but got a %s" %
                            type(fname).__name__)

    def _check_config_total(self, driver, model):
        """
        Check the config of this total Coloring vs. the existing driver config.

        Parameters
        ----------
        driver : Driver
            Current driver object.
        model : Group
            Current model object.
        """
        ofs = model._active_responses(driver._get_ordered_nl_responses(), driver._responses)
        of_sizes = [m['size'] for m in ofs.values()]

        wrts = model._active_desvars(driver._designvars.keys(), driver._designvars)
        wrt_sizes = [m['size'] for m in wrts.values()]

        self._config_check_msgs(ofs, of_sizes, wrts, wrt_sizes, driver)

    def _check_config_partial(self, system):
        """
        Check the config of this partial (or semi-total) Coloring vs. the existing model config.

        Parameters
        ----------
        system : System
            System being colored.
        """
        # check the contents (vars and sizes) of the input and output vectors of system
        info = Partial_ColoringMeta(wrt_patterns=self._meta.get('wrt_patterns', ('*',)))
        info._update_wrt_matches(system)
        if system.pathname:
            # for partial and semi-total derivs, convert to promoted names
            ordered_of_info = system._jac_var_info_abs2prom(system._jac_of_iter())
            ordered_wrt_info = \
                system._jac_var_info_abs2prom(system._jac_wrt_iter(info.wrt_matches))
        else:
            ordered_of_info = list(system._jac_of_iter())
            ordered_wrt_info = list(system._jac_wrt_iter(info.wrt_matches))

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

        of_names = list(of_names)
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

        wrt_names = list(wrt_names)
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
            f", pct nonzero: {self._pct_nonzero:.2f}, tol: {self._meta.get('good_tol')})"
        )

    def summary(self, out_stream=_DEFAULT_OUT_STREAM):
        """
        Print a summary of this coloring.

        Parameters
        ----------
        out_stream : file-like or _DEFAULT_OUT_STREAM
            The destination stream to which the text representation of coloring is to be written.
        """
        nrows = self._shape[0] if self._shape else -1
        ncols = self._shape[1] if self._shape else -1

        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        print(f"\nJacobian shape: ({nrows}, {ncols})  ({self._pct_nonzero:.2f}% nonzero)",
              file=out_stream)
        if self._fwd is None and self._rev is None:
            tot_size = min(nrows, ncols)
            if tot_size < 0:
                tot_size = '?'
            print(f"Simultaneous derivatives can't improve on the total number of solves "
                  f"required ({tot_size}) for this configuration", file=out_stream)
        else:
            tot_size, tot_colors, fwd_solves, rev_solves, pct = self._solves_info()
            print(f"FWD solves: {fwd_solves}   REV solves: {rev_solves}", file=out_stream)
            print(f"Total colors vs. total size: {tot_colors} vs {tot_size}  "
                  f"({pct:.2f}% improvement)",
                  file=out_stream)

        meta = self._meta
        print('', file=out_stream)
        good_tol = meta.get('good_tol')
        if good_tol is not None:
            print("Sparsity computed using tolerance: %g" % meta['good_tol'], file=out_stream)
            if meta['n_tested'] > 1:
                print("Most common number of nonzero entries (%d of %d) repeated %d times out "
                      "of %d tolerances tested.\n" % (meta['J_size'] - meta['zero_entries'],
                                                      meta['J_size'],
                                                      meta['nz_matches'], meta['n_tested']),
                      file=out_stream)

        sparsity_time = meta.get('sparsity_time', None)
        if sparsity_time is not None:
            print(f"Time to compute sparsity: {sparsity_time:8.4f} sec", file=out_stream)

        coloring_time = meta.get('coloring_time', None)
        if coloring_time is not None:
            print(f"Time to compute coloring: {coloring_time:8.4f} sec", file=out_stream)

        coloring_mem = meta.get('coloring_memory', None)
        if coloring_mem is not None:
            print(f"Memory to compute coloring: {coloring_mem:8.4f} MB", file=out_stream)

        coloring_timestamp = meta.get('timestamp', None)
        if coloring_timestamp is not None:
            print(f"Coloring created on: {coloring_timestamp}", file=out_stream)

    def display_txt(self, out_stream=_DEFAULT_OUT_STREAM, html=False, summary=True,
                    use_prom_names=True):
        """
        Print the structure of a boolean array with coloring info for each nonzero value.

        Forward mode colored nonzeros are denoted by 'f', reverse mode nonzeros by 'r',
        overlapping nonzeros by 'O' and uncolored nonzeros by 'x'.  Zeros are denoted by '.'.
        Note that x's and O's should never appear unless there is a bug in the coloring
        algorithm.

        If names and sizes of row and column vars are known, print the name of the row var
        alongside each row and print the names of the column vars, aligned with each column,
        at the bottom.

        Parameters
        ----------
        out_stream : file-like or _DEFAULT_OUT_STREAM
            The destination stream to which the text representation of coloring is to be written.
        html : bool
            If True, the output will be formatted as HTML. If False, the resulting output will
            be plain text.
        summary : bool
            If True, include the coloring summary.
        use_prom_names : bool
            If True, display promoted names rather than absolute path names for variables.
        """
        if out_stream == _DEFAULT_OUT_STREAM:
            out_stream = sys.stdout

        source_name = self._meta['source']

        shape = self._shape
        nrows, ncols = shape

        if html:
            out_stream.write(f'<html>\n'
                             f'<head>\n'
                             f'<title>total coloring report: {source_name}</title>\n'
                             f'</head>\n'
                             f'<body>\n'
                             f'Total Coloring Report<br>\n'
                             f'{source_name}\n'
                             f'<hr style="width:100%;text-align:left;margin-left:0">\n'
                             f'<pre>\n')

        # array of chars the same size as dense jacobian
        charr = np.full(shape, '.', dtype=str)

        # mark all nonzero entries as 'x' initially, so the 'x' will be left
        # if not covered with an 'f' or an 'r'
        charr[self._nzrows, self._nzcols] = 'x'

        if self._fwd:
            full_rows = np.arange(nrows, dtype=INT_DTYPE)
            col2row = self._fwd[1]
            for grp in self._fwd[0]:
                for c in grp:
                    rows = col2row[c]
                    if rows is None:
                        rows = full_rows
                    charr[rows, c] = 'f'

        has_overlap = False
        if self._rev:
            full_cols = np.arange(ncols, dtype=INT_DTYPE)
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

        if (self._row_vars is None or self._row_var_sizes is None or
                self._col_vars is None or self._col_var_sizes is None):
            # we don't have var name/size info, so just show the unadorned matrix
            for r in range(nrows):
                for c in range(ncols):
                    print(charr[r, c], end='', file=out_stream)
                print(' %d' % r, file=out_stream)
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
                            print(charr[r, c], end='', file=out_stream)
                        colstart = colend
                    if use_prom_names and self._abs2prom:
                        row_var_name = self._get_prom_name(rv)
                    else:
                        row_var_name = rv
                    # include row variable with row
                    print(' %d  %s' % (r, row_var_name), file=out_stream)
                rowstart = rowend

            # now print the column vars below the matrix, with each one spaced over to line up
            # with the appropriate starting column of the matrix ('|' marks the start of each var)
            start = 0

            for name, size in zip(self._col_vars, self._col_var_sizes):
                tab = ' ' * start
                if use_prom_names and self._abs2prom:
                    col_var_name = self._get_prom_name(name)
                else:
                    col_var_name = name
                print('%s|%s' % (tab, col_var_name), file=out_stream)
                start += size

        if html:
            print('</pre>\n', file=out_stream)

        if summary:
            if html:
                print('<pre>\n', file=out_stream)
            self.summary(out_stream=out_stream)
            if html:
                print('</pre>\n', file=out_stream)

        if html:
            out_stream.write(f'</body>\n'
                             f'</html>')

        if has_overlap:
            raise RuntimeError("Internal coloring bug: jacobian has entries where fwd and rev "
                               "colorings overlap!")

    def display(self, show=True, fname=_default_coloring_imagefile):
        """
        Display a plot of the sparsity pattern, showing grouping by color.

        Parameters
        ----------
        show : bool
            If True, show the plot. Otherwise, just save the plot in a file. Default is True.
        fname : str
            Path to the location where the plot file should be saved.
        """
        issue_warning('display is deprecated. Use display_bokeh for rich html displays of coloring'
                      'or display_txt for a text-based display.', category=OMDeprecationWarning)

        if mpl is None:
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
            entry_xnames = np.zeros(ncols, dtype=INT_DTYPE)
            entry_ynames = np.zeros(nrows, dtype=INT_DTYPE)
            entry_xcolors = np.zeros(ncols, dtype=INT_DTYPE)
            entry_ycolors = np.zeros(nrows, dtype=INT_DTYPE)

            # pick two colors for our checkerboard pattern
            if Version(mpl.__version__) < Version("3.6"):
                sjcolors = [cm.get_cmap('Greys')(0.3), cm.get_cmap('Greys')(0.4)]
            else:
                sjcolors = [mpl.colormaps['Greys'](0.3), mpl.colormaps['Greys'](0.4)]

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

        if self._fwd:
            # winter is a blue/green color map
            if Version(mpl.__version__) < Version("3.6"):
                cmap = cm.get_cmap('winter')
            else:
                cmap = mpl.colormaps['winter']

            icol = 1
            full_rows = np.arange(nrows, dtype=INT_DTYPE)
            col2row = self._fwd[1]
            for i, grp in enumerate(self._fwd[0]):
                for c in grp:
                    rows = col2row[c]
                    if rows is None:
                        rows = full_rows
                    idx = icol / fwd_solves
                    for r in rows:
                        J[r, c][:] = cmap(idx)[:3]
                    entry_xcolors[c] = icol
                icol += 1

        if self._rev:
            # autumn_r is a red/yellow color map
            if Version(mpl.__version__) < Version("3.6"):
                cmap = cm.get_cmap('autumn_r')
            else:
                cmap = mpl.colormaps['autumn_r']

            icol = 1
            full_cols = np.arange(ncols, dtype=INT_DTYPE)
            row2col = self._rev[1]
            for i, grp in enumerate(self._rev[0]):
                for r in grp:
                    cols = row2col[r]
                    if cols is None:
                        cols = full_cols
                    idx = icol / rev_solves
                    for c in cols:
                        J[r, c][:] = cmap(idx)[:3]
                    entry_ycolors[r] = icol
                icol += 1

        typ = self._meta['type'].upper()[0] + self._meta['type'][1:]

        ax.set_title("%s Jacobian Coloring (%d x %d)\n%d fwd colors, %d rev colors "
                     "(%.1f%% improvement)" %
                     (typ, self._shape[0], self._shape[1], fwd_solves, rev_solves, pct))

        pyplot.imshow(J, interpolation="none")
        fig.tight_layout()

        if show:
            pyplot.show()
        else:
            pyplot.savefig(fname)

        pyplot.close(fig)

    def display_bokeh(source, output_file='total_coloring.html', show=False, max_colors=200,
                      use_prom_names=True):
        """
        Display a plot of the sparsity pattern, showing grouping by color.

        Parameters
        ----------
        source : str or Coloring Driver
            The source for the coloring information, which can either be a string of the filepath
            to the coloring data file, an instance of Coloring, or the Driver containing the
            coloring information.
        output_file : str or None
            The name of the output html file in which the display is written. If None, the resulting
            plots will not be saved.
        show : bool
            If True, a browswer will be opened to display the generated file.
        max_colors : int
            Bokeh supports at most 256 colors in a colormap. This function reduces that number
            to some default length, otherwise both forward and reverse displays may share shades
            very near white and be difficult to distinguish. Once the number of forward or reverse
            solves exceeds this threshold, the color pattern restarts.
        use_prom_names : bool
            If True, display promoted names rather than absolute path names for variables.
        """
        if bokeh_resources is None:
            print("bokeh is not installed so this coloring viewer is not available. The ascii "
                  "based coloring viewer can be accessed by calling display_txt() on the Coloring "
                  "object or by using 'openmdao view_coloring --textview <your_coloring_file>' "
                  "from the command line.")
            return

        if isinstance(source, str):
            coloring = Coloring.load(source)
            source_name = pathlib.Path(source).absolute()
        elif isinstance(source, Coloring):
            coloring = source
            source_name = coloring._meta['source']
        elif hasattr(source, '_coloring_info'):
            coloring = source._coloring_info.coloring
            source_name = source._problem()._metadata['pathname']
        else:
            raise ValueError(f'display_bokeh was expecting the source to be a valid coloring file '
                             f'or an instance of driver but instead got f{type(source)}')

        if coloring is None:
            # Save and show
            summary_div = PreText(text='No total derivative coloring found.',
                                  styles={'font-size': '12pt'})

            fig = PreText(text='')
        else:
            nrows, ncols = coloring._shape
            aspect_ratio = ncols / nrows

            tot_size, tot_colors, fwd_solves, rev_solves, pct = coloring._solves_info()

            data = {}

            # The row and column indices of the individual jacobian elements
            data['col_idx'] = np.tile(np.arange(ncols, dtype=int), nrows)
            data['row_idx'] = np.repeat(np.arange(nrows, dtype=int), ncols)

            have_vars = None not in (coloring._col_vars, coloring._row_vars,
                                     coloring._col_var_sizes, coloring._row_var_sizes)

            # The indices of the responses and desvars obtained by binning the row/col indices
            if have_vars:
                desvar_idx_bins = [] if coloring._col_var_sizes is None else \
                    np.cumsum(coloring._col_var_sizes)
                response_idx_bins = [] if coloring._row_var_sizes is None else \
                    np.cumsum(coloring._row_var_sizes)

                response_idx = np.digitize(data['row_idx'], response_idx_bins)
                desvar_idx = np.digitize(data['col_idx'], desvar_idx_bins)

                data['pattern'] = np.full(nrows * ncols, '', dtype=str)
                data['pattern'][...] = np.asarray(desvar_idx % 2 + response_idx % 2, dtype=str)
            data['fwd_color_idx'] = np.full(nrows * ncols, '', dtype=object)
            data['rev_color_idx'] = np.full(nrows * ncols, '', dtype=object)

            # Add the color group information to the data source
            fwd_map = {}
            if coloring._fwd is not None:
                for idx_fwd, (_cols, _nz_rows) in enumerate(coloring.color_nonzero_iter('fwd')):
                    for _row_idx, _col_idx in zip(_nz_rows, _cols):
                        fwd_map.update({(i, _col_idx): idx_fwd for i in _row_idx})

            rev_map = {}
            if coloring._rev is not None:
                for idx_rev, (_rows, _nz_cols) in enumerate(coloring.color_nonzero_iter('rev')):
                    for _row_idx, _col_idx in zip(_rows, _nz_cols):
                        rev_map.update({(_row_idx, j): idx_rev for j in _col_idx})

            for i in range(nrows * ncols):
                r = data['row_idx'][i]
                c = data['col_idx'][i]
                if (r, c) in fwd_map:
                    data['fwd_color_idx'][i] = str(fwd_map[r, c])
                if (r, c) in rev_map:
                    data['rev_color_idx'][i] = str(rev_map[r, c])

            data_source = ColumnDataSource(data)

            HEIGHT = 600

            fig = figure(toolbar_location="above",
                         x_range=(-1, ncols + 1), y_range=(nrows + 1, -1),
                         x_axis_location="above", width=int(HEIGHT * aspect_ratio), height=HEIGHT,
                         sizing_mode='scale_both')

            fig.xaxis.visible = False
            fig.yaxis.visible = False

            fig.xgrid.grid_line_color = None
            fig.ygrid.grid_line_color = None

            # Plot the background pattern
            if have_vars:
                gray_cm = gray(12)
                background_mapper = CategoricalColorMapper(factors=[str(i) for i in range(3)],
                                                           palette=gray_cm[-4:-1])

                fig.rect(x='col_idx', y='row_idx', width=1, height=1, source=data_source, alpha=0.5,
                         line_color=None, fill_color=transform('pattern', background_mapper))

            # Plot the fwd solve groups
            if fwd_solves > 0:
                fwd_colors = interp_palette(list(Blues256)[:max_colors], fwd_solves)
                fwd_mapper = CategoricalColorMapper(factors=[str(i) for i in range(fwd_solves)],
                                                    palette=fwd_colors,
                                                    nan_color=(0, 0, 0, 0))
                fwd_rect = fig.rect(x='col_idx', y='row_idx', width=1, height=1,
                                    source=data_source, alpha=1.0, line_color=None,
                                    fill_color=transform('fwd_color_idx', fwd_mapper))

                fig.add_layout(fwd_rect.construct_color_bar(
                    major_label_text_font_size="7px",
                    label_standoff=6,
                    border_line_color=None,
                    padding=5,
                    title='forward solves',
                ), 'right')

            # Plot the rev solve groups
            if rev_solves > 0:
                rev_colors = interp_palette(list(Reds256)[:max_colors], rev_solves)
                rev_mapper = CategoricalColorMapper(factors=[str(i) for i in range(rev_solves)],
                                                    palette=rev_colors,
                                                    nan_color=(0, 0, 0, 0))

                rev_rect = fig.rect(x='col_idx', y='row_idx', width=1, height=1,
                                    source=data_source, alpha=1.0, line_color=None,
                                    fill_color=transform('rev_color_idx', rev_mapper))

                fig.add_layout(rev_rect.construct_color_bar(
                    major_label_text_font_size="7px",
                    label_standoff=6,
                    border_line_color=None,
                    padding=0,
                    title='reverse solves',
                ), 'right')

            # Add a tooltip on hover
            if have_vars:
                desvar_col_map = {desvar_name: set() for desvar_name in coloring._col_vars}
                for col_idx in range(ncols):
                    desvar_name = coloring._col_vars[np.digitize(col_idx, desvar_idx_bins)]
                    desvar_col_map[desvar_name].add(col_idx)

                if use_prom_names and coloring._abs2prom:
                    desvar_col_map = {coloring._get_prom_name(k): v
                                      for k, v in desvar_col_map.items()}

                resvar_col_map = {varname: set() for varname in coloring._row_vars}
                for row_idx in range(nrows):
                    resvar_name = coloring._row_vars[np.digitize(row_idx, response_idx_bins)]
                    resvar_col_map[resvar_name].add(row_idx)

                if use_prom_names and coloring._abs2prom:
                    resvar_col_map = {coloring._get_prom_name(k): v
                                      for k, v in resvar_col_map.items()}

                design_var_js = CustomJSHover(code="""
                for (var name in varnames_map) {
                    if (varnames_map[name].has(special_vars.snap_x)) {
                        return name;
                    }
                }
                return '';
                """, args=dict(varnames_map=desvar_col_map))

                response_var_js = CustomJSHover(code="""
                for (var name in varnames_map) {
                    if (varnames_map[name].has(special_vars.snap_y)) {
                        return name;
                    }
                }
                return '';
                """, args=dict(varnames_map=resvar_col_map))

                tooltips = [('Response', '$snap_y{0}'),  # {0} triggers the formatter
                            ('Design Var', '$snap_x{0}'),
                            ('Forward solve', '@fwd_color_idx'),
                            ('Reverse solve', '@rev_color_idx')]
                formatters = {'$snap_y': response_var_js,
                              '$snap_x': design_var_js}
            else:
                tooltips = [('Forward solve', '@fwd_color_idx'),
                            ('Reverse solve', '@rev_color_idx')]
                formatters = {}

            fig.add_tools(HoverTool(tooltips=tooltips, formatters=formatters))

            ss = io.StringIO()
            coloring.summary(out_stream=ss)
            summary_div = PreText(text=ss.getvalue(), styles={'font-size': '12pt'})

        header_div = Div(text=f'Total Coloring Report<br>{source_name}',
                         styles={'font-size': '16pt', 'font-style': 'bold'})

        report_layout = column(children=[header_div,
                                         fig,
                                         summary_div],
                               sizing_mode='scale_height')

        # Save and show
        bokeh.io.curdoc().theme = 'light_minimal'
        if output_file is not None:
            bokeh.io.save(report_layout, filename=output_file,
                          title=f'total coloring report for {source_name}',
                          resources=bokeh_resources.INLINE)

        if show:
            bokeh.io.show(report_layout)

    @property
    def sparsity(self):
        """
        Return the sparsity matrix as a COO sparse matrix.

        Returns
        -------
        coo_matrix
            The sparsity matrix.
        """
        return coo_matrix((np.ones(len(self._nzrows), dtype=np.uint8),
                           (self._nzrows, self._nzcols)), shape=self._shape)

    def get_dense_sparsity(self, dtype=np.uint8):
        """
        Return a dense array representing the full sparsity.

        Parameters
        ----------
        dtype : object
            Data type of returned numpy array.

        Returns
        -------
        ndarray
            Dense sparsity matrix.
        """
        J = np.zeros(self._shape, dtype=dtype)
        J[self._nzrows, self._nzcols] = dtype(1)
        return J

    def _jac2subjac_sparsity(self):
        """
        Given a boolean jacobian and variable names and sizes, compute subjac sparsity.

        Returns
        -------
        dict
            Nested dict of form sparsity[of][wrt] = (rows, cols, shape)
        """
        sparsity = {}
        row_start = row_end = 0

        for of, of_size in zip(self._row_vars, self._row_var_sizes):
            sparsity[of] = {}
            row_end += of_size
            rowbool = np.logical_and(self._nzrows >= row_start, self._nzrows < row_end)

            col_start = col_end = 0
            for wrt, wrt_size in zip(self._col_vars, self._col_var_sizes):
                col_end += wrt_size
                colbool = np.logical_and(self._nzcols >= col_start, self._nzcols < col_end)
                mask = np.logical_and(rowbool, colbool)

                # save sparsity structure as  (rows, cols, shape)
                sparsity[of][wrt] = (self._nzrows[mask] - row_start, self._nzcols[mask] - col_start,
                                     (of_size, wrt_size))

                col_start = col_end

            row_start = row_end

        return sparsity

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
            return self._jac2subjac_sparsity()

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
                nzrows = self._fwd[1]
                for color_group in self.color_iter('fwd'):
                    subJ[:, :] = False
                    for c in color_group:
                        J[nzrows[c], c] = True

                    # if any color in the group has nonzeros in our variable, add a solve
                    if np.any(subJ):
                        fwd_solves += 1

            if self._rev:
                for color_group in self.color_iter('rev'):
                    subJ[:, :] = False
                    J[color_group, :] = True

                    if np.any(subJ):
                        rev_solves += 1

        return fwd_solves, rev_solves

    def _local_indices(self, inds, mode):
        # this is currently only used when dumping debug info for coloring
        if self._names_array[mode] is None:
            if mode == 'fwd':
                col_info = zip(self._col_vars, self._col_var_sizes)
            else:
                col_info = zip(self._row_vars, self._row_var_sizes)

            names = []
            indices = []
            for name, size in col_info:
                names.append(np.repeat(name, size))
                indices.append(np.arange(size))

            self._names_array[mode] = np.concatenate(names)
            self._local_array[mode] = np.concatenate(indices)

        if isinstance(inds, list):
            var_name_and_sub_indices = \
                [(key, [x[1] for x in group]) for key, group in
                 groupby(zip(self._names_array[mode][inds], self._local_array[mode][inds]),
                         key=lambda x: x[0])]
        else:
            var_name_and_sub_indices = [(self._names_array[mode][inds],
                                         self._local_array[mode][inds])]

        return var_name_and_sub_indices

    def tangent_iter(self, direction, arr=None, trans=None):
        """
        Given a direction, return input (fwd) or output (rev) tangent arrays.

        Each array will contain multiple columns/rows of the identity matrix that share the
        same color.

        Parameters
        ----------
        direction : str
            Indicates which coloring subdict ('fwd' or 'rev') to use.
        arr : ndarray or None
            Storage for the current array value.
        trans : ndarray or None
            Index translation array.

        Yields
        ------
        ndarray
            tangent array for inputs (fwd) or outputs (rev)
        """
        if direction == 'fwd':
            size = self._shape[1]
            fwd = True
        else:
            size = self._shape[0]
            fwd = False

        if arr is None:
            arr = np.empty(size)
        elif size != arr.size:
            raise RuntimeError("Size of given storage array doesn't match shape of the coloring for"
                               f" the '{direction}' direction.")

        for nzs, nzparts in self.color_nonzero_iter(direction):
            if trans is not None:
                if fwd:
                    nzs = trans[nzs]
                else:
                    nzparts = trans[nzparts]
            arr[:] = 0
            arr[nzs] = 1
            yield arr, nzs, nzparts

    def tangent_matrix(self, direction, trans=None):
        """
        Return a tangent or cotangent matrix for use with jax.

        Parameters
        ----------
        direction : str
            Derivative computation direction ('fwd' or 'rev').
        trans : ndarray or None
            Index translation array.

        Returns
        -------
        ndarray
            The tangent or cotangent matrix.
        """
        if direction == 'fwd':
            shape = (self.total_solves(rev=False), self._shape[1])
            tangent = np.empty(shape)
            for i, (arr, _, _) in enumerate(self.tangent_iter(direction, trans=trans)):
                tangent[i, :] = arr
        else:  # rev
            shape = (self._shape[0], self.total_solves(fwd=False))
            tangent = np.empty(shape)
            for i, (arr, _, _) in enumerate(self.tangent_iter(direction, trans=trans)):
                tangent[:, i] = arr
            tangent = tangent.T

        return tangent

    def _get_prom_name(self, abs_name):
        """
        Get promoted name for specified variable.
        """
        abs2prom = self._abs2prom

        # if we don't have prom names, just return abs name
        if not abs2prom:
            return abs_name

        # if we can't find a prom name, just return abs name
        if abs_name in abs2prom['input']:
            return abs2prom['input'][abs_name]
        elif abs_name in abs2prom['output']:
            return abs2prom['output'][abs_name]
        else:
            return abs_name


def _order_by_ID(col_adj_matrix):
    """
    Return columns in order of incidence degree (ID).

    ID is the number of already colored neighbors (neighbors are dependent columns).

    The parameters given are assumed to correspond to a those of a column dependency matrix,
    i.e., (i, j) nonzero entries in the matrix indicate that column i is dependent on column j.

    Parameters
    ----------
    col_adj_matrix : csc matrix
        CSC column adjacency matrix.

    Yields
    ------
    int
        Column index.
    ndarray
        Boolean array that's True where the column matches nzcols.
    """
    ncols = col_adj_matrix.shape[1]
    colored_degrees = np.zeros(ncols, dtype=INT_DTYPE)
    colored_degrees[col_adj_matrix.indices] = 1  # make sure zero cols aren't considered

    for i in range(np.nonzero(colored_degrees)[0].size):
        col = colored_degrees.argmax()
        colnzrows = col_adj_matrix.getcol(col).indices
        colored_degrees[colnzrows] += 1
        colored_degrees[col] = -ncols  # ensure that this col will never have max degree again
        yield col, colnzrows


def _2col_adj_rows_cols(J):
    """
    Convert nonzero rows/cols of sparsity matrix to those of a column adjacency matrix.

    Parameters
    ----------
    J : coo_matrix
        Sparse matrix to be colored.

    Returns
    -------
    csc_matrix
        Sparse column adjacency matrix.
    """
    nrows, ncols = J.shape
    nzrows, nzcols = J.row, J.col

    adjrows = []
    adjcols = []

    csr = csr_matrix((np.ones(nzrows.size, dtype=bool), (nzrows, nzcols)), shape=J.shape)

    # mark col_matrix entries as True when nonzero row entries make them dependent
    for row in np.unique(nzrows):
        row_nzcols = csr.getrow(row).indices

        if row_nzcols.size > 0:
            for c in row_nzcols:
                adjrows.append(row_nzcols)
                adjcols.append(np.full(row_nzcols.size, c))

    if adjrows:
        adjrows = np.hstack(adjrows)
        adjcols = np.hstack(adjcols)
    else:
        adjrows = np.zeros(0, dtype=INT_DTYPE)
        adjcols = np.zeros(0, dtype=INT_DTYPE)

    return csc_matrix((np.ones(adjrows.size, dtype=bool), (adjrows, adjcols)), shape=(ncols, ncols))


def _Jc2col_matrix_direct(Jrows, Jcols, shape):
    """
    Convert a partitioned jacobian sparsity matrix to a column adjacency matrix.

    This creates the column adjacency matrix used for direct jacobian determination
    as described in Coleman, T.F., Verma, A. (1998) The efficient Computation of Sparse Jacobian
    Matrices Using Automatic Differentiation. SIAM Journal on Scientific Computing, 19(4),
    1210-1233.

    Parameters
    ----------
    Jrows : ndarray
        Nonzero rows of a partition of the matrix being colored.
    Jcols : ndarray
        Nonzero columns of a partition of the matrix being colored.
    shape : tuple
        Shape of the partition of the matrix being colored.

    Returns
    -------
    tuple
        (nzrows, nzcols, shape) of column adjacency matrix.
    """
    nrows, ncols = shape

    allnzr = []
    allnzc = []

    Jrow = np.zeros(ncols, dtype=bool)
    csr = csr_matrix((np.ones(Jrows.size, dtype=bool), (Jrows, Jcols)), shape=shape)

    # mark col_matrix[col1, col2] as True when Jpart[row, col1] is True OR Jpart[row, col2] is True
    for row in np.unique(Jrows):
        nzr = []
        nzc = []
        row_nzcols = csr.getrow(row).indices

        if row_nzcols.size == 1:
            # if there's only 1 nonzero column in a row, include it
            nzr.append(row_nzcols[0])
            nzc.append(row_nzcols[0])
        else:
            Jrow[:] = False
            Jrow[row_nzcols] = True
            for col1, col2 in combinations(row_nzcols, 2):
                if Jrow[col1] or Jrow[col2]:
                    nzr.append(col1)
                    nzc.append(col2)
        if nzr:
            allnzr.append(nzr)
            allnzc.append(nzc)

    csr = Jrow = None  # free up memory

    if allnzr:
        # matrix is symmetric, so duplicate
        rows = np.hstack(allnzr + allnzc)
        cols = np.hstack(allnzc + allnzr)
    else:
        rows = np.zeros(0, dtype=INT_DTYPE)
        cols = np.zeros(0, dtype=INT_DTYPE)

    allnzr = allnzc = None

    return csc_matrix((np.ones(rows.size, dtype=bool), (rows, cols)), shape=(ncols, ncols))


def _get_full_disjoint_cols(J):
    """
    Find sets of disjoint columns in J and their corresponding rows using a col adjacency matrix.

    Parameters
    ----------
    J : coo_matrix
        Sparse matrix to be colored.

    Returns
    -------
    list
        List of lists of disjoint columns
    """
    return _get_full_disjoint_col_matrix_cols(_2col_adj_rows_cols(J))


def _get_full_disjoint_col_matrix_cols(col_adj_matrix):
    """
    Find sets of disjoint columns in a column intersection matrix.

    Parameters
    ----------
    col_adj_matrix : csc_matrix
        Sparse column adjacency matrix.

    Returns
    -------
    list
        List of lists of disjoint columns.
    """
    color_groups = []
    _, ncols = col_adj_matrix.shape

    # -1 indicates that a column has not been colored
    colors = np.full(ncols, -1, dtype=INT_DTYPE)

    for icol, colnzrows in _order_by_ID(col_adj_matrix):
        neighbor_colors = colors[colnzrows]
        for color, grp in enumerate(color_groups):
            if color not in neighbor_colors:
                grp.append(icol)
                colors[icol] = color
                break
        else:
            colors[icol] = len(color_groups)
            color_groups.append([icol])

    return color_groups


def _color_partition(Jprows, Jpcols, shape):
    """
    Compute a single directional fwd coloring using partition Jpart.

    This routine is used to compute a fwd coloring on Jc and a rev coloring on Jr.T.

    Parameters
    ----------
    Jprows : ndarray
        Nonzero rows of a partition of the matrix being colored.
    Jpcols : ndarray
        Nonzero columns of a partition of the matrix being colored.
    shape : tuple
        Shape of a partition of the matrix being colored.

    Returns
    -------
    list
        List of color groups.  First group is uncolored.
    list
        List of nonzero rows for each column.
    """
    _, ncols = shape

    col_adj_matrix = _Jc2col_matrix_direct(Jprows, Jpcols, shape)
    col_groups = _get_full_disjoint_col_matrix_cols(col_adj_matrix)

    col_adj_matrix = None

    for i, group in enumerate(col_groups):
        col_groups[i] = sorted(group)

    csc = csc_matrix((np.ones(Jprows.size), (Jprows, Jpcols)), shape=shape)
    col2row = [None] * ncols
    for col in np.unique(Jpcols):
        col2row[col] = csc.getcol(col).indices

    return [col_groups, col2row]


def MNCO_bidir(J):
    """
    Compute bidirectional coloring using Minimum Nonzero Count Order (MNCO).

    Based on the algorithm found in Coleman, T.F., Verma, A. (1998) The efficient Computation
    of Sparse Jacobian Matrices Using Automatic Differentiation. SIAM Journal on Scientific
    Computing, 19(4), 1210-1233.

    Parameters
    ----------
    J : coo_matrix
        Jacobian sparsity matrix (boolean).

    Returns
    -------
    Coloring
        See docstring for Coloring class.
    """
    nzrows, nzcols = J.row, J.col
    nrows, ncols = J.shape

    coloring = Coloring(sparsity=J)

    M_col_nonzeros = np.zeros(ncols, dtype=INT_DTYPE)
    M_row_nonzeros = np.zeros(nrows, dtype=INT_DTYPE)

    sparse = csc_matrix((np.ones(nzrows.size, dtype=bool), (nzrows, nzcols)), shape=J.shape)

    for c in range(ncols):
        M_col_nonzeros[c] = sparse.getcol(c).indices.size
    sparse = sparse.tocsr()
    for r in range(nrows):
        M_row_nonzeros[r] = sparse.getrow(r).indices.size

    sparse = None

    M_rows, M_cols = nzrows, nzcols

    Jf_rows = [None] * nrows
    Jr_cols = [None] * ncols

    row_i = col_i = 0

    # partition J into Jf and Jr
    # Jf is colored by column and those columns will be solved in fwd mode
    # Jr is colored by row and those rows will be solved in reverse mode
    # We build Jf from bottom up (by row) and Jr from right to left (by column).

    # get index of row with fewest nonzeros and col with fewest nonzeros
    r = M_row_nonzeros.argmin()
    c = M_col_nonzeros.argmin()

    # get number of nonzeros in the selected row and column
    nnz_r = M_row_nonzeros[r]
    nnz_c = M_col_nonzeros[c]

    Jf_nz_max = 0   # max row nonzeros in Jf
    Jr_nz_max = 0   # max col nonzeros in Jr

    while M_rows.size > 0:
        # what the algorithm is doing is basically minimizing the total of the max number of nonzero
        # columns in Jf + the max number of nonzero rows in Jr, so it's basically minimizing
        # the upper bound of the number of colors that will be needed.

        # we differ from the algorithm in the paper here slightly because we add ncols and nrows to
        # different sides of the inequality in order to prevent bad colorings when we have
        # matrices that have many more rows than columns or many more columns than rows.
        if ncols + Jr_nz_max + max(Jf_nz_max, nnz_r) < (nrows + Jf_nz_max + max(Jr_nz_max, nnz_c)):
            Jf_rows[r] = M_cols[M_rows == r]
            Jf_nz_max = max(nnz_r, Jf_nz_max)

            M_row_nonzeros[r] = ncols + 1  # make sure we don't pick this one again
            M_col_nonzeros[Jf_rows[r]] -= 1

            # remove row r
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

            # remove column c
            keep = M_cols != c
            r = M_row_nonzeros.argmin()
            c = M_col_nonzeros.argmin()
            nnz_c = M_col_nonzeros[c]

            col_i += 1

        M_rows = M_rows[keep]
        M_cols = M_cols[keep]

    M_row_nonzeros = M_col_nonzeros = None

    nnz_Jf = nnz_Jr = 0

    if row_i > 0:
        Jfr = []
        Jfc = []
        # build Jf and do fwd coloring on it
        for i, cols in enumerate(Jf_rows):
            if cols is not None:
                Jfc.append(cols)
                Jfr.append(np.full(cols.size, i, dtype=INT_DTYPE))
                nnz_Jf += len(cols)

        Jf_rows = None
        Jfr = np.hstack(Jfr)
        Jfc = np.hstack(Jfc)
        coloring._fwd = _color_partition(Jfr, Jfc, J.shape)
        Jfr = Jfc = None

    if col_i > 0:
        Jrr = []
        Jrc = []
        # build Jr and do rev coloring
        for i, rows in enumerate(Jr_cols):
            if rows is not None:
                Jrr.append(rows)
                Jrc.append(np.full(rows.size, i, dtype=INT_DTYPE))
                nnz_Jr += len(rows)

        Jr_cols = None
        Jrr = np.hstack(Jrr)
        Jrc = np.hstack(Jrc)
        coloring._rev = _color_partition(Jrc, Jrr, J.T.shape)

    if nzrows.size != nnz_Jf + nnz_Jr:
        raise RuntimeError("Nonzero mismatch for J vs. Jf and Jr")

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
                nz = np.count_nonzero(arr > itol)
                if nzeros and nzeros[-1][1] == nz:
                    nzeros[-1][0].append(itol)
                else:
                    nzeros.append(([itol], nz))
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
def _compute_total_coloring_context(problem):
    """
    Context manager for computing total jac sparsity for simultaneous coloring.

    Parameters
    ----------
    problem : Problem
        The problem where coloring will be done.
    """
    problem._metadata['coloring_randgen'] = np.random.default_rng(41)  # set seed for consistency
    problem._computing_coloring = True

    try:
        yield
    finally:
        problem._metadata['coloring_randgen'] = None
        problem._computing_coloring = False


def _get_total_jac_sparsity(prob, num_full_jacs=_DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                            tol=_DEF_COMP_SPARSITY_ARGS['tol'],
                            orders=_DEF_COMP_SPARSITY_ARGS['orders'], setup=False, run_model=False,
                            of=None, wrt=None, driver=None):
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
    driver : Driver, None, or False
        The driver that will be used to compute the total jacobian.  If None, the driver
        from the problem will be used.  If False, compute_totals will be called directly
        on the problem.

    Returns
    -------
    ndarray
        A boolean composite of 'num_full_jacs' total jacobians.
    """
    # clear out any old simul coloring info
    if driver is None:
        driver = prob.driver
        driver._con_subjacs = {}

    if not prob._computing_coloring:
        if setup:
            prob.setup(mode=prob._orig_mode)

        if run_model:
            prob.run_model(reset_iter_counts=False)

    if of is None or wrt is None:
        if driver:
            wrt = driver_wrt = list(_src_name_iter(driver._designvars))
            of = driver_of = driver._get_ordered_nl_responses()

        if not driver or not driver_wrt or not driver_of:
            raise RuntimeError("When computing total jacobian sparsity, either 'of' and 'wrt' "
                               "must be provided or design_vars/constraints/objective must be "
                               "added to the driver.")

    use_driver = driver and driver._coloring_info.use_scaling

    with _compute_total_coloring_context(prob):
        start_time = time.perf_counter()
        fullJ = None
        for i in range(num_full_jacs):
            if use_driver:
                Jabs = driver._compute_totals(of=of, wrt=wrt, return_format='array')
            else:
                Jabs = prob.compute_totals(of=of, wrt=wrt, return_format='array',
                                           coloring_info=False)
            if fullJ is None:
                fullJ = np.abs(Jabs)
            else:
                fullJ += np.abs(Jabs)

        Jabs = None
        elapsed = time.perf_counter() - start_time

    fullJ *= (1.0 / np.max(fullJ))

    spmeta = _tol_sweep(fullJ, tol, orders)
    spmeta['num_full_jacs'] = num_full_jacs
    spmeta['sparsity_time'] = elapsed
    spmeta['type'] = 'total'

    print(f"Full total jacobian for problem '{prob._metadata['pathname']}' was computed "
          f"{num_full_jacs} times, taking {elapsed} seconds.")
    print("Total jacobian shape:", fullJ.shape, "\n")

    nzrows, nzcols = np.nonzero(fullJ > spmeta['good_tol'])
    shape = fullJ.shape
    fullJ = None

    return coo_matrix((np.ones(nzrows.size, dtype=bool), (nzrows, nzcols)), shape=shape), spmeta


def _compute_coloring(J, mode):
    """
    Compute a good coloring in a specified dominant direction.

    Parameters
    ----------
    J : ndarray or coo_matrix
        The sparsity matrix.
    mode : str
        The direction for solving for total derivatives.  Must be 'fwd', 'rev' or 'auto'.
        If 'auto', use bidirectional coloring.

    Returns
    -------
    Coloring
        See Coloring class docstring.
    """
    start_time = time.perf_counter()
    try:
        start_mem = mem_usage()
    except RuntimeError:
        start_mem = None

    if mode == 'auto':  # use bidirectional coloring
        if isinstance(J, np.ndarray):
            nzrows, nzcols = np.nonzero(J)
            J = coo_matrix((np.ones(nzrows.size, dtype=bool), (nzrows, nzcols)), shape=J.shape)

        coloring = MNCO_bidir(J)
        fallback = _compute_coloring(J, 'fwd')
        if coloring.total_solves() >= fallback.total_solves():
            coloring = fallback
            coloring._meta['fallback'] = True
        fallback = _compute_coloring(J, 'rev')
        if coloring.total_solves() > fallback.total_solves():
            coloring = fallback
            coloring._meta['fallback'] = True
        fallback = None

        # record the total time and memory usage for bidir, fwd, and rev
        coloring._meta['coloring_time'] = time.perf_counter() - start_time
        if start_mem is not None:
            coloring._meta['coloring_memory'] = mem_usage() - start_mem

        return coloring

    rev = mode == 'rev'

    coloring = Coloring(sparsity=J)

    if rev:
        J = J.T

    nrows, ncols = J.shape

    if isinstance(J, np.ndarray):
        nzrows, nzcols = np.nonzero(J)
        J = coo_matrix((np.ones(nzrows.size), (nzrows, nzcols)), shape=J.shape)

    nzrows, nzcols = J.row, J.col
    col_groups = _get_full_disjoint_cols(J)

    col2rows = [None] * ncols  # will contain list of nonzero rows for each column

    for r, c in zip(nzrows, nzcols):
        if col2rows[c] is None:
            col2rows[c] = [r]
        else:
            col2rows[c].append(r)

    for c, rows in enumerate(col2rows):
        if rows is not None:
            col2rows[c] = sorted(rows)

    if rev:
        coloring._rev = (col_groups, col2rows)
    else:  # fwd
        coloring._fwd = (col_groups, col2rows)

    coloring._meta['coloring_time'] = time.perf_counter() - start_time
    if start_mem is not None:
        coloring._meta['coloring_memory'] = mem_usage() - start_mem

    return coloring


def compute_total_coloring(problem, mode=None, of=None, wrt=None,
                           num_full_jacs=_DEF_COMP_SPARSITY_ARGS['num_full_jacs'],
                           tol=_DEF_COMP_SPARSITY_ARGS['tol'],
                           orders=_DEF_COMP_SPARSITY_ARGS['orders'],
                           setup=False, run_model=False, fname=None,
                           driver=None):
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
    driver : <Driver>, None, or False
        The driver associated with the coloring.  If None, use problem.driver.  If False, no
        driver will be used.

    Returns
    -------
    Coloring
        See docstring for Coloring class.
    """
    if driver is None:
        driver = problem.driver

    ofs, wrts, _ = problem.model._get_totals_metadata(driver, of, wrt)

    model = problem.model

    if mode is None:
        mode = problem._orig_mode

    if mode != problem._orig_mode and mode != problem._mode:
        raise RuntimeError("given mode (%s) does not agree with Problem mode (%s)" %
                           (mode, problem._mode))

    if model._approx_schemes:  # need to use total approx coloring
        if driver and len(ofs) != len(driver._responses):
            raise NotImplementedError("Currently there is no support for approx coloring when "
                                      "linear constraint derivatives are computed separately "
                                      "from nonlinear ones.")
        _initialize_model_approx(model, driver, ofs, wrts)
        if model._coloring_info.coloring is None:
            kwargs = {n: v for n, v in model._coloring_info
                      if n in _DEF_COMP_SPARSITY_ARGS and v is not None}
            if 'use_scaling' in kwargs:
                del kwargs['use_scaling']
            kwargs['method'] = list(model._approx_schemes)[0]
            model.declare_coloring(**kwargs)
        if run_model:
            problem.run_model()
        coloring = model._compute_coloring(method=list(model._approx_schemes)[0],
                                           num_full_jacs=num_full_jacs, tol=tol, orders=orders)[0]
    else:
        J, sparsity_info = _get_total_jac_sparsity(problem, num_full_jacs=num_full_jacs, tol=tol,
                                                   orders=orders, setup=setup,
                                                   run_model=run_model, of=ofs, wrt=wrts,
                                                   driver=driver)
        coloring = _compute_coloring(J, mode)
        if coloring is not None:
            coloring._row_vars = list(ofs)
            coloring._row_var_sizes = [m['size'] for m in ofs.values()]
            coloring._col_vars = list(wrts)
            coloring._col_var_sizes = [m['size'] for m in wrts.values()]

            # save metadata we used to create the coloring
            coloring._meta.update(sparsity_info)

            if fname is not None:
                if ((model._full_comm is not None and model._full_comm.rank == 0) or
                        (model._full_comm is None and model.comm.rank == 0)):
                    coloring.save(fname)

    # save a copy of the abs2prom dict on the coloring object
    # so promoted names can be used when displaying coloring data
    # (also map auto_ivc names to the prom name of their connected input)
    if coloring is not None:
        coloring._abs2prom = abs2prom = model._var_allprocs_abs2prom.copy()
        conns = model._conn_global_abs_in2out
        for abs_out in abs2prom['output']:
            if abs_out.startswith('_auto_ivc.'):
                abs_in = _convert_auto_ivc_to_conn_name(conns, abs_out)
                abs2prom['output'][abs_out] = abs2prom['input'][abs_in]

    if driver:
        driver._total_jac = None

    # if we're running under MPI, make sure the coloring object is identical on all ranks
    # by broadcasting rank 0's coloring to the other ranks.
    if problem.comm.size > 1:
        if problem.comm.rank == 0:
            problem.comm.bcast(coloring, root=0)
        else:
            coloring = problem.comm.bcast(None, root=0)

    if coloring is not None:
        coloring._meta['timestamp'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        coloring._meta['source'] = problem._metadata['pathname']

    return coloring


def dynamic_total_coloring(driver, run_model=True, fname=None, of=None, wrt=None):
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
    of : iter of str or None
        Names of the 'response' variables.
    wrt : iter of str or None
        Names of the 'design' variables.

    Returns
    -------
    Coloring or None
        The computed coloring.
    """
    problem = driver._problem()
    if not problem.model._use_derivatives:
        msg = "Derivatives have been turned off. Skipping dynamic simul coloring."
        issue_warning(msg, category=DerivativesWarning)
        return

    driver._total_jac = None

    problem.driver._coloring_info.coloring = None

    num_full_jacs = driver._coloring_info.get('num_full_jacs',
                                              _DEF_COMP_SPARSITY_ARGS['num_full_jacs'])
    tol = driver._coloring_info.get('tol', _DEF_COMP_SPARSITY_ARGS['tol'])
    orders = driver._coloring_info.get('orders', _DEF_COMP_SPARSITY_ARGS['orders'])

    coloring = compute_total_coloring(problem, of=of, wrt=wrt, num_full_jacs=num_full_jacs, tol=tol,
                                      orders=orders, setup=False, run_model=run_model, fname=fname,
                                      driver=driver)

    driver._coloring_info.coloring = coloring

    if coloring is not None:
        driver._coloring_info.display()
        driver._setup_tot_jac_sparsity(coloring)

    driver._total_jac = None

    return coloring


def _run_total_coloring_report(driver):
    reports_dir = driver._problem().get_reports_dir()
    htmlpath = pathlib.Path(reports_dir).joinpath('total_coloring.html')

    display_coloring(source=driver, output_file=htmlpath,
                     as_text=bokeh_resources is None, show=False)


# entry point for coloring report
def _total_coloring_report_register():
    register_report('total_coloring', _run_total_coloring_report, 'Total coloring', 'Driver',
                    '_get_coloring', 'post')


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

            coloring_info = prob.driver._coloring_info.copy()
            if options.tolerance is not None:
                coloring_info.tol = options.tolerance
            if options.orders is not None:
                coloring_info.orders = options.orders
            if options.num_jacs is not None:
                coloring_info.num_full_jacs = options.num_jacs
            if options.show_sparsity:
                coloring_info.show_sparsity = options.show_sparsity

            with profiling('coloring_profile.out') if options.profile else do_nothing_context():
                coloring_info.coloring = \
                    compute_total_coloring(prob, num_full_jacs=coloring_info.num_full_jacs,
                                           tol=coloring_info.tol, orders=coloring_info.orders,
                                           setup=False, run_model=True, fname=outfile,
                                           driver=prob.driver)

            coloring_info.display()
        else:
            print("Derivatives are turned off.  Cannot compute simul coloring.")

    hooks._register_hook('final_setup', 'Problem', post=_total_coloring, exit=True)

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

    kwargs['recurse'] = not options.norecurse and not system._subsystems_allprocs

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
            coloring.display_bokeh(show=True)
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
                                    coloring = s._compute_coloring(**kwargs)[0]
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
                    colorings = system._compute_coloring(**kwargs)
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

    hooks._register_hook('final_setup', 'Problem', post=_partial_coloring, exit=True)

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
        if bokeh_resources is not None:
            Coloring.display_bokeh(source=options.file[0], show=True)
        else:
            Coloring.display_txt(source=options.file[0], html=False)

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
    if of is None or wrt is None:
        of, wrt, _ = model._get_totals_metadata(driver, of, wrt)

    # Initialization based on driver (or user) -requested "of" and "wrt".
    if (not model._owns_approx_jac or model._owns_approx_of is None or
            model._owns_approx_of != of or model._owns_approx_wrt is None or
            model._owns_approx_wrt != wrt):

        model._owns_approx_of = of
        model._owns_approx_wrt = wrt


class _ColSparsityJac(object):
    """
    A class to manage the assembly of a sparsity matrix by columns without allocating a dense jac.
    """

    def __init__(self, system, coloring_info):
        self._coloring_info = coloring_info

        nrows = sum([end - start for _, start, end, _, _ in system._jac_of_iter()])
        for _, _, end, _, _, _ in system._jac_wrt_iter(coloring_info.wrt_matches):
            pass

        ncols = end
        self._col_list = [None] * ncols
        self._ncols = ncols
        self._nrows = nrows

    def set_col(self, system, i, column):
        # record only the nonzero part of the column.
        # Depending on user specified tolerance, the number of nonzeros may be further reduced later
        nzs = np.nonzero(column)[0]
        if nzs.size > 0:
            if self._col_list[i] is None:
                self._col_list[i] = [nzs, np.abs(column[nzs])]
            else:
                oldnzs, olddata = self._col_list[i]
                if oldnzs.size == nzs.size and np.all(nzs == oldnzs):
                    olddata += np.abs(column[nzs])
                else:  # nonzeros don't match
                    scratch = np.zeros(column.size)
                    scratch[oldnzs] = olddata
                    scratch[nzs] += np.abs(column[nzs])
                    newnzs = np.nonzero(scratch)[0]
                    self._col_list[i] = [newnzs, scratch[newnzs]]

    def set_dense_jac(self, system, jac):
        """
        Assign a dense jacobian to this jacobian.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.
        jac : ndarray
            Dense jacobian.
        """
        for i in range(jac.shape[1]):
            self.set_col(system, i, jac[:, i])

    def __setitem__(self, key, value):
        # ignore any setting of subjacs based on analytic derivs
        pass

    def get_sparsity(self, system):
        """
        Assemble the sparsity matrix (COO) based on data collected earlier via set_col.

        Parameters
        ----------
        system : System
            The system that owns this jacobian.

        Returns
        -------
        coo_matrix
            The sparsity matrix.
        dict
            Metadata describing the sparsity computation.
        """
        rows = []
        cols = []
        data = []
        coloring_info = self._coloring_info
        for icol, tup in enumerate(self._col_list):
            if tup is None:
                continue
            rowinds, d = tup
            rows.append(rowinds)
            cols.append(np.full(rowinds.size, icol))
            data.append(d)

        if rows:
            rows = np.hstack(rows)
            cols = np.hstack(cols)
            data = np.hstack(data)

            # scale the data
            data *= (1. / np.max(data))

            info = _tol_sweep(data, coloring_info.tol, coloring_info.orders)
            data = data > info['good_tol']  # data is now a bool
            rows = rows[data]
            cols = cols[data]
            data = data[data]
        else:
            rows = np.zeros(0, dtype=int)
            cols = np.zeros(0, dtype=int)
            data = np.zeros(0, dtype=bool)
            info = {
                'tol': coloring_info.tol,
                'orders': coloring_info.orders,
                'good_tol': coloring_info.tol,
                'nz_matches': 0,
                'n_tested': 0,
                'zero_entries': 0,
                'J_size': 0,
            }

        return coo_matrix((data, (rows, cols)), shape=(self._nrows, self._ncols)), info


def display_coloring(source, output_file='total_coloring.html', as_text=False, show=True,
                     max_colors=200):
    """
    Display the coloring information from source to html format.

    Parameters
    ----------
    source : str or Coloring or Driver
        The source of the coloring information. If given as a string, source should
        be a valid coloring file path. If given as a Driver, display_colroing will
        attempt to obtain coloring information from the Driver.
    output_file : str or Path or None
        The output file to which the coloring display should be sent. If as_text
        is True and output_file ends with .html, then the coloring will be sent
        to that file as html, otherwise it will
        the html file will be saved in a temporary file.
    as_text : bool
        If True, render the coloring information using plain text.
    show : bool
        If True, open the resulting html file in the system browser.
    max_colors : int
        Bokeh colormaps support at most 256 colors. Near the upper end of this interval,
        the colors are nearly white and may be difficult to distinguish. This
        setting sets the upper limit for the color index before the pattern repeats.
    """
    if isinstance(source, str):
        coloring = Coloring.load(source)
    elif isinstance(source, Coloring):
        coloring = source
    elif hasattr(source, '_coloring_info'):
        coloring = source._coloring_info.coloring
    else:
        raise ValueError(f'display_coloring was expecting the source to be a valid '
                         f'coloring file or an instance of Coloring or driver '
                         f'but instead got f{type(source)}')

    if coloring is None:
        return

    if as_text or bokeh_resources is None:
        if bokeh_resources is None and not as_text:
            issue_warning("bokeh is not installed.\n"
                          "display_coloring will render output in plain text.")

        if output_file is None:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                output_file = f.name
                coloring.display_txt(out_stream=f, html=True)
        else:
            with open(output_file, 'w') as f:
                coloring.display_txt(out_stream=f, html=True)

        if show:
            webbrowser.open(f'file://{output_file}')

    else:
        coloring.display_bokeh(output_file, show=show, max_colors=max_colors)
