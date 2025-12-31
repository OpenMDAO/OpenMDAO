from difflib import get_close_matches
import networkx as nx
from networkx import dfs_edges, dfs_postorder_nodes
import numpy as np
from numpy import isscalar, reshape
from numbers import Number
from collections import deque
from copy import deepcopy
import textwrap
from itertools import chain

import webbrowser
import threading
import time
from http.server import HTTPServer

from openmdao.visualization.graph_viewer import write_graph
from openmdao.utils.general_utils import common_subpath, is_undefined, truncate_str, \
    all_ancestors, collect_error, collect_errors
from openmdao.utils.array_utils import array_connection_compatible, shape_to_len, \
    get_global_dist_shape, evenly_distrib_idxs, array_hash
from openmdao.utils.units import is_compatible
from openmdao.utils.units import unit_conversion
from openmdao.utils.indexer import indexer, Indexer, idx_list_to_index_array
from openmdao.utils.om_warnings import issue_warning, UnitsWarning
from openmdao.utils.units import _is_unitless, has_val_mismatch
from openmdao.visualization.tables.table_builder import generate_table
from openmdao.core.constants import INT_DTYPE
from openmdao.utils.mpi import MPI


# use hex colors here because the using english names was sometimes causing failure to show
# proper colors in the help dialog.
GRAPH_COLORS = {
    'input': 'peachpuff3',
    'output': 'skyblue3',
    'highlight': '#66ff00',
    'ambiguous': '#FF0800',
    'boundary': '#D3D3D3',
}

_shape_func_map = {
    # (src distributed, tgt distributed, fwd)
    (False, False, True): 'serial2serialfwd',
    (False, False, False): 'serial2serialrev',
    (False, True, True): 'serial2distfwd',
    (False, True, False): 'serial2distrev',
    (True, False, True): 'dist2serialfwd',
    (True, False, False): 'dist2serialrev',
    (True, True, True): 'dist2distfwd',
    (True, True, False): 'dist2distrev',
}



_continuous_copy_meta = ['val', 'units', 'shape', 'discrete', 'remote', 'distributed',
                         'global_shape']
_discrete_copy_meta = ['val', 'discrete', 'remote']


def _strip_np(shape):
    """
    Strip np.int64 from a shape string.

    Parameters
    ----------
    shape : tuple or None
        Shape to strip.

    Returns
    -------
    tuple or None
        Stripped shape.
    """
    if shape is None:
        return shape

    # so displayed shapes won't have np.int64 in them
    ret = []
    for item in shape:
        ret.append(int(item))
    return tuple(ret)


def is_equal(a, b):
    """
    Check equality of a and b.

    Parameters
    ----------
    a : any
        First value to compare.
    b : any
        Second value to compare.

    Returns
    -------
    bool
        True if a and b are equal, False otherwise.
    """
    if not (isinstance(b, type(a)) or isinstance(a, type(b))):
        return False

    if isinstance(a, np.ndarray):
        return a.size == b.size and np.all(np.squeeze(a) == np.squeeze(b))

    return a == b


def are_compatible_values(a, b, discrete, src_indices=None):
    """
    Check compatibility of values a and b.

    Parameters
    ----------
    a : any
        First value to compare.
    b : any
        Second value to compare.
    discrete : bool
        Whether the values are discrete.
    src_indices : Indexer or None
        The src_indices applied to a, if any.

    Returns
    -------
    bool
        True if a and b are compatible, False otherwise.
    """
    if discrete:
        return (isinstance(b, type(a)) or isinstance(a, type(b)))

    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    try:
        np.promote_types(a.dtype, b.dtype)
    except TypeError:
        return False

    if src_indices is None:
        return array_connection_compatible(a.shape, b.shape)

    return array_connection_compatible(b.shape, src_indices.indexed_src_shape)


class ConnError(ValueError):
    """
    An error raised when a connection is incompatible.

    Parameters
    ----------
    msg : str
        The error message.
    ident : hashable object
        Identifier of the object responsible for issuing the error.

    Attributes
    ----------
    ident : hashable object
        Identifier of the object responsible for issuing the error.
    """

    def __init__(self, msg, ident=None):
        super().__init__(msg)
        self.ident = ident


flag_type = np.uint8

base = flag_type(1)

DISCRETE = base << 0
REMOTE = base << 1
REQUIRE_CONNECTION = base << 2
SHAPE_BY_CONN = base << 3
UNITS_BY_CONN = base << 4
AMBIGUOUS_UNITS = base << 5
AMBIGUOUS_VAL = base << 6
# NOTE: if adding more flags, make sure the flag_type is large enough to hold all the flags.

AMBIGUOUS = AMBIGUOUS_VAL | AMBIGUOUS_UNITS
NONE_UP_VAL = REMOTE | AMBIGUOUS_VAL
BY_CONN = SHAPE_BY_CONN | UNITS_BY_CONN


class Defaults():
    """
    Container for default values of a variable.

    Attributes
    ----------
    val : any
        Default value.
    units : str or None
        Default units.
    src_shape : tuple or None
        Default source shape.
    """

    __slots__ = ['val', 'units', 'src_shape']

    def __init__(self, val=None, units=None, src_shape=None):
        self.val = val
        self.units = units
        self.src_shape = src_shape

    def __iter__(self):
        """
        Iterate over the attributes.

        Yields
        ------
        any
            The attribute value.
        str or None
            The units.
        tuple or None
            The source shape.
        """
        yield self.val
        yield self.units
        yield self.src_shape


_global_to_update = ['global_shape', 'global_size']
_local_to_update = ['global_shape', 'global_size']


class NodeAttrs():
    """
    Container for per-node variable metadata used by the connection graph.

    Attributes
    ----------
    pathname : str or None
        The pathname of System adding the node.
    rel_name : str or None
        The name relative to the System adding the node.
    _val : any or None
        The value of the node.
    _shape : tuple or None
        The shape of the node.
    _global_shape : tuple or None
        The global shape of the node.
    _units : str or None
        The units of the node.
    _src_inds_list : list or None
        The source indices list for the node.
    _meta : dict or None
        The global metadata of the node. This is always None for promoted variables nodees.
    _locmeta : dict or None
        The local metadata of the node. This is always None for promoted variables nodes.
    copy_shape : tuple or None
        The name of the variable to copy shape from.
    compute_shape : tuple or None
        The function to compute the shape of the node.
    copy_units : str or None
        The name of the variable to copy units from.
    compute_units : str or None
        The function to compute the units of the node.
    distributed : bool or None
        Whether the node is distributed.
    defaults : Defaults
        The default values for the node.
    flags : int
        The flags for the node.
    """

    __slots__ = ('pathname', 'rel_name', '_val', '_shape', '_global_shape', '_units', 'defaults',
                 '_src_inds_list', 'flags', '_meta', '_locmeta', 'copy_shape', 'compute_shape',
                 'copy_units', 'compute_units', 'distributed')

    def __init__(self):
        """
        Initialize NodeAttrs with all attributes set to None or default values.
        """
        self.pathname = None
        self.rel_name = None
        self.flags = flag_type(0)
        self._src_inds_list = []
        self._val = None
        self._shape = None
        self._global_shape = None
        self._units = None
        self._meta = None
        self._locmeta = None
        self.defaults = Defaults(None, None, None)
        self.copy_shape = None
        self.compute_shape = None
        self.copy_units = None
        self.compute_units = None
        self.distributed = None  # 3 states: None (unknown), False or True

    def __getattr__(self, key):
        """
        Get missing attribute.

        Parameters
        ----------
        key : str
            The attribute name.

        Returns
        -------
        None
            Always returns None for missing attributes.
        """
        return None

    def __getitem__(self, key):
        """
        Get attribute by key.

        Parameters
        ----------
        key : str
            The attribute name.

        Returns
        -------
        any
            The value of the attribute.
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        Set attribute by key.

        Parameters
        ----------
        key : str
            The attribute name.
        value : any
            The value to set.
        """
        setattr(self, key, value)

    def get(self, key, default=None):
        """
        Get attribute by key with a default value if not found.

        Parameters
        ----------
        key : str
            The attribute name.
        default : any, optional
            The default value to return if the attribute is not found.

        Returns
        -------
        any
            The value of the attribute or the default value.
        """
        return getattr(self, key, default)

    def update(self, kwargs):
        """
        Update attributes from a dictionary.

        This method is used by networkx to populate initial node attributes.

        Parameters
        ----------
        kwargs : dict
            Dictionary of attribute names and values to set.
        """
        # networkx uses this to populate the initial node attributes.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def msgname(self):
        """
        Get the full message name combining pathname and relative name.

        Returns
        -------
        str
            The full name if pathname exists, otherwise just the relative name.
        """
        if self.pathname:
            return f"{self.pathname}.{self.rel_name}"
        return self.rel_name

    def __repr__(self):
        """
        Return a string representation of the NodeAttrs object.

        Returns
        -------
        str
            A formatted table representation of non-None attributes.
        """
        rows = [[key, getattr(self, key)] for key in self.__slots__
                if getattr(self, key) is not None and key != 'meta']
        table = generate_table(rows, tablefmt='plain')
        return f"{self.__class__.__name__}:\n{table}"

    @property
    def meta(self):
        return self._meta

    @property
    def locmeta(self):
        return self._locmeta

    @property
    def src_inds_list(self):
        return self._src_inds_list

    @src_inds_list.setter
    def src_inds_list(self, value):
        self._src_inds_list = value
        if self._locmeta is not None:
            self._locmeta['src_inds_list'] = value

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, value):
        if self.flags & DISCRETE:
            self._val = value
        elif value is not None:
            if self._shape is None:
                self._val = value
                if np.ndim(value) > 0:
                    # setting 'shape' here will update the meta and locmeta if appropriate
                    self.shape = np.shape(value)
            else:
                if self._shape == ():
                    if isscalar(value):
                        self._val = value
                    else:
                        self._val = value.item()
                elif self._val is None:
                    if np.isscalar(value):
                        self._val = np.full(self._shape, value, dtype=float)
                    else:
                        self._val = reshape(value, self._shape)
                else:
                    self._val[:] = reshape(value, self._shape)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self._shape is None and shape is not None:
            shape = _strip_np(shape)
            self._shape = shape
            if self._shape != ():
                if self._val is not None and np.ndim(self._val) == 0:
                    # if val is a scalar, reshape it to the new shape
                    self._val = np.full(shape, self._val)

            if self.distributed is False:
                self.global_shape = shape

            if self._meta is not None:
                self._meta['shape'] = shape
                size = shape_to_len(shape)
                self._meta['size'] = size
                if self._locmeta is not None:
                    if self._val is None:
                        self._val = np.ones(shape)
                    self._locmeta['shape'] = shape
                    self._locmeta['size'] = size

    @property
    def size(self):
        return shape_to_len(self._shape)

    @property
    def global_shape(self):
        if self.distributed:
            return self._global_shape
        return self._shape

    @global_shape.setter
    def global_shape(self, global_shape):
        self._global_shape = global_shape
        if self._meta is not None:
            self._meta['global_shape'] = global_shape
            self._meta['global_size'] = self.global_size
            if self._locmeta is not None:
                self._locmeta['global_shape'] = global_shape
                self._locmeta['global_size'] = self.global_size

    @property
    def global_size(self):
        return shape_to_len(self.global_shape)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        if units is not None and self._units is None:
            if self.discrete:
                raise ValueError("Cannot set units for discrete variable "
                                 f"'{self.msgname()}'.")
            self._units = units
            if self._meta is not None:
                self._meta['units'] = units
                if self._locmeta is not None:
                    self._locmeta['units'] = units

    @property
    def discrete(self):
        return bool(self.flags & DISCRETE)

    @discrete.setter
    def discrete(self, value):
        if value:
            self.flags |= DISCRETE
        else:
            self.flags &= ~DISCRETE

    @property
    def remote(self):
        return bool(self.flags & REMOTE)

    @remote.setter
    def remote(self, value):
        if value:
            self.flags |= REMOTE
        else:
            self.flags &= ~REMOTE

    @property
    def require_connection(self):
        return bool(self.flags & REQUIRE_CONNECTION)

    @require_connection.setter
    def require_connection(self, value):
        if value:
            self.flags |= REQUIRE_CONNECTION
        else:
            self.flags &= ~REQUIRE_CONNECTION

    @property
    def shape_by_conn(self):
        return bool(self.flags & SHAPE_BY_CONN)

    @shape_by_conn.setter
    def shape_by_conn(self, value):
        # if shape defaults are are set, then this node's shape is known
        if self.defaults.src_shape is not None:
            return

        if value:
            self.flags |= SHAPE_BY_CONN
        else:
            self.flags &= ~SHAPE_BY_CONN

    @property
    def units_by_conn(self):
        return bool(self.flags & UNITS_BY_CONN)

    @units_by_conn.setter
    def units_by_conn(self, value):
        # if unit defaults are are set, then this node's units are known
        if self.defaults.units is not None:
            return
        if value:
            self.flags |= UNITS_BY_CONN
        else:
            self.flags &= ~UNITS_BY_CONN

    @property
    def ambiguous_units(self):
        return bool(self.flags & AMBIGUOUS_UNITS)

    @ambiguous_units.setter
    def ambiguous_units(self, value):
        if value:
            self.flags |= AMBIGUOUS_UNITS
        else:
            self.flags &= ~AMBIGUOUS_UNITS

    @property
    def ambiguous_val(self):
        return bool(self.flags & AMBIGUOUS_VAL)

    @ambiguous_val.setter
    def ambiguous_val(self, value):
        if value:
            self.flags |= AMBIGUOUS_VAL
        else:
            self.flags &= ~AMBIGUOUS_VAL

    @property
    def ambiguous(self):
        return bool(self.flags & AMBIGUOUS)

    @property
    def by_conn(self):
        return bool(self.flags & BY_CONN)

    @property
    def dyn_shape(self):
        return self.shape_by_conn or self.copy_shape is not None or self.compute_shape is not None

    @property
    def dyn_units(self):
        return self.units_by_conn or self.copy_units is not None or self.compute_units is not None

    @property
    def dynamic(self):
        return self.dyn_shape or self.dyn_units

    def units_from_child(self):
        """
        Get units, preferring defaults if available.

        Returns
        -------
        str or None
            The units from defaults if available, None if ambiguous, otherwise the node's units.
        """
        if self.defaults.units is not None:
            return self.defaults.units

        if self.ambiguous_units:
            return None

        return self.units

    def shape_from_child(self, node_meta, src_indices):
        """
        Get shape for a child node, accounting for defaults and src_indices.

        Parameters
        ----------
        node_meta : NodeAttrs
            The metadata of the child node.
        src_indices : Indexer or None
            The source indices applied to this node.

        Returns
        -------
        tuple or None
            The shape from defaults if available, None if src_indices are partial or
            if this node is distributed but the child is not, otherwise this node's shape.
        """
        if self.defaults.src_shape is not None:
            return self.defaults.src_shape

        if src_indices is not None and not src_indices.is_full_slice():
            # can't determine the shape of the parent if src_indices are present
            return None

        if self.distributed and not node_meta.distributed:
            return None

        return self.shape

    def val_from_child(self, node_meta, src_indices):
        """
        Get value for a child node, accounting for defaults and compatibility.

        Parameters
        ----------
        node_meta : NodeAttrs
            The metadata of the child node.
        src_indices : Indexer or None
            The source indices applied to this node.

        Returns
        -------
        any or None
            The value from defaults if available, None if ambiguous, incompatible,
            or if this node is distributed but the child is not, otherwise this node's value.
        """
        if self.flags & NONE_UP_VAL or src_indices is not None:
            return None

        if self.defaults.val is not None:
            return self.defaults.val

        if self.ambiguous_val:
            return None

        if self.defaults.src_shape is not None and self.val is not None:
            # this covers the situation where they set the src_shape but not the default val
            # and the val in the node is not compatible with the src_shape being passed to the
            # parent.
            if np.shape(self.val) != self.defaults.src_shape:
                return None

        if self.distributed and not node_meta.distributed:
            return None

        return self.val

    def dist_shapes(self, comm):
        """
        Gather shapes from all processes in a communicator.

        Parameters
        ----------
        comm : MPI communicator
            The MPI communicator to gather from.

        Returns
        -------
        list
            A list of shapes from all processes if MPI size > 1, otherwise a list
            containing only this node's shape.
        """
        if comm.size > 1:
            return comm.allgather(self.shape)
        else:
            return [self.shape]

    def as_dict(self):
        """
        Return attributes as a dictionary, excluding internal metadata.

        Returns
        -------
        dict
            A dictionary of non-None attributes excluding _meta, _locmeta, and _val.
        """
        skip = {'_meta', '_locmeta', '_val'}
        ret = {}
        for name in self.__slots__:
            if name not in skip:
                metaval = getattr(self, name)
                if metaval is not None:
                    ret[name] = metaval
        return ret

    def update_model_meta(self):
        """
        Update system metadata with the metadata from this node.

        Synchronizes global and local metadata dictionaries with current node state.
        """
        # update system metadata with the metadata from this node
        if self._meta is not None:
            for key in _global_to_update:
                self._meta[key] = getattr(self, key)
            if self._locmeta is not None:
                for key in _local_to_update:
                    self._locmeta[key] = getattr(self, key)


class AllConnGraph(nx.DiGraph):
    """
    A graph for all connection info.  Covers manual, implicit, and all promotions.

    Every connection in the graph forms a tree structure with an absolute output name at its
    root and all connected absolute input names as the leaf nodes.

    Node keys are tuples of the form (io, name), where io is either 'i' or 'o', and name is the
    name (either promoted or absolute) of a variable in the model.

    src_indices are stored in the edges between nodes.

    Attributes
    ----------
    _mult_inconn_nodes : set
        A set of nodes that have multiple incoming connections.
    _input_input_conns : set
        A set of nodes that have input to input connections.
    _first_pass : bool
        True if we're in the first pass of node data updates.
    _required_conns : set
        A set of input nodes that have required connections, direct or indirect, to an output node.
    _resolved : set
        A set of nodes that have been resolved and so can be skipped in the second pass.
    _has_dynamic_shapes : bool
        Whether the graph has dynamic shape behavior.
    _has_dynamic_units : bool
        Whether the graph has dynamic units behavior.
    _dist_shapes : list or None
        A list of shapes from all processes if MPI size > 1.
    _dist_sizes : list or None
        A list of sizes from all processes if MPI size > 1.
    _dist_nodes : set
        A set of nodes that are distributed.
    _problem_meta : dict or None
        The metadata of the problem.
    _bad_conns : set or None
        A set of nodes that have bad connections.
    msginfo : str or None
        The message information for the top level System.
    comm : MPI communicator or None
        The MPI communicator for the model.
    _var_existence : dict or None
        A dictionary of variable existence from the model.
    _var_allprocs_abs2meta : dict or None
        A dictionary of variable all processes absolute metadata from the model.
    _var_allprocs_discrete : dict or None
        A dictionary of variable all processes discrete from the model.
    _var_allprocs_abs2idx : dict or None
        A dictionary of variable all processes absolute index from the model.
    _var_abs2meta : dict or None
        A dictionary of variable absolute metadata from the model.
    _sync_auto_ivcs : dict
        A dictionary of auto_ivcs that require sync when setting intial values.
    _dangling_prom_inputs : set
        A set of dangling promoted inputs.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mult_inconn_nodes = set()
        self._input_input_conns = set()
        self._first_pass = True
        self._required_conns = set()
        self._resolved = set()
        self._has_dynamic_shapes = False
        self._has_dynamic_units = False
        self._dist_shapes = None
        self._dist_sizes = None
        self._dist_nodes = set()
        self._problem_meta = None
        self.msginfo = None
        self.comm = None
        self._var_existence = None
        self._var_allprocs_abs2meta = None
        self._var_allprocs_discrete = None
        self._var_allprocs_abs2idx = None
        self._var_abs2meta = None
        self._bad_conns = set()
        self._sync_auto_ivcs = {}  # auto_ivcs that require sync when setting intial values
        self._dangling_prom_inputs = set()

    def _collect_error(self, msg, exc_type=None, tback=None, ident=None):
        """
        Save an error message to raise as an exception later.

        Parameters
        ----------
        msg : str
            The connection error message to be saved.
        exc_type : class or None
            The type of exception to be raised if this error is the only one collected.
        tback : traceback or None
            The traceback of a caught exception.
        ident : int
            Identifier of the object responsible for issuing the error.
        """
        collect_error(msg, self._get_saved_errors(), exc_type, tback, ident, msginfo=self.msginfo)

    def _get_saved_errors(self):
        """Get saved errors.

        Returns
        -------
        any
            Returned value.
        """
        if self._problem_meta is None:
            return None
        return self._problem_meta['saved_errors']

    def find_node(self, pathname, varname, io=None):
        """
        Find a node in the graph.

        Parameters
        ----------
        pathname : str
            The current scoping system pathname.
        varname : str
            The variable name to find.
        io : str
            The io type of the variable

        Returns
        -------
        tuple of the form (io, name), where io is either 'i' or 'o'.
            The node found.
        """
        if pathname:
            prefix = pathname + '.'
            if varname.startswith(prefix):
                name = varname
            else:
                name = pathname + '.' + varname
        else:
            name = varname

        if io is None:
            node = ('o', name)
            if node not in self:
                node = ('i', name)
        else:
            node = (io[0], name)

        if node in self:
            return node

        msg = f"{pathname}: Variable '{varname}' not found."
        guesses = sorted(set(get_close_matches(name, [n[1] for n in self.nodes()],
                                               n=3, cutoff=0.15)))
        guesses = [g for g in guesses if not g.startswith('_auto_ivc.')]
        if guesses:
            msg = f"{msg} Perhaps you meant one of the following variables: {guesses}."
        raise KeyError(msg)

    def top_name(self, node):
        """Top name.

        Parameters
        ----------
        node : any
            node.

        Returns
        -------
        any
            Returned value.
        """
        if node[0] == 'i':
            root = self.input_root(node)
            return root[1] if root else None
        else:
            if self.out_degree(node) == 0:
                return node[1]

            for _, v in dfs_edges(self, node):
                if v[0] == 'o' and self.out_degree(v) == 0:
                    return v[1]
            return None

    def base_error(self, msg, src, tgt, src_indices=None):
        """
        Return the error message for a connection error.

        Parameters
        ----------
        msg : any
            msg.
        src : any
            source.
        tgt : any
            target.
        src_indices : any
            source indices.

        Returns
        -------
        any
            Returned value.
        """
        edge = (src, tgt)
        edge_meta = self.edges[edge]
        typ = edge_meta.get('type', None)
        type_map = {None: 'promote', 'manual': 'connect', 'implicit': 'implicitly connect'}
        typestr = type_map[typ]

        if typestr == 'promote' and src[0] == 'i':
            src, tgt = tgt, src   # promotion for inputs is up the tree

        fromto = f"'{src[1]}' to '{tgt[1]}'"

        indstr = ''
        if src_indices is not False:  # False means don't include src_indices in the error message
            src_indices = edge_meta.get('src_indices', None)

            if src_indices is not None:
                indstr = f" when applying index {truncate_str(src_indices, max_len=50)}"

        return f"Can't {typestr} {fromto}{indstr}: {msg}"

    def shape_error(self, src, tgt, src_shape, tgt_shape):
        """

        Return an error message for a shape incompatibility.

        Parameters
        ----------
        src : any
            source.
        tgt : any
            target.
        src_shape : any
            source shape.
        tgt_shape : any
            target shape.

        Returns
        -------
        any
            Returned value.
        """
        return self.base_error(f"shape {src_shape} of '{src[1]}' is incompatible with shape "
                               f"{tgt_shape} of '{tgt[1]}'.", src, tgt)

    def value_error(self, going_up, src, tgt, src_val, tgt_val):
        """
        Return an error message for a value incompatibility.

        Parameters
        ----------
        going_up : any
            going up.
        src : any
            source.
        tgt : any
            target.
        src_val : any
            source val.
        tgt_val : any
            target val.

        Returns
        -------
        any
            Returned value.
        """
        if going_up:
            src, tgt = tgt, src
            src_val, tgt_val = tgt_val, src_val

        if not self.nodes[tgt]['attrs'].discrete:
            sshp = np.shape(src_val)
            tshp = np.shape(tgt_val)
            if sshp != tshp:
                return self.shape_error(src, tgt, sshp, tshp)

        return self.base_error(f"value {truncate_str(src_val, max_len=50)} of '{src[1]}' is "
                               f"incompatible with value {truncate_str(tgt_val, max_len=50)} of "
                               f"'{tgt[1]}'.", src, tgt)

    def units_error(self, going_up, src, tgt, src_units, tgt_units):
        """
        Return an error message for a units incompatibility.

        Parameters
        ----------
        going_up : any
            going up.
        src : any
            source.
        tgt : any
            target.
        src_units : any
            source units.
        tgt_units : any
            target units.

        Returns
        -------
        any
            Returned value.
        """
        if going_up:
            src, tgt = tgt, src
            src_units, tgt_units = tgt_units, src_units

        return self.base_error(f"units '{src_units}' of '{src[1]}' are incompatible with units "
                               f"'{tgt_units}' of '{tgt[1]}'.", src, tgt, src_indices=False)

    def handle_error(self, going_up, src, tgt, exc):
        """
        Given an exception either raise it or save it for later.

        Parameters
        ----------
        going_up : any
            going up.
        src : any
            source.
        tgt : any
            target.
        exc : any
            exc.
        """
        if going_up:
            src, tgt = tgt, src

        excstr = str(exc)
        if isinstance(exc, ConnError):
            ident = exc.ident
        else:
            ident = frozenset((src, tgt))
            edge_meta = self.edges[src, tgt]
            src_indices = edge_meta.get('src_indices', None)
            excstr = self.base_error(msg=excstr, src=src, tgt=tgt, src_indices=src_indices)

        self._collect_error(f"{self.msginfo}: {excstr}", tback=exc.__traceback__, ident=ident)

    def input_root(self, node):
        """
        Return the top input predecessor to the given node.

        Parameters
        ----------
        node : (str, str)
            Tuple of the form ('i' or 'o', variable name)

        Returns
        -------
        (str, str) or None
            Node name ('i' or 'o', var name), or None.
        """
        assert node[0] == 'i'
        in_degree = self.in_degree
        preds = self.predecessors
        dangling = []
        for n in self.bfs_up_iter(node, include_self=True):
            if n[0] == 'i':
                if in_degree(n) == 0:  # over-promoted input or dangling input
                    dangling.append(n)
                else:
                    for io, _ in preds(n):
                        if io == 'o':
                            return n

        if dangling:
            return dangling[0]

    def msgname(self, node):
        """
        Get full name of node, including absolute names if they differ from the promoted name.

        Parameters
        ----------
        node : tuple or str
            The node to get the full name of.  Tuple of the form (io, name), where io is either
            'i' or 'o', or a variable name.

        Returns
        -------
        str
            The full name of the node.
        """
        if node not in self:  # might be a var name instead of a node
            orig = node
            node = ('i', node)
            if node not in self:
                node = ('o', node)
                if node not in self:
                    raise ValueError(f"Node '{orig}' not found in the graph.")

        names = self.absnames(node)

        names = sorted(names)
        if len(names) == 1:
            names = names[0]

        if node[1] == names:
            return names

        return f'{node[1]} ({names})'

    def startswith(self, prefix, node):
        """Startswith.

        Parameters
        ----------
        prefix : any
            prefix.
        node : any
            node.

        Returns
        -------
        any
            Returned value.
        """
        if prefix:
            return node[1].startswith(prefix)

        return True

    def sync_auto_ivcs(self, model):
        """
        Synchronize auto_ivc values that require it.

        Parameters
        ----------
        model : System
            The system containing the auto_ivcs to be synchronized.
        """
        has_vectors = model.has_vectors()
        all_sync_auto_ivcs = {k: v for k, v in self._sync_auto_ivcs.items() if v is not None}
        for allsync in model.comm.allgather(all_sync_auto_ivcs):
            for name, owner in allsync.items():
                if name not in all_sync_auto_ivcs:
                    if owner is not None:
                        all_sync_auto_ivcs[name] = owner
                elif owner is not None and owner < all_sync_auto_ivcs[name]:
                    all_sync_auto_ivcs[name] = owner

        for name, owner in sorted(all_sync_auto_ivcs.items()):
            node = self.find_node('', name, io='o')
            node_meta = self.nodes[node]['attrs']
            if owner == model.comm.rank:
                if has_vectors:
                    val = model._outputs[name]
                else:
                    val = node_meta.val
                model.comm.bcast(val, root=owner)
            else:
                val = model.comm.bcast(None, root=owner)
                if has_vectors:
                    model._outputs[name] = val
                else:
                    node_meta.val = val

            # reset the flag so we don't sync again unless the user changes the value again
            self._sync_auto_ivcs[name] = None

    def get_val_from_src(self, system, name, units=None, indices=None, get_remote=False, rank=None,
                         vec_name='nonlinear', kind=None, flat=False, use_vec=False, src_node=None):
        """Get val from source.

        Parameters
        ----------
        system : any
            system.
        name : any
            name.
        units : any
            units.
        indices : any
            indices.
        get_remote : any
            get remote.
        rank : any
            rank.
        vec_name : any
            vector name.
        kind : any
            kind.
        flat : any
            flat.
        use_vec : any
            use vector.
        src_node : any
            source node.

        Returns
        -------
        any
            Returned value.
        """
        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]['attrs']
        if src_node is None:
            src_node = self.get_root(node)
        src_meta = self.nodes[src_node]['attrs']

        src_inds_list = node_meta.src_inds_list

        if not get_remote and self.comm.size > 1:
            if src_meta.distributed and not node_meta.distributed:
                raise RuntimeError(f"{self.msginfo}: Non-distributed variable '{node[1]}' has "
                                   f"a distributed source, '{src_node[1]}', so you must retrieve "
                                   "its value using 'get_remote=True'.")
            elif node_meta.distributed and src_inds_list:
                # if src_meta.distributed or src_meta.remote:
                if src_meta.distributed:
                    model = system._problem_meta['model_ref']()
                    src_inds_list = self.inds_into_local_distrib(model, src_node, node,
                                                                    src_inds_list)

                    # min_idx, max_idx = idx_list_to_extent(src_meta.global_shape, src_inds_list)
                    # var_idx = self._var_allprocs_abs2idx[src_node[1]]
                    # sizes = self._var_sizes['output'][:, var_idx]
                    # # sizes for src var in each proc
                    # start = np.sum(sizes[:self.comm.rank])  # start index of src on this proc
                    # end = start + sizes[self.comm.rank]
                    # err = min_idx != max_idx and (not (start <= min_idx < end) or
                    #                               not (start <= max_idx < end))
                elif src_meta.remote:
                    err = True

                    if self.comm.allreduce(err, op=MPI.LOR):
                        raise RuntimeError(f"{self.msginfo}: Can't retrieve distributed variable "
                                            f"'{node[1]}' because its src_indices reference "
                                            "entries from other processes. You can retrieve values "
                                            "from all processes using "
                                            "`get_val(<name>, get_remote=True)`.")
        if use_vec:
            val = system._abs_get_val(src_node[1], get_remote, rank, vec_name, kind, flat,
                                      from_root=True)
        else:
            val = src_meta.val

            if is_undefined(val):
                raise ValueError(f"{system.msginfo}: Variable '{self.msgname(src_node)}' has not "
                                 "been initialized.")

        try:
            val = self.convert_get(node, val, src_meta.units, node_meta.units,
                                   src_inds_list, units, indices,
                                   get_remote=get_remote)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't get value of '{node[1]}': {str(err)}")

        if node[0] == 'i' and get_remote and node_meta.distributed and self.comm.size > 1:
            # gather parts of the distrib input value from all procs
            full_val = np.zeros(shape_to_len(node_meta.global_shape))
            sizes = np.asarray(self.get_dist_sizes(node), dtype=INT_DTYPE)
            offsets = np.zeros(sizes.size, dtype=INT_DTYPE)
            offsets[1:] = np.cumsum(sizes[:-1])
            self.comm.Allgatherv(val.ravel(), [full_val, sizes, offsets, MPI.DOUBLE])
            val = np.reshape(full_val, node_meta.global_shape)

        if flat and not node_meta.discrete:
            val = val.ravel()

        # print(f"{self.msginfo}: get_val_from_src: {name} {val}") # DBG

        return val

    def get_local_abs_in(self, system, name):
        """
        Retrieve the absolute name of a local input attached to the given name.

        The name may be promoted or absolute.

        Parameters
        ----------
        system : System
            The scoping system.
        name : str
            The promoted or absolute input name.
        """
        absnames = system._resolver.absnames(name, 'input')
        for absname in absnames:
            if not self.nodes[('i', absname)]['attrs'].remote:
                return absname

    def get_val(self, system, name, units=None, indices=None, get_remote=False, rank=None,
                vec_name='nonlinear', kind=None, flat=False, from_src=True):
        """
        Return the value of a variable.

        Parameters
        ----------
        system : System
            The System requesting the value.
        name : str
            The name of the variable to get the value of.
        units : str or None
            The units to convert to before returning the value.
        indices : int or iter of ints or None
            The indices or slice to return.
        get_remote : bool
            If True, retrieve the value even if it is on a remote process.
        rank : int or None
            If not None, only gather the value to this rank.
        vec_name : str
            The name of the vector to use.
        kind : str or None
            The kind of variable to get the value of.
        flat : bool
            If True, return the flattened version of the value.
        from_src : bool
            If True, retrieve the value of an input variable from its connected source.

        Returns
        -------
        any
            Returned value.
        """
        if indices is not None and not isinstance(indices, Indexer):
            indices = indexer(indices, flat_src=flat)

        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]['attrs']

        # ambiguous units aren't fatal during setup, but if we're getting a specific promoted
        # input that has ambiguous units, it becomes fatal, so we need to check that here.
        if node_meta.ambiguous_units:
            raise ValueError(self.ambig_units_msg(node))

        if from_src or node[0] == 'o':
            return self.get_val_from_src(system, name, units=units, indices=indices,
                                         get_remote=get_remote, rank=rank, vec_name=vec_name,
                                         kind=kind, flat=flat, use_vec=system.has_vectors())

        # since from_src is False, we're getting a specific input
        # (must use absolute name or have only a single leaf node and no src_indices between the
        # promoted input node and the leaf node)
        leaves = list(self.leaf_input_iter(node))

        if len(leaves) > 1:
            raise ValueError(
                f"{system.msginfo}: Promoted variable '{name}' refers to multiple "
                "input variables so the choice of input is ambiguous.  Either "
                "use the absolute name of the input or set 'from_src=True' to "
                "retrieve the value from the connected output.")

        leaf_meta = self.nodes[leaves[0]]['attrs']

        if system.has_vectors():
            model = system._problem_meta['model_ref']()
            if model._resolver.is_prom(node[1], 'input' if node[0] == 'i' else 'output'):
                abs_name = model._resolver.prom2abs(node[1])
            else:
                abs_name = node[1]
            val = system._abs_get_val(abs_name, get_remote, rank, vec_name, kind, flat,
                                      from_root=True)
        else:
            val = leaf_meta.val

        if is_undefined(val):
            raise ValueError(f"{system.msginfo}: Variable '{self.msgname(node)}' has not "
                                "been initialized.")

        if node != leaves[0]:
            if leaf_meta.src_inds_list:
                src_inds_list = leaf_meta.src_inds_list[len(node_meta.src_inds_list):]
                if src_inds_list:
                    raise RuntimeError(f"Can't get the value of promoted input '{node[1]}' when "
                                       "from_src is False because src_indices exist between the "
                                       f"promoted input and the actual input '{leaves[0][1]}'.")

        try:
            val = self.convert_get(node, val, node_meta.units, leaf_meta.units, (), units, indices,
                                   get_remote=get_remote)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't get value of '{node[1]}': {str(err)}")

        if flat and not node_meta.discrete:
            val = val.ravel()

        # print(f"{system.msginfo}: get_val: {name} {val}") # DBG

        return val

    def inds_into_local_distrib(self, model, src_node, tgt_node, inds):
        """
        Convert indices into distributed indices and verify that they only reference local entries.

        Parameters
        ----------
        model : Model
            The model.
        src_node : tuple of the form ('i' or 'o', name)
            The source node.
        tgt_node : tuple of the form ('i' or 'o', name)
            The target node.
        inds : list of Indexers
            The indices to convert.

        Returns
        -------
        list of Indexers
            The converted indices.
        """
        if inds and self.comm.size > 1 and self.nodes[src_node]['attrs'].distributed:
            src = src_node[1]
            src_indices = idx_list_to_index_array(inds)
            ssizes = model._var_sizes['output']
            sidx = model._var_allprocs_abs2idx[src]
            ssize = ssizes[self.comm.rank, sidx]
            start = np.sum(ssizes[:self.comm.rank, sidx])
            end = start + ssize
            if np.any(src_indices < start) or np.any(src_indices >= end):
                err = True
            else:
                err = False

            if self.comm.allreduce(err, op=MPI.LOR):
                raise RuntimeError(f"{model.msginfo}: Can't retrieve distributed variable "
                                    f"'{tgt_node[1]}' because its src_indices reference "
                                    "entries from other processes. You can retrieve values "
                                    "from all processes using "
                                    "`get_val(<name>, get_remote=True)`.")
            if start > 0:
                src_indices = src_indices - start
            inds = [indexer(src_indices)]

        return inds

    def set_val(self, system, name, val, units=None, indices=None):
        """
        Set the value of a variable.

        Parameters
        ----------
        system : System
            The System setting the value.
            system.
        name : str
            The name of the variable to set the value of.
        val : any
            The value to set.
        units : str or None
            The units to convert to before setting the value.
        indices : int or iter of ints or None
            The indices or slice to set.
        """
        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]['attrs']

        nodes = self.nodes
        src_node = self.get_root(node)
        src_meta = nodes[src_node]['attrs']
        src = src_node[1]

        model = system._problem_meta['model_ref']()

        if src_meta.discrete:
            if system.has_vectors():
                if src in model._discrete_outputs:
                    model._discrete_outputs[src] = val
                if node[0] == 'i':
                    for abs_in in self.absnames(node):
                        if abs_in in model._discrete_inputs:
                            model._discrete_inputs[abs_in] = val
            else:
                self.set_tree_val(model, src_node, val)

            return

        # every variable is continuous from here down
        if node[0] == 'o':
            tgt_units = None
            tgt_inds_list = ()
        else:
            tgt_units = node_meta.units
            tgt_inds_list = node_meta.src_inds_list

        src_units = src_meta.units

        if indices is None:
            inds = tgt_inds_list
        else:
            if not isinstance(indices, Indexer):
                indices = indexer(indices)
            inds = list(tgt_inds_list) + [indices]

        # do unit conversion on given val if needed
        try:
            sval = self.convert_set(val, src_units, tgt_units, (),  units)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't set value of '{self.msgname(node)}': "
                             f"{str(err)}")

        if node_meta.remote:
            if tgt_inds_list:
                issue_warning(f"{model.msginfo}: Cannot set the value of '{node[1]}':"
                                " Setting the value of a remote connected input with"
                                " src_indices is currently not supported, you must call"
                                " `run_model()` to have the outputs populate their"
                                " corresponding inputs.")

        if src_meta.remote:
            return

        if model.has_vectors():
            srcval = model._abs_get_val(src, get_remote=False)
            if inds and node[0] == 'i' and src_meta.distributed:
                inds = self.inds_into_local_distrib(model, src_node, node, inds)

            if np.ndim(srcval) > 0:
                self.set_subarray(srcval, inds, sval, node)
            else:
                srcval = sval

            model._outputs._abs_set_val(src, srcval)

            if src in self._sync_auto_ivcs:
                for leaf in self.leaf_input_iter(node):
                    if leaf[1] in model._vars_to_gather:
                        # mark this auto_ivc to be synchronized later
                        self._sync_auto_ivcs[src] = model._vars_to_gather[leaf[1]]
                        break

            # also set the input if it's absolute
            if node[0] == 'i' and node[1] in model._var_abs2meta['input']:
                try:
                    tval = self.convert_set(val, tgt_units, tgt_units, (),  units)
                except Exception as err:
                    raise ValueError(f"{system.msginfo}: Can't set value of '{self.msgname(node)}':"
                                     f" {str(err)}")
                if indices is None:
                    model._inputs._abs_set_val(node[1], tval)
                else:
                    model._inputs._abs_set_val(node[1], tval, idx=indices())
        else:
            srcval = src_meta.val

            if srcval is not None:
                if isinstance(srcval, Number):
                    if inds:
                        raise RuntimeError("Can't set a non-array using indices.")
                    src_meta.val = sval
                    srcval = src_meta.val
                else:
                    self.set_subarray(srcval, inds, sval, node)
            else:
                if inds:
                    raise RuntimeError(f"Shape of '{name}' isn't known yet so you can't use "
                                       f"indices to set it.")
                srcval = sval

            # propagate shape and value down the tree
            self.set_tree_val(model, src_node, srcval)

    def set_tree_val(self, model, src_node, srcval):
        """
        Set the value of a source in the tree and propagate it down the tree.

        Parameters
        ----------
        model : Model
            The model.
        src_node : tuple of the form ('i' or 'o', name)
            The source node.
        srcval : any
            The starting value to set.
        """
        nodes = self.nodes
        src_meta = nodes[src_node]['attrs']
        src_meta.val = srcval
        src_dist = src_meta.distributed
        has_vecs = model.has_vectors()

        for leaf in self.leaf_input_iter(src_node):
            tgt_meta = nodes[leaf]['attrs']
            if tgt_meta.remote:
                continue

            if tgt_meta.discrete:
                tgt_meta.val = srcval
                continue

            src_inds_list = tgt_meta.src_inds_list

            if tgt_meta.distributed:
                if src_dist:  # dist --> dist
                    pass
                else:  # serial --> dist
                    pass
            else:
                if src_dist:  # dist --> serial
                    pass
                else:  # serial --> serial
                    if src_inds_list:
                        tgt_meta.val = self.get_subarray(srcval, src_inds_list)
                    else:
                        tgt_meta.val = srcval

            if has_vecs:
                if tgt_meta.discrete:
                    pass
                else:
                    model._inputs._abs_set_val(leaf[1], tgt_meta.val)

    def bfs_up_iter(self, node, include_self=True):
        """
        Iterate up the tree from the given node.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node to start from.
        include_self : bool, optional
            Whether to include the given node in the iteration.
        """
        mypreds = self.predecessors
        if include_self:
            yield node
        stack = [mypreds(node)]
        while stack:
            for n in stack.pop():
                yield n
                stack.append(mypreds(n))

    def bfs_down_iter(self, node, include_self=True):
        """
        Iterate down the tree from the given node.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node to start from.
        include_self : bool, optional
            Whether to include the given node in the iteration.
        """
        mysuccs = self.successors
        if include_self:
            yield node
        stack = [mysuccs(node)]
        while stack:
            for n in stack.pop():
                yield n
                stack.append(mysuccs(n))

    def check_add_edge(self, group, src, tgt, **kwargs):
        """
        Check if an edge can be added to the graph and add it if it can.

        Parameters
        ----------
        group : Group
            The group.
        src : tuple of the form ('i' or 'o', name)
            The source node.
        tgt : tuple of the form ('i' or 'o', name)
            The target node.
        **kwargs : any
            The keyword arguments to add the edge.
            group.
        """
        if (src, tgt) in self.edges():
            return

        if src not in self or tgt not in self:
            raise ValueError(f"Node {src} or {tgt} not found in the graph.")

        if self.in_degree(tgt) != 0:
            self._mult_inconn_nodes.add(tgt)
            if src[0] == 'o':
                for p in self.pred[tgt]:
                    if p[0] == 'o':
                        self._bad_conns.add((src, tgt))
                        # sort names because sometimes the order is reversed
                        names = sorted([self.msgname(src), self.msgname(p)])
                        group._collect_error(
                            f"{group.msginfo}: Target '{self.msgname(tgt)}' cannot be "
                            f"connected to '{names[0]}' because it's already "
                            f"connected to '{names[1]}'.", ident=(src, tgt))
                        return

        self.add_edge(src, tgt, **kwargs)

    def node_name(self, system, name, io):
        """
        Return the name of a node.

        Parameters
        ----------
        system : System
            The system.
        name : str
            The name of the variable.
        io : str
            The I/O type of the variable.

        Returns
        -------
        tuple of the form ('i' or 'o', name)
            The node name.
        """
        return (io[0], system.pathname + '.' + name if system.pathname else name)

    def get_node_attrs(self, system, name, io):
        """
        Get attrs from a node, adding the node if necessary.
        """
        node = self.node_name(system, name, io)
        if node not in self:
            attrs = NodeAttrs()
            attrs.pathname = system.pathname
            attrs.rel_name = name

            self.add_node(node, attrs=attrs)
        else:
            attrs = self.nodes[node]['attrs']

        return node, attrs

    def get_path_prom(self, node):
        """
        Get the system path and promoted name of a node.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.

        Returns
        -------
        tuple of the form (str, str)
            The system pathname and promoted name of the node relative to the system.
        """
        meta = self.nodes[node]['attrs']
        return meta.pathname, meta.rel_name

    def set_model_meta(self, model, node, meta, locmeta):
        # this helps us keep graph nodes and variable metadata in sync.
        # TODO: these need to be consolidated into a single data structure!
        """
        Update node meta and locmete from the model.

        Parameters
        ----------
        model : Group
            The model.
        node : tuple of the form ('i' or 'o', name)
            The node.
        meta : dict or None
            The metadata of the node.
        locmeta : dict or None
            The local metadata of the node.
        """
        node_meta = self.nodes[node]['attrs']

        # this is only called on nodes corresponding to variables in the model, not on
        # nodes internal to the tree.
        if meta is not None:
            node_meta._meta = meta
            node_meta._locmeta = locmeta

            if locmeta is None and not node[1].startswith('_auto_ivc.'):
                node_meta.remote = True

            if node_meta.discrete:
                if locmeta is not None:
                    node_meta._val = locmeta['val']
            else:
                for key in ('shape_by_conn', 'copy_shape', 'compute_shape',
                            'units_by_conn', 'copy_units', 'compute_units'):
                    setattr(node_meta, key, meta[key])

                node_meta.distributed = meta['distributed'] if model.comm.size > 1 else False

                if not node_meta.rel_name.startswith('_auto_ivc.'):
                    if node_meta.distributed:
                        self._distributed_nodes.add(node)
                    if not node_meta.dyn_shape:
                        if node_meta.remote and node_meta.distributed:
                            node_meta._shape = (0,)
                        else:
                            node_meta._shape = meta['shape']
                        if locmeta is not None:
                            val = locmeta['val']
                            if isinstance(val, Number):
                                node_meta._val = val
                                if not node_meta.discrete:
                                    node_meta._val = float(val)
                            else:
                                node_meta._val = val.copy()
                                if not node_meta.discrete:
                                    node_meta._val = np.asarray(node_meta._val, dtype=float)

                if not node_meta.dyn_units:
                    node_meta._units = meta['units']

    def add_continuous_var(self, model, name, meta, locmeta, io):
        """
        Add a continuous variable to the graph.

        Parameters
        ----------
        model : Group
            The model.
        name : str
            The name of the variable.
        meta : dict or None
            The metadata of the variable.
        locmeta : dict or None
            The local metadata of the variable.
        io : str
            The I/O type of the variable, either 'input' or 'output'.
        """
        node, node_meta = self.get_node_attrs(model, name, io)
        node_meta.discrete = False

        self.set_model_meta(model, node, meta, locmeta)

        if node_meta.dyn_shape:
            self._has_dynamic_shapes = True
        if node_meta.dyn_units:
            self._has_dynamic_units = True

        if io == 'input' and meta['require_connection']:
            self._required_conns.add(name)
            node_meta.require_connection = True

    def add_discrete_var(self, model, name, meta, locmeta, io):
        """
        Add a discrete variable to the graph.

        Parameters
        ----------
        model : Group
            The model.
        name : str
            The name of the variable.
        meta : dict or None
            The metadata of the variable.
        locmeta : dict or None
            The local metadata of the variable.
        io : str
            The I/O type of the variable, either 'input' or 'output'.
        """
        node, node_meta = self.get_node_attrs(model, name, io)
        node_meta.discrete = True
        self.set_model_meta(model, node, meta, locmeta)

    def add_variable_meta(self, model):
        """
        Add variable metadata to the graph.

        Parameters
        ----------
        model : Group
            The model.
        """
        self.comm = model.comm
        self._problem_meta = model._problem_meta
        self.msginfo = model.msginfo

        self._distributed_nodes = set()

        for io in ['input', 'output']:
            loc = model._var_abs2meta[io]
            for name, meta in model._var_allprocs_abs2meta[io].items():
                locmeta = loc[name] if name in loc else None
                self.add_continuous_var(model, name, meta, locmeta, io)

            loc = model._var_discrete[io]
            for name, meta in model._var_allprocs_discrete[io].items():
                locmeta = loc[name] if name in loc else None
                self.add_discrete_var(model, name, meta, locmeta, io)

    def get_dist_shapes(self, node=None):
        """
        Get the distributed shapes of a variable.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.

        Returns
        -------
        dict or list of tuples
            The full distributed shapes dict or a single entry for the given node.
        """
        if self._dist_shapes is None:
            if self.comm.size > 1:
                existence = self._get_var_existence()

                # make sure we have all of the distributed vars
                dshapes = {}
                # at this point, _distributed_nodes only contains local vars
                for gnode in self._distributed_nodes:
                    io, name = gnode
                    mode = 'output' if io == 'o' else 'input'
                    if existence[mode][self.comm.rank, self._var_allprocs_abs2idx[name]]:
                        dshapes[gnode] = self.nodes[gnode]['attrs'].shape

                all_dshapes = {}

                for rank, dshp in enumerate(self.comm.allgather(dshapes)):
                    for n, shp in dshp.items():
                        if n not in all_dshapes:
                            all_dshapes[n] = [None] * self.comm.size
                        all_dshapes[n][rank] = shp

                self._distributed_nodes.update(all_dshapes.keys())
                self._dist_shapes = all_dshapes
            else:
                self._dist_shapes = {}

        if node is None:
            return self._dist_shapes

        if node in self._dist_shapes:
            for s in self._dist_shapes[node]:
                if s is not None:
                    return self._dist_shapes[node]

        # if we get here, node is dynamically shaped so we didn't know the shape the first time
        # around
        if node in self._distributed_nodes:
            node_meta = self.nodes[node]['attrs']
            dist_shapes = self.comm.allgather(node_meta.shape)
            self._dist_shapes[node] = dist_shapes
            return dist_shapes
        else:
            raise ValueError(f"Can't get distributed shapes for variable '{node[1]}' because it is "
                             "not a distributed variable in the model.")

    def get_dist_sizes(self, node=None):
        """
        Get the distributed sizes of a variable.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.

        Returns
        -------
        dict or list of ints
            The full distributed sized dict or a single entry for the given node.
        """
        if self._dist_sizes is None:
            if self.comm.size > 1:
                dshapes = self.get_dist_shapes()
                self._dist_sizes = dsizes = {}
                for n, shapes in dshapes.items():
                    sizes = [shape_to_len(shape) if shape is not None else 0
                                 for shape in shapes]
                    # if total size is 0, dshapes haven't been set yet
                    if np.sum(sizes) > 0:
                        dsizes[n] = sizes
            else:
                self._dist_sizes = {}

        if node is None:
            return self._dist_sizes

        if node in self._dist_sizes:
            return self._dist_sizes[node]

        if node in self._distributed_nodes:
            shapes = self.get_dist_shapes(node)
            sizes = [shape_to_len(shape) if shape is not None else 0 for shape in shapes]
            self._dist_sizes[node] = sizes
            return sizes
        else:
            raise ValueError(f"Can't get distributed sizes for variable '{node[1]}' because it is "
                             "not a distributed variable in the model.")

    def compute_global_shape(self, node):
        """
        Compute the global shape of a variable.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            node.

        Returns
        -------
        tuple of ints
            The global shape of the variable.
        """
        try:
            return get_global_dist_shape(self.get_dist_shapes(node))
        except ValueError as err:
            self._collect_error(f"Can't get global shape of distributed variable '{node[1]}': "
                                f"{err}", ident=node)
            return

    def add_promotion(self, io, group, prom_name, subsys, sub_prom, pinfo=None):
        # we invert the order here for inputs vs. outputs.  For inputs, the promoted name
        # is the source and the subsys name is the target.  For outputs, the promoted name is the
        # target and the subsys name is the source.  This gives us a nice tree that flows from
        # the absolute output to all of the connected absolute inputs which lets us use
        # dfs_postorder_nodes.
        """
        Add a promotion to the graph.

        Parameters
        ----------
        io : str
            The I/O type of the promotion.
        group : Group
            The group.
        prom_name : str
            The promoted name.
        subsys : Group
            The subsystem.
        sub_prom : str
            The sub promoted name.
        pinfo : dict or None
            The promotion information.
        """
        if io == 'input':
            src, _ = self.get_node_attrs(group, prom_name, io)
            tgt, tgt_attrs = self.get_node_attrs(subsys, sub_prom, io)
        else:
            src, _ = self.get_node_attrs(subsys, sub_prom, io)
            tgt, tgt_attrs = self.get_node_attrs(group, prom_name, io)

        if pinfo is None:
            src_indices = flat_src_indices = src_shape = None
        else:
            src_indices = pinfo.src_indices
            flat_src_indices = pinfo.flat
            src_shape = pinfo.src_shape

        self.check_add_edge(group, src, tgt, src_indices=src_indices,
                            flat_src_indices=flat_src_indices)

        if src_shape is not None:
            # group input defaults haven't been added yet, so just put src_shape there so we
            # can deal with it in the same way as the defaults.
            tgt_attrs.defaults.src_shape = src_shape

    def add_manual_connections(self, group):
        """
        Add manual connections to the graph for the given group.

        Parameters
        ----------
        group : Group
            The group.
        """
        manual_connections = group._manual_connections
        resolver = group._resolver
        allprocs_discrete_in = group._var_allprocs_discrete['input']
        allprocs_discrete_out = group._var_allprocs_discrete['output']

        for prom_tgt, (prom_src, src_indices, flat) in manual_connections.items():
            src_io = resolver.get_iotype(prom_src)
            if src_io is None:
                guesses = get_close_matches(prom_src, list(resolver.prom_iter('output')) +
                                            list(allprocs_discrete_out.keys()))
                self._bad_conns.add((prom_src, prom_tgt))
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                     f"'{prom_tgt}', but '{prom_src}' doesn't exist. Perhaps you "
                                     "meant to connect to one of the following outputs: "
                                     f"{guesses}.")
                continue

            if resolver.is_prom(prom_tgt, 'input'):
                tgt_io = 'input'
            else:
                tgt_io = resolver.get_iotype(prom_tgt)

            if tgt_io == 'output':
                # check that target is not also an input
                self._bad_conns.add((prom_src, prom_tgt))
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                    f"'{prom_tgt}', but '{prom_tgt}' is an output. All "
                                    "connections must be to an input.")
                continue

            if tgt_io is None:
                guesses = get_close_matches(prom_tgt, list(resolver.prom_iter('input')) +
                                            list(allprocs_discrete_in.keys()))
                self._bad_conns.add((prom_src, prom_tgt))
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                     f"'{prom_tgt}', but '{prom_tgt}' doesn't exist. Perhaps you "
                                     f"meant to connect to one of the following inputs: {guesses}.")
                continue

            out_comps = {n.rpartition('.')[0] for n in resolver.absnames(prom_src, src_io)}
            in_comps = {n.rpartition('.')[0] for n in resolver.absnames(prom_tgt, tgt_io)}

            if out_comps & in_comps:
                self._bad_conns.add((prom_src, prom_tgt))
                group._collect_error(f"{group.msginfo}: Source and target are in the same System "
                                     f"for connection from '{prom_src}' to '{prom_tgt}'.")
                continue

            src_node, _ = self.get_node_attrs(group, prom_src, src_io)
            tgt_node, _ = self.get_node_attrs(group, prom_tgt, tgt_io)

            if src_io == 'input' and tgt_io == 'input':
                self._input_input_conns.add((src_node, tgt_node))

            self.check_add_edge(group, src_node, tgt_node, type='manual', src_indices=src_indices,
                                flat_src_indices=flat)

    def add_group_input_defaults(self, group):
        """
        Add group input defaults to the graph for the given group.

        Parameters
        ----------
        group : Group
            The group.
        """
        notfound = []
        for name, gin_meta in group._group_inputs.items():
            path = group.pathname + '.' + name if group.pathname else name
            node = ('i', path)
            if node not in self:
                if not group._resolver.is_prom(name, 'input'):
                    notfound.append(name)
                    continue

            node, node_meta = self.get_node_attrs(group, name, 'input')

            defaults = node_meta.defaults

            for k, v in gin_meta.items():
                old = getattr(defaults, k)
                if old is not None:
                    issue_warning(f"{group.msginfo}: skipping default input {k} for '{name}' "
                                  f"because it was already set to {truncate_str(old)}.")
                    continue

                setattr(defaults, k, v)

        if notfound:
            group._collect_error(f"{group.msginfo}: The following group inputs, passed to "
                                 f"set_input_defaults(), could not be found: {sorted(notfound)}.",
                                 ident=frozenset(notfound))

    def update_all_node_meta(self, model):
        # this is called twice, once in _setup_global_connections and once after all of the
        # dynamic shapes have been computed.
        """
        Update all node metadata for the given model.

        Parameters
        ----------
        model : Group
            The model.
        """
        for abs_out in chain(self._var_allprocs_abs2meta['output'],
                             self._var_allprocs_discrete['output']):
            node = ('o', abs_out)
            if self.out_degree(node) == 0:
                continue

            self.resolve_conn_tree(model, node)

        self.update_dangling_prom_inputs(model)

        self._first_pass = False

    def gather_data(self, model):
        """
        Gather graph node and edge data from all processes for the given model.

        Parameters
        ----------
        model : Group
            The model.
        """
        # we don't have auto-ivcs yet, so some inputs are dangling
        myrank = model.comm.rank
        vars_to_gather = model._vars_to_gather
        nodes_to_send = {}
        edges_to_send = {}
        in_degree = self.in_degree
        nodes = self.nodes
        edges = self.edges
        for start_node in self.nodes():
            if in_degree(start_node) == 0:  # src node or dangling input node
                start_io, name = start_node
                if start_io == 'o':
                    own_start = name in vars_to_gather and vars_to_gather[name] == myrank
                    if own_start:
                        # include other promoted output nodes
                        for u, v in nx.dfs_edges(self, start_node):
                            if u not in nodes_to_send:
                                nodes_to_send[u] = nodes[u]['attrs'].as_dict()
                            if v[0] == 'i':
                                # input nodes 'own' their edge, so bail here
                                continue
                            nodes_to_send[v] = nodes[v]['attrs'].as_dict()
                            edges_to_send[u, v] = edges[u, v]

                    for in_node in self.leaf_input_iter(start_node):
                        _, abs_in = in_node
                        own_in = abs_in in vars_to_gather and vars_to_gather[abs_in] == myrank
                        if own_start or own_in:
                            path = nx.shortest_path(self, start_node, in_node)
                            for i, node in enumerate(path):
                                if node[0] == 'i':
                                    opath = path[:i]
                                    ipath = path[i:]
                                    break

                            if own_in:
                                edges_to_send[opath[-1], ipath[0]] = edges[opath[-1], ipath[0]]
                                for i, p in enumerate(ipath):
                                    nodes_to_send[p] = nodes[p]['attrs'].as_dict()
                                    if i > 0:
                                        edges_to_send[ipath[i-1], p] = edges[ipath[i-1], p]
                            else:  # own_start
                                if ipath:
                                    edges_to_send[opath[-1], ipath[0]] = edges[opath[-1],
                                                                                    ipath[0]]
                else:  # dangling input node, may be promoted
                    for in_node in self.leaf_input_iter(start_node):
                        _, abs_in = in_node
                        own_in = abs_in in vars_to_gather and vars_to_gather[abs_in] == myrank
                        if own_in:
                            path = nx.shortest_path(self, start_node, in_node)
                            for i, p in enumerate(path):
                                nodes_to_send[p] = nodes[p]['attrs'].as_dict()
                                if i > 0:
                                    edges_to_send[path[i-1], p] = edges[path[i-1], p]

        graph_info = model.comm.allgather((nodes_to_send, edges_to_send))

        all_abs2meta = model._var_allprocs_abs2meta
        all_discrete = model._var_allprocs_discrete
        for rank_nodes, _ in graph_info:
            for node, data in rank_nodes.items():
                if node not in self:
                    data['remote'] = True
                    attrs = NodeAttrs()
                    attrs.update(data)
                    self.add_node(node, attrs=attrs)
                    if node[0] == 'i':
                        na2m = all_abs2meta['input']
                        ndisc = all_discrete['input']
                    else:
                        na2m = all_abs2meta['output']
                        ndisc = all_discrete['output']

                    if node[1] in na2m:
                        nodes[node]['attrs'].meta = na2m[node[1]]
                    elif node[1] in ndisc:
                        nodes[node]['attrs'].meta = ndisc[node[1]]

        for _, rank_edges in graph_info:
            for edge, data in rank_edges.items():
                if edge in edges:
                    if data.get('src_indices', None) is not None:
                        if edges[edge].get('src_indices', None) is None:
                            edges[edge]['src_indices'] = data['src_indices']
                            edges[edge]['flat_src_indices'] = data['flat_src_indices']
                else:
                    if edge[0] not in nodes:
                        pass
                    if edge[1] not in nodes:
                        pass
                    self.add_edge(edge[0], edge[1], **data)

    def resolve_from_children(self, model, src_node, node, auto=False):
        """
        Resolve metadata from children for the given node.

        This propagates information up the tree from leaf inputs up to either the root input node
        or to the root auto_ivc node.

        Parameters
        ----------
        model : Group
            The model.
        src_node : tuple of the form ('i' or 'o', name)
            The source node.
        node : tuple of the form ('i' or 'o', name)
            The node.
        auto : bool
            Whether the source node is an auto_ivc node.
        """
        if self.out_degree(node) == 0:  # skip leaf nodes
            return

        try:
            children_meta = \
                [(self.nodes[child]['attrs'], self.edges[(node, child)].get('src_indices', None))
                 for child in self.succ[node]]

            node_meta = self.nodes[node]['attrs']
            if node[0] == 'i':
                remote = all(m.remote for m, _ in children_meta)
            else:
                remote = False  # dont' transfer 'remote' status to connected outputs

            node_meta.remote = remote
            node_meta.require_connection = any(m.require_connection for m, _ in children_meta)
            node_meta.discrete = discrete = self.get_discrete_from_children(model, node,
                                                                            children_meta)
            node_meta.distributed = \
                self.get_distributed_from_children(model, node, children_meta, auto,
                                                   self.nodes[src_node]['attrs'].distributed)

            ambig_units = ambig_val = None
            if node[0] == 'o':
                for i, tup in enumerate(children_meta):
                    cm, _ = tup
                    ambig = cm.ambiguous
                    if ambig:
                        nlist = [s for s in self.succ[node]]
                        if cm.ambiguous_units:
                            ambig_units = nlist[i]
                        elif cm.ambiguous_val:
                            ambig_val = nlist[i]
                        break

            if not discrete:
                node_meta.units_by_conn = all(m.units_by_conn for m, _ in children_meta)
                if not ambig_units:
                    node_meta.units = self.get_units_from_children(model, node, children_meta,
                                                                   node_meta.defaults)
                node_meta.shape_by_conn = all(m.shape_by_conn for m, _ in children_meta)
                node_meta.shape = \
                    self.get_shape_from_children(node, children_meta, node_meta.defaults)

            if not ambig_val:
                val = self.get_val_from_children(model, node, children_meta, node_meta.defaults,
                                                 auto)
                if val is not None:
                    if node[1].startswith('_auto_ivc.'):
                        val = deepcopy(val)
                        node_meta.val = val
                        if node_meta._locmeta is not None:
                            node_meta._locmeta['val'] = val
                    else:
                        node_meta.val = val

            if ambig_units:
                raise ConnError(self.ambig_units_msg(ambig_units))
            if ambig_val:
                raise ConnError(self.ambig_values_msg(ambig_val))

        except Exception as err:
            if isinstance(err, ConnError):
                model._collect_error(f"{model.msginfo}: {err}", tback=err.__traceback__,
                                     ident=node)
            else:
                model._collect_error(f"{model.msginfo}: While resolving children of '{node[1]}': "
                                     f"{err}", tback=err.__traceback__,
                                     ident=node)

    def ambig_units_msg(self, node, incompatible=False):
        """
        Generate a message for an ambiguous or incompatible units error.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.
        incompatible : bool
            Whether the units are incompatible.

        Returns
        -------
        any
            Returned value.
        """
        rows = []
        self.find_ambiguous_causes(node, rows, 'units')
        rows = sorted(rows, key=lambda x: x[0])
        table = textwrap.indent(str(generate_table(rows, tablefmt='plain')), '   ')
        msg = (f"The following inputs promoted to '{node[1]}' have different "
               f"units:\n{table}")
        if incompatible:
            msg += "\n   These units are incompatible."
        else:
            msg += ("\n   Call model.set_input_defaults('"
                    f"{self.top_name(node)}', units=?) to remove the ambiguity.")
        return msg

    def ambig_shapes_msg(self, node, children_meta):
        """
        Generate a message for an ambiguous shapes error.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.
        children_meta : list of tuples of the form (NodeAttrs, list of Indexers or None)
            The children metadata and source indices.

        Returns
        -------
        str
            The message.
        """
        node_meta = self.nodes[node]['attrs']
        shapes = [m.shape_from_child(node_meta, src_indices) for m, src_indices in children_meta]
        children = [n for _, n in self.succ[node]]
        rows = sorted((n, s) for n, s in zip(children, shapes))
        table = textwrap.indent(str(generate_table(rows, tablefmt='plain')), '   ')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"incompatible shapes:\n{table}")

    def ambig_values_msg(self, node):
        """
        Generate a message for an ambiguous values error.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.

        Returns
        -------
        str
            The message.
        """
        causes = []
        self.find_ambiguous_causes(node, causes, 'val')
        children = [n for n, _ in causes]
        child_nodes = [('i', n) for n in children]
        causing_meta = [self.nodes[n]['attrs'] for n in child_nodes]
        units_list = [m.units for m in causing_meta]
        vals = [m.val for m in causing_meta]
        ulist = [u if u is not None else '' for u in units_list]
        vlist = [truncate_str(v, max_len=60) for v in vals]
        rows = sorted((n, u, v) for n, u, v in zip(children, ulist, vlist))
        table = textwrap.indent(str(generate_table(rows, tablefmt='plain')), '   ')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"values, so the value of '{node[1]}' is ambiguous:\n{table}\n   Call "
                f"model.set_input_defaults('"
                f"{self.top_name(node)}', val=?) to remove the ambiguity.")

    def find_ambiguous_causes(self, node, causes, data_name):
        """
        Find all of the nodes that are causing the ambiguity, starting from an ambiguous node.

        Parameters
        ----------
        node : tuple
            The node to start from.
        causes : list
            A list of tuples of the form (node_name, data_value).
        data_name : str
            The name of the data to find the causes of.
        """
        attr = f"ambiguous_{data_name}"

        for child in self.succ[node]:
            child_meta = self.nodes[child]['attrs']
            if getattr(child_meta, attr):
                self.find_ambiguous_causes(child, causes, data_name)
            else:
                causes.append((child[1], getattr(child_meta, data_name)))

    def get_units_from_children(self, model, node, children_meta, defaults):
        """
        Get the units from the children of the given node.

        Parameters
        ----------
        model : Group
            The model.
        node : tuple of the form ('i' or 'o', name)
            The node.
        children_meta : list of tuples of the form (NodeAttrs, list of Indexers or None)
            The children metadata and source indices.
        defaults : Defaults
            The default metadata for the node.

        Returns
        -------
        str or None
            The units from the children of the given node.
        """
        start = None
        nodes = self.nodes

        is_output = node[0] == 'o'
        node_ambig = False
        child_units_differ = False

        unset = object()
        start = unset
        for chmeta, _ in children_meta:
            if chmeta.units_by_conn and self._first_pass:
                continue

            if start is unset:
                start = chmeta.units_from_child()
                node_ambig = chmeta.ambiguous_units
                continue

            u = chmeta.units_from_child()
            node_ambig |= chmeta.ambiguous_units
            if is_output and node_ambig:
                # we want the ambiguous child (input) for error reporting
                for s in self.succ[node]:
                    if nodes[s]['attrs'].ambiguous_units:
                        raise ConnError(self.ambig_units_msg(s))

            one_none = (start is not None and u is None) or (start is None and u is not None)
            child_units_differ |= start != u

            if not one_none and not is_compatible(start, u):
                raise ConnError(self.ambig_units_msg(node, incompatible=True))

        if start is unset:
            start = None

        if is_output:
            return start

        if defaults.units is not None:
            # setting default units removes any ambiguity
            nodes[node]['attrs'].ambiguous_units = False
            return defaults.units

        if child_units_differ:
            nodes[node]['attrs'].ambiguous_units = True
        else:
            # if a child is ambiguous, this node is also ambiguous if the default units are not set
            if node_ambig:
                nodes[node]['attrs'].ambiguous_units = True
            return start

    def get_shape_from_children(self, node, children_meta, defaults):
        """
        Get the shape from the children of the given node.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.
        children_meta : list of tuples of the form (NodeAttrs, list of Indexers or None)
            The children metadata and source indices.
        defaults : Defaults
            The default metadata for the node.

        Returns
        -------
        tuple or None
            The shape from the children of the given node.
        """
        node_meta = self.nodes[node]['attrs']
        start = None
        for chmeta, src_indices in children_meta:
            shape = chmeta.shape_from_child(node_meta, src_indices)
            val = chmeta.val_from_child(node_meta, src_indices)
            if shape is None and val is not None and np.ndim(val) > 0:
                if src_indices is None:
                    shape = np.shape(val)

            if shape is not None:
                if start is None:
                    start = shape
                else:
                    if not array_connection_compatible(start, shape):
                        raise ConnError(self.ambig_shapes_msg(node, children_meta))

        if defaults.val is not None and np.ndim(defaults.val) > 0:
            return np.shape(defaults.val)

        return start

    def get_val_from_children(self, model, node, children_meta, defaults, auto):
        """
        Get the value from the children of the given node.

        Parameters
        ----------
        model : Group
            The model.
        node : tuple of the form ('i' or 'o', name)
            The node.
        children_meta : list of tuples of the form (NodeAttrs, list of Indexers or None)
            The children metadata and source indices.
        defaults : Defaults
            The default metadata for the node.
        auto : bool
            Whether the source node is an auto_ivc node.

        Returns
        -------
        any or None
            The value from the children of the given node.
        """
        node_meta = self.nodes[node]['attrs']

        start = None
        unshaped_scalar = False
        for i, tup in enumerate(children_meta):
            ch_meta, src_indices = tup
            val = ch_meta.val_from_child(node_meta, src_indices)

            if ch_meta.ambiguous_val:
                node_meta.ambiguous_val = True
                if auto:
                    continue

            if val is not None:
                if start is None or (unshaped_scalar and ch_meta.shape is not None):
                    start = val
                    start_type = type(start)
                    start_units = ch_meta.units_from_child()
                    unshaped_scalar = ch_meta.shape is None and np.ndim(start) == 0

                elif auto:  # values must be the same or value of auto_ivc will be ambiguous

                    if ch_meta.discrete:
                        if start_type is not type(val):
                            node_meta.ambiguous_val = True
                            continue

                        if isinstance(start, np.ndarray):
                            if not np.all(start == val):
                                node_meta.ambiguous_val = True
                                continue

                        if start != val:
                            node_meta.ambiguous_val = True

                    else:  # continuous
                        if has_val_mismatch(start_units, start, ch_meta.units, val):
                            node_meta.ambiguous_val = True
                elif not (ch_meta.shape is None and np.ndim(val) == 0):
                    if not are_compatible_values(start, val, ch_meta.discrete,
                                                 src_indices=src_indices):
                        slist = list(self.succ[node])
                        raise ConnError(self.value_error(True, slist[i], node, start, val))

        if defaults.val is not None:
            node_meta.ambiguous_val = False
            return defaults.val

        if not node_meta.ambiguous_val:
            return start

    def get_discrete_from_children(self, group, node, children_meta):
        """
        Get the discrete flag from the children of the given node.

        Parameters
        ----------
        group : Group
            The group.
        node : tuple of the form ('i' or 'o', name)
            The node.
        children_meta : list of tuples of the form (NodeAttrs, list of Indexers or None)
            The children metadata and source indices.

        Returns
        -------
        bool
            The discrete flag from the children of the given node.
        """
        discretes = [m.discrete for m, _ in children_meta]
        dset = set(discretes)
        if len(dset) == 1:
            discrete = dset.pop()
            if discrete:
                node_meta = self.nodes[node]['attrs']
                if node_meta.defaults.units is not None:
                    group._collect_error(f"Cannot set 'units={node_meta.defaults.units}' for "
                                         f"discrete variable '{node[1]}'.")
                if node_meta.defaults.src_shape is not None:
                    group._collect_error(f"Cannot set 'shape={node_meta.defaults.src_shape}' for "
                                         f"discrete variable '{node[1]}'.")

            return discrete
        else:
            slist = list(self.succ[node])
            discs = [s for s, d in zip(slist, discretes) if d]
            non_discs = [s for s, d in zip(slist, discretes) if not d]
            raise ConnError(f"'{node[1]}' has discrete ({sorted(discs)}) and non-discrete "
                            f"({sorted(non_discs)}) children.")

    def get_distributed_from_children(self, model, node, children_meta, auto, src_distributed):
        """
        Get the distributed flag from the children of the given node.

        Parameters
        ----------
        model : Group
            The model.
        node : tuple of the form ('i' or 'o', name)
            The node.
        children_meta : list of tuples of the form (NodeAttrs, list of Indexers or None)
            The children metadata and source indices.
        auto : bool
            Whether the source node is an auto_ivc node.
        src_distributed : bool
            Whether the source node is distributed.

        Returns
        -------
        bool or None
            The distributed flag from the children of the given node.
        """
        # A parent is only desginated as distributed if it has only one child and that child is
        # distributed.

        dist = set()
        has_src_indices = False
        for m, src_indices in children_meta:
            dist.add(m.distributed)
            if src_indices is not None:
                has_src_indices = True

        if auto and True in dist:
            return None  # error will be collected later

        if len(dist) > 1:
            return None  # leave dist ambiguous

        ret = dist.pop()

        if ret is False or not src_distributed:
            return False

        if has_src_indices or len(children_meta) > 1:
            return None  # leave dist ambiguous

        return ret

    def get_defaults(self, meta):
        """
        Get the default metadata for the given node.

        Parameters
        ----------
        meta : NodeAttrs
            The metadata for the node.

        Returns
        -------
        tuple
            The default value, units, and source shape.
        """
        return meta.defaults.val, meta.defaults.units, meta.defaults.src_shape

    def get_parent_val_shape_units(self, parent, child):
        """
        Get the value, shape, and units from the parent of the given child node.

        Parameters
        ----------
        parent : tuple of the form ('i' or 'o', name)
            The parent node.
        child : tuple of the form ('i' or 'o', name)
            child.

        Returns
        -------
        any
            The value, shape, and units from the parent of the given child node.
        """
        parent_meta = self.nodes[parent]['attrs']
        src_indices = self.edges[(parent, child)].get('src_indices', None)
        val = parent_meta.val
        shape = parent_meta.shape

        if not (src_indices is None or shape is None):
            if src_indices._src_shape is None:
                src_indices.set_src_shape(shape)
            shape = src_indices.indexed_src_shape
            if val is not None:
                val = src_indices.indexed_val(np.atleast_1d(val))

        if parent_meta.ambiguous_val:
            val = None

        units = None if parent_meta.ambiguous_units else parent_meta.units

        return val, shape, units

    def resolve_output_input_connection(self, model, src, tgt):
        """
        Check the compatibility of a connection between an output and input node.

        This happends when going down the tree.

        Parameters
        ----------
        model : Group
            The model.
        src : tuple of the form ('i' or 'o', name)
            The source output node.
        tgt : tuple of the form ('i' or 'o', name)
            The target input node.
        """
        src_meta = self.nodes[src]['attrs']
        tgt_meta = self.nodes[tgt]['attrs']
        src_discrete = src_meta.discrete
        tgt_discrete = tgt_meta.discrete

        if src_discrete != tgt_discrete:
            dmap = {True: 'discrete', False: 'continuous'}
            raise TypeError(f"Can't connect {dmap[tgt_discrete]} variable "
                            f"'{tgt[1]}' to {dmap[src_discrete]} variable '{src[1]}'.")

        src_val = src_meta.val

        if src_discrete:
            if not src_meta.remote and not tgt_meta.remote and not tgt_meta.ambiguous_val:
                if not are_compatible_values(src_val, tgt_meta.val, src_discrete):
                    raise ConnError(self.value_error(False, src, tgt, src_val, tgt_meta.val))
        else:
            edge = (src, tgt)
            src_indices = self.edges[edge].get('src_indices', None)

            src_units = src_meta.units
            tgt_units = tgt_meta.units

            skip_val_shape = (src_meta.distributed or tgt_meta.distributed) and model.comm.size > 1

            src_shape = src_meta.shape
            tgt_shape = tgt_meta.shape

            if src_indices is not None and src_shape is not None:
                if src_indices._src_shape is None:
                    src_indices.set_src_shape(src_shape)
                src_shape = src_indices.indexed_src_shape
                if src_val is not None:
                    src_val = src_indices.indexed_val(np.atleast_1d(src_val))

            if src_shape is not None and not skip_val_shape:
                if tgt_shape is not None:
                    if not array_connection_compatible(src_shape, tgt_shape):
                        raise ConnError(self.shape_error(src, tgt, src_shape, tgt_shape))
                elif not tgt_meta.ambiguous_val:
                    tgt_meta.shape = src_shape

            if src_val is not None and not skip_val_shape:
                if tgt_meta.val is not None:
                    if not are_compatible_values(src_val, tgt_meta.val, src_discrete, src_indices):
                        raise ConnError(self.value_error(False, src, tgt, src_val, tgt_meta.val))
                elif not tgt_meta.ambiguous_val:
                    tgt_meta.val = src_val

            if src_units is not None:
                if tgt_units is not None:
                    if tgt_units != 'ambiguous' and not is_compatible(src_units, tgt_units):
                        raise ConnError(self.units_error(False, src, tgt, src_units, tgt_units))
                elif tgt_meta.units_by_conn:
                    tgt_meta.units = src_units

    def check_src_to_tgt_indirect(self, model, src, tgt, src_shape, tgt_shape):
        """
        Check compatibility between nodes that are not directly connected.

        Parameters
        ----------
        model : Group
            model.
        src : tuple of the form ('i' or 'o', name)
            The source node.
        tgt : tuple of the form ('i' or 'o', name)
            The target node.
        src_shape : tuple
            The source shape.
        tgt_shape : tuple
            The target shape.
        """
        src_meta = self.nodes[src]['attrs']
        tgt_meta = self.nodes[tgt]['attrs']

        if src_shape is not None and not (src_meta.distributed and src_meta.remote):
            if tgt_shape is not None:
                if not (tgt_meta.distributed and tgt_meta.remote):
                    if not array_connection_compatible(src_shape, tgt_shape):
                        self._collect_error(self.shape_error(src, tgt, src_shape, tgt_shape))
                        return
            elif not tgt_meta.ambiguous_val:
                tgt_meta.shape = src_shape

        src_units = src_meta.units
        tgt_units = tgt_meta.units
        if src_units is not None:
            if tgt_units is not None:
                if tgt_units != 'ambiguous' and not is_compatible(src_units, tgt_units):
                    self._collect_error(self.units_error(False, src, tgt, src_units, tgt_units))
            elif tgt_meta.units_by_conn:
                tgt_meta.units = src_units

    def get_dist_offset(self, node, rank, flat):
        """
        Get the distributed offset and size of axis 0 of the given node.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.
        rank : int
            rank.
        flat : bool
            Whether the source is flat.

        Returns
        -------
        tuple
            The distributed offset and size of axis 0 of the given node.
        """
        offset = 0
        for i, dshape in enumerate(self._dist_shapes[node]):
            if i == rank:
                break
            if dshape is not None:
                if flat:
                    offset += shape_to_len(dshape)
                else:
                    offset += dshape[0] if len(dshape) > 0 else 1

        if dshape is None:
            return offset, 0

        if flat:
            sz = shape_to_len(dshape)
        else:
            sz = dshape[0] if len(dshape) > 0 else 1

        return offset, sz

    def check_dist_connection(self, model, src_node):
        """
        Check a connection starting at src where src and/or a target is distributed.

        Parameters
        ----------
        model : Group
            The model.
        src_node : tuple of the form ('i' or 'o', name)
            The source node.
        """
        nodes = self.nodes
        src_meta = nodes[src_node]['attrs']
        if src_meta.shape is None:
            return  # an earlier error has occurred, so skip

        src_dist = src_meta.distributed

        if src_dist:
            if src_meta.global_shape is None:
                src_meta.global_shape = self.compute_global_shape(src_node)
            src_inds_shape = src_meta.global_shape
        else:
            src_inds_shape = src_meta.shape

        leaves = list(self.leaf_input_iter(src_node))
        for tgt_node in leaves:
            tgt_meta = nodes[tgt_node]['attrs']
            tgt_dist = tgt_meta.distributed

            src_inds_list = tgt_meta.src_inds_list

            if tgt_meta.distributed:
                if tgt_meta.global_shape is None:
                    tgt_meta.global_shape = self.compute_global_shape(tgt_node)

            if src_dist:
                if tgt_dist:  # dist --> dist
                    src_shape = src_meta.shape
                    tgt_shape = tgt_meta.shape
                    if not src_inds_list:
                        # no src_indices, so shape of dist src must match shape of dist tgt on
                        # each rank, and we must specify src_indices to match src and tgt on
                        # each rank.
                        if tgt_shape is not None:
                            offset, sz = self.get_dist_offset(tgt_node, self.comm.rank, False)
                            if sz is not None:
                                if sz == 0:
                                    src_indices = \
                                        indexer(slice(0, 0), flat_src=True, src_shape=(0,))
                                else:
                                    src_indices = \
                                        indexer(slice(offset, offset + sz),
                                                flat_src=False, src_shape=src_meta.global_shape)

                                path = nx.shortest_path(self, src_node, tgt_node)
                                self.edges[(path[-2], tgt_node)]['src_indices'] = src_indices
                                tgt_meta.src_inds_list = src_inds_list = [src_indices]

                else:  # dist --> serial
                    if not src_inds_list:
                        self._collect_error(f"Can't automatically determine src_indices for "
                                             f"connection from distributed variable '{src_node[1]}'"
                                             f" to serial variable '{tgt_node[1]}'.")
                        return
                    src_shape = src_meta.global_shape
                    tgt_shape = tgt_meta.shape
            else:
                src_shape = src_meta.shape
                tgt_shape = tgt_meta.shape
                if src_shape is None:
                    # an earlier error has occurred, so skip
                    return

                if tgt_dist:  # serial --> dist
                    # src_inds_shape = (shape_to_len(src_shape) * model.comm.size,)
                    if not src_inds_list:
                        # we can automatically add src_indices to match up this distributed target
                        # with the serial source
                        if tgt_shape is not None and src_shape == tgt_meta.global_shape:
                            offset, sz = self.get_dist_offset(tgt_node, model.comm.rank, False)
                            src_indices = \
                                indexer(slice(offset, offset + sz),
                                                                    flat_src=False,
                                                                    src_shape=src_inds_shape)

                            path = nx.shortest_path(self, src_node, tgt_node)
                            self.edges[(path[-2], tgt_node)]['src_indices'] = src_indices
                            tgt_meta.src_inds_list = src_inds_list = [src_indices]
                else:  # serial --> serial
                    pass

            try:
                for i, src_inds in enumerate(src_inds_list):
                    if i == 0:
                        src_inds.set_src_shape(src_inds_shape)
                    else:
                        src_inds.set_src_shape(src_shape)
                    src_shape = src_inds.indexed_src_shape
            except Exception as err:
                if not src_meta.dyn_shape: # if dyn_shape, error is already collected
                    self._collect_error(f"Error in connection between '{src_node[1]}' and "
                                        f"'{tgt_node[1]}': {err}")
                return

            self.check_src_to_tgt_indirect(model, src_node, tgt_node, src_shape, tgt_shape)

    def resolve_output_to_output_down(self, parent, child):
        """
        Resolve metadata for a target output node based on the metadata of a parent output node.

        Parameters
        ----------
        model : Group
            The model.
        parent : tuple of the form ('o', name)
            The source output node.
        child : tuple of the form ('i' or 'o', name)
            The child output node.
        """
        child_meta = self.nodes[child]['attrs']
        parent_meta = self.nodes[parent]['attrs']

        if parent_meta['discrete']:
            for key in _discrete_copy_meta:
                setattr(child_meta, key, getattr(parent_meta, key))
        else:
            for key in _continuous_copy_meta:
                setattr(child_meta, key, getattr(parent_meta, key))
            if parent_meta.distributed:
                shapes = self.get_dist_shapes()
                shapes[child] = shapes[parent]

    def resolve_input_to_input_down(self, model, parent, child, auto):
        """
        Resolve a connection between two input nodes.

        Parameters
        ----------
        model : Model
            The model.
        parent : tuple of the form ('i', name)
            The parent input node.
        child : tuple of the form ('i', name)
            The child node.
        auto : bool
            Whether the source node of the connection tree is an auto_ivc node.
        """
        child_meta = self.nodes[child]['attrs']
        parent_meta = self.nodes[parent]['attrs']

        val = None

        if parent_meta.discrete:
            pass
        else:  # continuous parent
            shape = None
            units = None

            val, shape, units = self.get_parent_val_shape_units(parent, child)

            if val is not None:
                if child_meta.val is not None and not child_meta.shape_by_conn:
                    if not are_compatible_values(val, child_meta.val, parent_meta.discrete):
                        raise ConnError(self.value_error(False, parent, child, val, child_meta.val))

            if units is not None:
                if child_meta.units is not None:
                    if not is_compatible(child_meta.units, units):
                        raise ConnError(self.units_error(False, parent, child, child_meta.units,
                                                         units))
                elif child_meta.units is None and auto:
                    child_meta.units = units

            if shape is None:
                if val is not None:
                    shape = np.shape(val)

            if shape is not None:
                if child_meta.shape is not None:
                        if not array_connection_compatible(shape, child_meta.shape):
                            raise ConnError(self.shape_error(parent, child, shape,
                                                             child_meta.shape))
                else:
                    child_meta.shape = shape

        if val is not None and shape is not None:
            val = np.reshape(val, shape)

        if auto and val is not None and child_meta.val is None:
            child_meta.val = val

    def resolve_down(self, model, parent, child, auto):
        """
        Update the child node's metadata based on the parent node's metadata.

        Parameters
        ----------
        model : Model
            The model.
        parent : tuple
            The parent node.
        child : tuple
            The target node.
        auto : bool
            Whether the source node of the connection tree is an auto_ivc node.
        """
        try:
            if parent[0] == 'o':
                if child[0] == 'i':
                    return self.resolve_output_input_connection(model, parent, child)
                else:
                    self.resolve_output_to_output_down(parent, child)
            else:
                self.resolve_input_to_input_down(model, parent, child, auto)

        except Exception as err:
            self.handle_error(False, parent, child, exc=err)

        return True

    def resolve_conn_tree(self, model, src_node):
        """
        Resolve the connection tree rooted at src_node.

        Metadata is first propagated up the tree from the absolute input nodes up to the root
        input node.  For auto_ivc rooted trees, the propagation continues to the root auto_ivc node.
        Then, checking of compatability between nodes is performed from parent to child down the
        tree, and in some cases metadata is propagated down the tree as well.

        Parameters
        ----------
        model : Model
            The model.
        src_node : tuple
            The source node. This is always an absolute output node.
        """
        if src_node in self._resolved:
            return

        nodes = self.nodes
        auto = src_node[1].startswith('_auto_ivc.')
        src_meta = nodes[src_node]['attrs']
        dynamic = src_meta.dynamic
        src_dist = src_meta.distributed
        first_pass = self._first_pass
        has_dist = False

        if src_dist:
            self._dist_nodes.add(src_node)
            has_dist = True

        dist_nodes = self._dist_nodes
        abs2meta_in = model._var_allprocs_abs2meta['input']

        # first, resolve inputs up from the bottom of the tree to the root input node.
        for node in dfs_postorder_nodes(self, src_node):
            if src_dist and first_pass and node[1] in abs2meta_in:
                self._dist_nodes.update(nx.shortest_path(self, src_node, node))
            if node[0] == 'i':
                node_meta = nodes[node]['attrs']
                if node_meta.dynamic:
                    # if tree contains no dynamic nodes then it's resolved after the first pass
                    dynamic = True
                if first_pass and node[1] in abs2meta_in:
                    if not src_dist and node_meta.distributed:
                        has_dist = True
                        self._dist_nodes.update(nx.shortest_path(self, src_node, node))
                        if auto:
                            self._dist_nodes.add(src_node)

                self.resolve_from_children(model, src_node, node, auto=auto)

        # resolve auto_ivc node  (these are never promoted so there is always only one output node)
        if auto:
            # if under MPI, check for ambiguity of value in root input node
            if model.comm.size > 1:
                leaves = list(self.leaf_input_iter(src_node))
                if len(leaves) > 1:
                    for src_node, tgt_node in dfs_edges(self, src_node):
                        if tgt_node[0] == 'i':
                            # root input
                            inp_meta = nodes[tgt_node]['attrs']
                            if inp_meta.ambiguous_units:
                                break
                            if inp_meta.remote:
                                val = None
                            elif inp_meta.discrete:
                                val = inp_meta.val
                            else:  # local and numerical
                                if isinstance(inp_meta.val, np.ndarray):
                                    val = array_hash(inp_meta.val)
                                else:
                                    val = inp_meta.val

                            ambig = False
                            tups = model.comm.gather((val, inp_meta.remote), root=0)
                            if model.comm.rank == 0:
                                start = None
                                for v, remote in tups:
                                    if remote:
                                        continue
                                    if start is None:
                                        start = v
                                    else:
                                        if start != v:
                                            ambig = True
                                            break
                                model.comm.bcast(ambig, root=0)
                            else:
                                ambig = model.comm.bcast(None, root=0)

                            if ambig:
                                inps = sorted([l for _, l in leaves])
                                model._collect_error(
                                    f"The inputs {inps}, promoted to '{tgt_node[1]}' have "
                                    f"different values, so the value of '{tgt_node[1]}' is "
                                    "ambiguous. Call model.set_input_defaults('"
                                    f"{self.top_name(tgt_node)}', val=?) to remove the ambiguity.")

                            break

            self.resolve_from_children(model, src_node, src_node, auto=auto)

            if src_meta.val is None:
                if src_meta.shape is not None:
                    src_meta.val = np.ones(src_meta.shape)

        # now try filling in any missing metadata going down the tree.  This can happen if
        # for example there are src_indices that block propagation of val and shape below a node
        # where shape or val has been set by set_input_defaults.  This will also fill in
        # missing metadata for promoted output nodes, and it can also set shapes for
        # shape_by_conn inputs.  Also perform compatibility checks between parent and child nodes.
        if not (dynamic and first_pass):
            for edge in dfs_edges(self, src_node):
                u, v = edge
                if v[0] == 'i' and u in dist_nodes and v in dist_nodes:
                    # if the edge is along a path between a src and leaf node that are dist-dist,
                    # dist-serial, or serial-dist, then we need to check the compatibility of the
                    # connection taking distributed shapes into account.
                    continue
                else:
                    self.resolve_down(model, u, v, auto)

            if has_dist:
                self.check_dist_connection(model, src_node)

        if first_pass and not dynamic:
            self._resolved.add(src_node)

    def add_auto_ivc_nodes(self, model):
        """Add auto IVC nodes.

        Parameters
        ----------
        model : any
            model.

        Returns
        -------
        any
            Returned value.
        """
        assert model.pathname == ''
        nodes = self.nodes
        in_degree = self.in_degree

        # this occurs before the auto_ivc variables actually exist
        dangling_inputs = [n for n in nodes() if n[0] == 'i' and in_degree(n) == 0]

        # because we can have manual connection to input nodes other than the 'root' input node,
        # we have to traverse them all to make sure there aren't any outputs connected anywhere
        # in the input tree..
        skip = set()
        for d in dangling_inputs:
            for _, v in dfs_edges(self, d):
                # if v has any output preds, then this input tree is not dangling
                for p in self.predecessors(v):
                    if p[0] == 'o':
                        skip.add(d)
                        break
                if d in skip:
                    break

                if v in self._mult_inconn_nodes:
                    skip.add(d)
                    break

        if skip:
            dangling_inputs = [d for d in dangling_inputs if d not in skip]

        auto_nodes = []
        for i, n in enumerate(dangling_inputs):
            auto_node, _ = self.get_node_attrs(model, f'_auto_ivc.v{i}', 'output')
            self.add_edge(auto_node, n, type='manual')
            auto_nodes.append(auto_node)

        return auto_nodes

    @collect_errors
    def check(self, model):
        """Check.

        Parameters
        ----------
        model : any
            model.
        """
        nodes = self.nodes
        in_degree = self.in_degree
        for abs_out in self._var_allprocs_abs2meta['output']:
            node = ('o', abs_out)
            if node[0] == 'o' and in_degree(node) == 0:  # a root output node

                auto = node[1].startswith('_auto_ivc.')

                for u, v in dfs_edges(self, node):
                    uunits = nodes[u]['attrs'].units
                    vmeta = nodes[v]['attrs']
                    if auto and vmeta.distributed:
                        for _, t in self._bad_conns:
                            if t[1] == v[1]:
                                break
                        else:
                            iroot = self.input_root(v)
                            if iroot != v:
                                promas = f", promoted as '{iroot[1]}',"
                            else:
                                promas = ""
                            self._collect_error(f"Distributed input '{v[1]}'{promas} is not "
                                                "connected. Declare an IndepVarComp and connect it "
                                                "to this input to eliminate this error.",
                                                ident=(u, v))
                            continue

                    if u[0] == 'o' and v[0] == 'i':  # an output to input connection
                        vunits = vmeta.units
                        if uunits is None or vunits is None:
                            uunitless = _is_unitless(uunits)
                            vunitless = _is_unitless(vunits)
                            if uunitless and not vunitless:
                                issue_warning(f"{model.msginfo}: Input '{v[1]}' with units of "
                                            f"'{vunits}' is connected to output '{u[1]}' "
                                            f"which has no units.", category=UnitsWarning)
                            elif not uunitless and vunitless:
                                if not nodes[v]['attrs'].ambiguous_units:
                                    issue_warning(f"{model.msginfo}: Output '{u[1]}' with units of "
                                                f"'{uunits}' is connected to input '{v[1]}' "
                                                f"which has no units.", category=UnitsWarning)

        desvars = model.get_design_vars()
        for req in self._required_conns:
            node = ('i', req)
            root = self.get_root(node)
            if root[1].startswith('_auto_ivc.'):
                iroot = self.input_root(node)
                if iroot[1] in desvars:  # design vars are 'connected'
                    continue

                if iroot == node:
                    promstr = ''
                else:
                    promstr = f", promoted as '{iroot[1]}',"
                self._collect_error(f"{self.msginfo}: Input '{req}'{promstr} requires a "
                                    f"connection but is not connected.")

    def add_implicit_connections(self, model, implicit_conn_vars):
        # implicit connections are added after all promotions are added, so any implicitly connected
        # nodes are guaranteed to already exist in the graph.
        """Add implicit connections.

        Parameters
        ----------
        model : any
            model.
        implicit_conn_vars : any
            implicit conn vars.
        """
        for prom_name in implicit_conn_vars:
            self.check_add_edge(model, ('o', prom_name), ('i', prom_name), type='implicit')

    def update_src_inds_lists(self, model):
        # propagate src_indices down the tree, but don't update shapes because we don't
        # know all of the shapes at the root and leaves of the tree yet.

        # Also, determine the list of dangling promoted inputs for later processing.
        """Update source indices lists.

        Parameters
        ----------
        model : any
            model.
        """
        edges = self.edges
        nodes = self.nodes
        self._dangling_prom_inputs = dangling = set()

        for node in self.nodes():
            if self.in_degree(node) == 0:
                if node[0] == 'o':
                    for u, v in dfs_edges(self, node):
                        if v[0] == 'i':
                            edge_meta = edges[u, v]
                            src_inds = edge_meta.get('src_indices', None)
                            src_inds_list = nodes[u]['attrs'].src_inds_list
                            if src_inds is not None:
                                src_inds_list = src_inds_list.copy()
                                src_inds_list.append(src_inds)

                            nodes[v]['attrs'].src_inds_list = src_inds_list
                else:
                    dangling.add(node)

    def get_anchored_input_node(self, node):
        """Get anchored input node.

        Parameters
        ----------
        node : any
            node.

        Returns
        -------
        any
            Returned value.
        """
        for n in self.bfs_down_iter(node, include_self=False):
            for p in self.predecessors(n):
                if p[0] == 'o':  # n is the src attachment point
                    return n, p
        return None, None

    def update_dangling_prom_inputs(self, model):
        # input nodes promoted above the source node attachment point of their tree, so finding
        # the root and resolving values is more complicated.  If there are no
        # src_indices between these nodes and their corresponding src attachment point then we
        # can treat them as equivalent to their src attachement point.
        """Update dangling promoted inputs.

        Parameters
        ----------
        model : any
            model.
        """
        edges = self.edges
        nodes = self.nodes
        first_pass = self._first_pass

        for d in self._dangling_prom_inputs:
            if d in self._resolved:
                continue

            anchor, src_node = self.get_anchored_input_node(d)
            if anchor is not None:
                src_inds_list = self.nodes[anchor]['attrs'].src_inds_list
                path = nx.shortest_path(self, d, anchor)
                for i in range(1, len(path)):
                    src_indices = edges[path[i - 1], path[i]].get('src_indices', None)
                    if src_indices is not None:
                        break
                else:
                    # no src_indices found so we can store src_inds_list of the src
                    # attachement node
                    donodes = path[:-1]  # all but the anchor node
                    donodes = donodes[::-1]  # reverse order
                    for pnode in donodes:
                        nodes[pnode]['attrs'].src_inds_list = src_inds_list
                        self.resolve_from_children(model, src_node, pnode)

                    src_meta = nodes[src_node]['attrs']
                    if first_pass and not src_meta.dynamic:
                        # see if leaf node is dynamic
                        for leaf in self.leaf_input_iter(d):
                            if nodes[leaf]['attrs'].dynamic:
                                break
                        else:
                            self._resolved.add(d)

    def transform_input_input_connections(self, model):
        """
        Transform input-to-input connections into input-to-output connections.

        Parameters
        ----------
        model : Group
            The top level group.
        """
        for inp_src, tgt in self._input_input_conns:
            root = self.get_root(inp_src)
            edge_meta = self.edges[inp_src, tgt]
            self.remove_edge(inp_src, tgt)
            self.add_edge(root, tgt, **edge_meta)

            tgt_syspath, tgt_prom = self.get_path_prom(tgt)
            if tgt_syspath:
                tgt_prom = model._resolver.abs2prom(self.absnames(tgt)[0], 'input')
            del model._manual_connections[tgt_prom]

            _, abs_out = root
            if abs_out.startswith('_auto_ivc.'):
                src_prom = abs_out
            else:
                src_prom = model._resolver.abs2prom(abs_out, 'output')

            _, inp_src_prom = self.get_path_prom(inp_src)
            inp_src_prom = model._resolver.abs2prom(self.absnames(inp_src)[0], 'input')

            model._manual_connections[inp_src_prom] = (src_prom, edge_meta.get('src_indices', None),
                                                       edge_meta.get('flat_src_indices', None))

    def create_all_conns_dict(self, model):
        """
        Create a dict of global connections 'owned' by a group, keyed by the group's pathname.

        This should only be called on the top level group.

        Parameters
        ----------
        model : Group
            The top level group.

        Returns
        -------
        dict
            A dict of connections 'owned' by a group, of the form
            {group_pathname: {abs_in: abs_out}}.
        """
        assert model.pathname == ''

        conns = {}
        abs_ins = model._var_allprocs_abs2meta['input']
        discrete_ins = model._var_allprocs_discrete['input']
        global_conns = {}
        for abs_out in  model._resolver.abs_iter('output'):
            node = ('o', abs_out)
            for _, abs_in in self.leaf_input_iter(node):
                assert abs_in in abs_ins or abs_in in discrete_ins
                if abs_in in global_conns:
                    global_conns[abs_in].append(abs_out)
                else:
                    global_conns[abs_in] = [abs_out]
                common = common_subpath((abs_out, abs_in))
                if common not in conns:
                    conns[common] = {}
                conns[common][abs_in] = abs_out

        multiple_conns = []
        for abs_in, abs_outs in global_conns.items():
            if len(abs_outs) > 1:
                multiple_conns.append((abs_in, abs_outs))

        if multiple_conns:
            msg = []
            for abs_in, abs_outs in multiple_conns:
                msg.append(f"'{abs_in}' from {abs_outs}")

            self._collect_error(f"{self.msginfo}: The following inputs have multiple connections:"
                                f" {', '.join(msg)}.")

        return conns

    def get_root(self, node):
        """Get root.

        Parameters
        ----------
        node : any
            node.

        Returns
        -------
        any
            Returned value.
        """
        in_degree = self.in_degree
        for n in self.bfs_up_iter(node):
            if n[0] == 'o' and in_degree(n) == 0:
                return n

        # if we get here, node is an input promoted above the attachment point of its source output
        for n in self.bfs_down_iter(node, include_self=False):
            # look for output predecessors
            for p in self.predecessors(n):
                if p[0] == 'o':
                    return self.get_root(p)

    def get_tree_iter(self, node):
        """
        For any node in the graph, yield a generator of all nodes in the tree the node belongs to.

        The nodes are yielded in depth-first preorder.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node to get the tree for.

        Yields
        ------
        tuple of the form (node, depth)
            The node and its depth.
        """
        root = self.get_root(node)
        if root is None:
            if node[0] == 'i':
                # may have a manual connection before a late promotion, so
                # traverse down the tree to the end, then back to the root from there
                for n in self.leaf_input_iter(node):
                    root = self.get_root(n)
                    break
            if root is None:
                raise RuntimeError(f"Can't find root of connection tree from node '{node}'.")

        dq = deque([(root, 0)])

        while dq:
            node, depth = dq.popleft()
            yield (node, depth)
            next_depth = depth + 1

            for child in self.successors(node):
                dq.appendleft((child, next_depth))

    def leaf_input_iter(self, node):
        # we may already be a leaf node
        """Leaf input iter.

        Parameters
        ----------
        node : any
            node.

        Yields
        ------
        any
            Yielded value.
        """
        if node[0] == 'i' and self.out_degree(node) == 0:
            yield node
        else:
            out_degree = self.out_degree
            for _, node in dfs_edges(self, node):
                if node[0] == 'i' and out_degree(node) == 0:
                    yield node

    def leaf_units(self, node):
        """Leaf units.

        Parameters
        ----------
        node : any
            node.

        Returns
        -------
        any
            Returned value.
        """
        return [self.nodes[n]['attrs'].units for n in self.leaf_input_iter(node)]

    def absnames(self, node):
        """Absnames.

        Parameters
        ----------
        node : any
            node.

        Returns
        -------
        any
            Returned value.
        """
        if node[0] == 'i':
            return [n for _, n in self.leaf_input_iter(node)]
        else:
            return [self.get_root(node)[1]]

    def io_conn_iter(self):
        """
        Iterate over all output-input connections in the graph.

        There will be only one of these per connection tree.  A connection tree has an absolute
        output at the root and absolute inputs at the leaves.

        Yields
        ------
        tuple of the form (u, v)
            The connected node pair (output to input).  Each node may be either promoted or
            absolute.
        """
        for u, v in self.edges():
            if u[0] == 'o' and v[0] == 'i':
                yield u, v

    def get_subarray(self, arr, indices_list):
        """
        Apply a sequence of indexing operations to the input array.

        Parameters
        ----------
        arr : numpy.ndarray
            The initial array.
        indices_list : list
            A list of indexing objects (e.g., slices, integers, arrays for advanced indexing).

        Returns
        -------
        subarray : numpy.ndarray
            The result after applying all indexing operations (may be a view or copy).
        """
        current = np.atleast_1d(arr)
        if indices_list is not None:
            for idx in indices_list:
                current = idx.indexed_val(current)
        return current

    def set_subarray(self, arr, indices_list, val, node):
        """
        Set the val into the positions of the original array based on indices_list.

        This function handles both views and copies by propagating changes back through the chain.

        Parameters
        ----------
        arr : numpy.ndarray
            The original array to modify.
        indices_list : list
            A list of indexing objects (e.g., slices, integers, arrays for advanced indexing).
        val : numpy.ndarray or compatible
            The val to set (must match the shape of the final subarray).
        node : tuple
            The node to set the val for.

        Raises
        ------
        ValueError
            If the shape of val does not match the final subarray shape.
        """
        msg = ''
        try:
            if indices_list:
                chain = [arr]
                for idx in indices_list:
                    chain.append(idx.indexed_val(chain[-1]))

                if np.shape(val) != () and np.squeeze(val).shape != np.squeeze(chain[-1]).shape:
                    msg = (f"Value shape {np.squeeze(val).shape} does not match shape "
                           f"{np.squeeze(chain[-1]).shape} of the destination")
            else:
                try:
                    arr[:] = val
                except ValueError:
                    arr[:] = val.reshape(arr.shape)
                return
        except Exception as err:
            msg = str(err)

        if msg:
            raise ValueError(f"Failed to set value of '{node[1]}': {msg}.")

        last = chain[-1]
        if (isinstance(last, np.ndarray) and last.ndim == 0) or np.isscalar(last):
            chain[-1] = val
        else:
            last[:] = val

        for i in range(len(chain) - 2, -1, -1):
            sub = chain[i + 1]
            prev = chain[i]
            idx = indices_list[i]
            if sub.base is not prev:
                idx.indexed_val_set(prev, sub)

    def get_src_index_array(self, abs_in):
        """Get source index array.

        Parameters
        ----------
        abs_in : any
            absolute in.

        Returns
        -------
        any
            Returned value.
        """
        node = ('i', abs_in)
        if node not in self:
            raise ValueError(f"Input '{abs_in}' not found.")
        src_inds_list = self.nodes[node]['attrs'].src_inds_list
        if not src_inds_list:
            return None
        elif len(src_inds_list) == 1:
            return src_inds_list[0].shaped_array()
        else:
            root = self.get_root(node)
            root_meta = self.nodes[root]['attrs']
            if root_meta.distributed:
                root_shape = root_meta.global_shape
            else:
                root_shape = root_meta.shape
            arr = np.arange(shape_to_len(root_shape)).reshape(root_shape)
            for inds in src_inds_list:
                arr = inds.indexed_val(arr)
            return arr

    def convert_get(self, node, val, src_units, tgt_units, src_inds_list=(), units=None,
                    indices=None, get_remote=False):
        """Convert value for get.

        Parameters
        ----------
        node : any
            node.
        val : any
            val.
        src_units : any
            source units.
        tgt_units : any
            target units.
        src_inds_list : any
            source indices list.
        units : any
            units.
        indices : any
            indices.
        get_remote : any
            get remote.

        Returns
        -------
        any
            Returned value.
        """
        node_meta = self.nodes[node]['attrs']

        if not node_meta.discrete:
            val = np.asarray(val)
            if src_inds_list:
                val = self.get_subarray(val, src_inds_list)

            if get_remote and not src_inds_list:
                val = val.reshape(node_meta.global_shape)
            else:
                val = val.reshape(node_meta.shape)

        if indices:
            val = self.get_subarray(val, [indices])

        if units is None:
            units = tgt_units

        if units is not None:
            if src_units is None:
                raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                f"'{units}'.")
            elif src_units != units:
                try:
                    scale, offset = unit_conversion(src_units, units)
                except Exception:
                    raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                    f"'{units}'.")

                return (val + offset) * scale

        return val

    def convert_set(self, val, src_units, tgt_units, src_inds_list=(), units=None, indices=None):
        """Convert value for set.

        Parameters
        ----------
        val : any
            val.
        src_units : any
            source units.
        tgt_units : any
            target units.
        src_inds_list : any
            source indices list.
        units : any
            units.
        indices : any
            indices.

        Returns
        -------
        any
            Returned value.
        """
        if indices:
            src_inds_list = list(src_inds_list) + [indices]

        val = self.get_subarray(val, src_inds_list)

        if units is None:
            units = tgt_units

        if units is not None:
            if src_units is None:
                raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                f"'{units}'.")
            elif src_units != units:
                try:
                    scale, offset = unit_conversion(units, src_units)
                except Exception:
                    raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                    f"'{units}'.")

                return (val + offset) * scale

        return val

    def setup_global_connections(self, model):
        """
        Compute dict of all connections between inputs and outputs.
        """
        from openmdao.core.group import Group

        # add nodes for all absolute inputs and connected absolute outputs
        self.add_variable_meta(model)

        systems = list(model.system_iter(include_self=True, recurse=True))
        groups = [s for s in systems if isinstance(s, Group)]
        for g in groups:
            self.add_manual_connections(g)

        for g in groups:
            self.add_group_input_defaults(g)

        if model.comm.size > 1:
            self.gather_data(model)

        self.add_implicit_connections(model, model._get_implicit_connections())

        # check for cycles
        if not nx.is_directed_acyclic_graph(self):
            cycle_edges = nx.find_cycle(self, orientation='original')
            errmsg = '\n'.join([f'     {edge[0]} ---> {edge[1]}'
                                for edge in cycle_edges])
            self._collect_error('Cycle detected in input-to-input connections. '
                                f'This is not allowed.\n{errmsg}')

        self.add_auto_ivc_nodes(model)
        self.update_src_inds_lists(model)

        model._setup_auto_ivcs()

        self._var_allprocs_abs2meta = model._var_allprocs_abs2meta
        self._var_allprocs_discrete = model._var_allprocs_discrete
        self._var_allprocs_abs2idx = model._var_allprocs_abs2idx
        self._var_abs2meta = model._var_abs2meta

        # update global shapes for distributed vars, now that data structures include
        # auto_ivcs
        for dnode in self._distributed_nodes:
            node_meta = self.nodes[dnode]['attrs']
            if not node_meta.dyn_shape:
                node_meta.global_shape = self.compute_global_shape(dnode)

        self.update_all_node_meta(model)
        self.transform_input_input_connections(model)

        conn_dict = self.create_all_conns_dict(model)

        global_conn_dict = {'': {}}
        root_dict = global_conn_dict['']
        for path, conn_data in conn_dict.items():
            for name in all_ancestors(path):
                if name in global_conn_dict:
                    global_conn_dict[name].update(conn_data)
                else:
                    global_conn_dict[name] = conn_data.copy()

            # don't forget the root path!
            root_dict.update(conn_data)

        for system in systems:
            if isinstance(system, Group):
                system._conn_abs_in2out = conn_dict.get(system.pathname, {})
                system._conn_global_abs_in2out = global_conn_dict.get(system.pathname, {})

            system._resolver._conns = model._conn_global_abs_in2out

        # model._setup_var_sizes()

    def create_node_label(self, node):
        """Create the label for a displayed node.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node.

        Returns
        -------
        str
            Returned html for the node label.
        """
        def get_table_row(name, meta, mods=(), align='LEFT', max_width=None, show_always=False):
            """
            Get the html for a table row.

            Parameters
            ----------
            name : str
                name.
            meta : NodeAttrs
                The metadata for the node.
            mods : list of str
                Modifiers for the html.
            align : any
                The alignment of the content.
            max_width : any
                The maximum width of the content.
            show_always : any
                Whether to show the content always.

            Returns
            -------
            str
                The html for the table row.
            """
            if '.' in name:
                parent, _, child = name.rpartition('.')
                meta = getattr(meta, parent)
                content = getattr(meta, child)
            elif isinstance(meta, dict):
                content = meta[name]
            else:
                content = getattr(meta, name)

                ambig = getattr(meta, f"ambiguous_{name}")
                if ambig:
                    content = '?'

            if (content is None or content is False) and not show_always:
                return ''

            if max_width is not None:
                content = truncate_str(content, max_width)

            if mods:
                starts = []
                ends = []
                for mod in mods:
                    starts.append(f"<{mod}>")
                    ends.append(f"</{mod}>")

                content = ''.join(starts) + content + ''.join(ends)

            content = f"<b>{name}:</b> {content}"

            return \
                f"<TR><TD ALIGN=\"{align}\"><FONT POINT-SIZE=\"10\">{content}</FONT></TD></TR>"

        name = node[1]
        meta = self.nodes[node]['attrs']
        rows = []
        rows.append(get_table_row('units', meta, ('i',)))
        rows.append(get_table_row('shape', meta))
        rows.append(get_table_row('val', meta, max_width=max(30, int(len(name) * 1.2))))
        rows.append(get_table_row('defaults.val', meta, max_width=max(30, int(len(name) * 1.2))))
        rows.append(get_table_row('defaults.src_shape', meta))
        rows.append(get_table_row('defaults.units', meta))
        rows.append(get_table_row('src_inds_list', meta))
        rows.append(get_table_row('shape_by_conn', meta))
        rows.append(get_table_row('copy_shape', meta))

        # dot doesn't like node labels containing functions, so just show yes if set
        rows.append(get_table_row('compute_shape',
                                  {'compute_shape': 'yes' if meta.compute_shape else None}))
        rows.append(get_table_row('compute_units',
                                  {'compute_units': 'yes' if meta.compute_units else None}))

        rows.append(get_table_row('units_by_conn', meta))
        rows.append(get_table_row('copy_units', meta))
        rows.append(get_table_row('discrete', meta))
        rows.append(get_table_row('distributed', meta, show_always=True))
        rows.append(get_table_row('remote', meta))
        rows = [r for r in rows if r]

        if rows:
            combined = '\n'.join(rows)
        else:
            combined = ''

        return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="0"><TR><TD ' \
            f' ALIGN=\"LEFT\"><FONT POINT-SIZE=\"12\">' \
            f'<b>{name}</b></FONT></TD></TR>{combined}</TABLE>>'

    def drawable_node_iter(self, pathname=''):
        """
        Yield nodes usable in a pydot graph.

        Parameters
        ----------
        pathname : str
            The pathname of the group to draw.

        Yields
        ------
        tuple of the form (node, data)
            The node and its metadata.
        """
        if pathname:
            pathname = pathname + '.'

        for node, data in self.nodes(data=True):
            meta = data['attrs']
            if pathname and not self.startswith(pathname, node):
                continue
            newdata = {}
            if node[0] == 'i':
                newdata['fillcolor'] = GRAPH_COLORS['input']
            else:
                newdata['fillcolor'] = GRAPH_COLORS['output']

            if meta.ambiguous:
                newdata['color'] = GRAPH_COLORS['ambiguous']
                newdata['penwidth'] = '4'  # Thick border

            newdata['label'] = self.create_node_label(node)
            newdata['tooltip'] = (meta.pathname, meta.rel_name)
            newdata['style'] = 'filled,rounded'
            newdata['shape'] = 'box'  # Use box shape with rounded corners
            newdata['pathname'] = meta.pathname
            newdata['rel_name'] = meta.rel_name
            yield node, newdata

    def drawable_edge_iter(self, pathname='', show_cross_boundary=True, max_width=50):
        """
        Yield edges usable in a pydot graph.

        Parameters
        ----------
        pathname : str, optional
            The pathname of the group to draw.
        show_cross_boundary : bool, optional
            Whether to show cross boundary connections.
        max_width : int or None
            The maximum allowable width of an edge label.

        Yields
        ------
        tuple of the form (u, v, data)
            The edge and its metadata.
        """
        if pathname:
            pathname = pathname + '.'

        # None means the edge is a promotion
        type_map = {None: 'dashed', 'manual': None, 'implicit': 'dotted'}

        for u, v, data in self.edges(data=True):
            style = type_map[data.get('type')]

            if pathname:
                u_internal = self.startswith(pathname, u)
                if not u_internal and not show_cross_boundary:
                    continue

                v_internal = self.startswith(pathname, v)
                if not v_internal and not show_cross_boundary:
                    continue

                if not (u_internal or v_internal):
                    continue

            newdata = {}
            if style:
                newdata['style'] = style

            if u[0] == 'i' and v[0] == 'i':  # show promotion arrows in the right direction
                edge = (v, u)
                newdata['dir'] = 'back'  # show arrows going backwards w/o messing up tree layout
            else:
                edge = (u, v)

            src_indices = data.get('src_indices')
            if src_indices is None:
                newdata['tooltip'] = f"{edge[0]} -> {edge[1]}"
            else:
                newdata['label'] = truncate_str(src_indices, max_len=max_width)
                newdata['tooltip'] = f"{edge[0]} -> {edge[1]}: src_indices: {src_indices}"

            yield u, v, newdata

    def get_drawable_graph(self, pathname='', varname=None, show_cross_boundary=True):
        """
        Display the connection graph.

        The collection graph is a collection of connection trees.  A connection tree has an absolute
        output at the root and absolute inputs at the leaves, with any intermediate nodes being
        promoted inputs and outputs.

        Parameters
        ----------
        pathname : str
            The pathname of the system to display.
        varname : str or None
            Display the connection tree associated with this variable.  If None, display the entire
            collection graph.
        show_cross_boundary : bool
            Whether to show cross boundary connections.
        """
        G = nx.DiGraph()

        if pathname:
            # special handling for cross boundary connections
            edges = list(self.drawable_edge_iter(pathname, show_cross_boundary))
            draw_nodes = list(self.drawable_node_iter(pathname))

            if show_cross_boundary:
                visible_nodes = {n for n, _, _ in edges}
                visible_nodes.update({n for _, n, _ in edges})
                outside_nodes = visible_nodes.difference([n for n, _ in draw_nodes])

                nodes = self.nodes
                for node in outside_nodes:
                    node_meta = nodes[node]['attrs']
                    meta = {
                        'label': self.create_node_label(node),
                        'tooltip': (node_meta['pathname'], node_meta['rel_name']),
                        'style': 'filled,rounded',
                        'shape': 'box',
                        'pathname': node_meta['pathname'],
                        'rel_name': node_meta['rel_name'],
                        'fillcolor': GRAPH_COLORS['boundary'],
                    }
                    draw_nodes.append((node, meta))

            for node, data in draw_nodes:
                G.add_node(node, **data)
            for u, v, data in edges:
                G.add_edge(u, v, **data)
        else:
            for node, data in self.drawable_node_iter(pathname):
                G.add_node(node, **data)
            for u, v, data in self.drawable_edge_iter(pathname):
                G.add_edge(u, v, **data)

        if varname:
            varnode = self.find_node(pathname, varname)
            G.nodes[varnode]['color'] = GRAPH_COLORS['highlight']
            G.nodes[varnode]['penwidth'] = '4'  # Thick border
            tree = nx.node_connected_component(G.to_undirected(as_view=True), varnode)
            G = nx.subgraph(G, tree)

        replace = {}
        for node in G.nodes():
            # quote node names containing certain characters for use in dot
            _, name = node
            if ':' in name or '<' in name:
                replace[node] = f'"{node}"'

        if replace:
            G = nx.relabel_nodes(G, replace)

        return G

    def get_pydot_graph(self, pathname='', varname=None, show_cross_boundary=True):
        """Get pydot graph.

        Parameters
        ----------
        pathname : any
            pathname.
        varname : any
            varname.
        show_cross_boundary : any
            show cross boundary.

        Returns
        -------
        any
            Returned value.
        """
        return nx.drawing.nx_pydot.to_pydot(self.get_drawable_graph(pathname, varname,
                                                                    show_cross_boundary))

    def get_dot(self, pathname='', varname=None, show_cross_boundary=True):
        """Get DOT.

        Parameters
        ----------
        pathname : any
            pathname.
        varname : any
            varname.
        show_cross_boundary : any
            show cross boundary.

        Returns
        -------
        any
            Returned value.
        """
        return self.get_pydot_graph(pathname, varname, show_cross_boundary).to_string()

    def get_svg(self, pathname='', varname=None, show_cross_boundary=True):
        """Get SVG.

        Parameters
        ----------
        pathname : any
            pathname.
        varname : any
            varname.
        show_cross_boundary : any
            show cross boundary.

        Returns
        -------
        any
            Returned value.
        """
        return self.get_pydot_graph(pathname, varname,
                                    show_cross_boundary).create_svg().decode('utf-8')

    def display(self, pathname='', varname=None, show_cross_boundary=True, outfile=None):
        """Display.

        Parameters
        ----------
        pathname : any
            pathname.
        varname : any
            varname.
        show_cross_boundary : any
            show cross boundary.
        outfile : any
            outfile.
        """
        write_graph(self.get_drawable_graph(pathname, varname, show_cross_boundary),
                    outfile=outfile)

    def print_tree(self, name):
        """Print tree.

        Parameters
        ----------
        name : any
            name.
        """
        if name in self:
            node = name
        else:
            node = ('i', name)
            if node not in self:
                node = ('o', name)
                if node not in self:
                    raise ValueError(f"Variable '{name}' not found in the graph.")

        nodes = self.nodes
        for node, depth in self.get_tree_iter(node):
            indent = '  ' * depth
            meta = nodes[node]['attrs']
            mnames = ['units', 'discrete', 'shape']
            dismeta = {k: meta[k] for k in mnames if k in meta and meta[k] is not None}
            print(f"{indent}{node[1]}  {dismeta}")

    def serve(self, port=None, open_browser=True):
        """Serve connection graph web UI."""
        from openmdao.visualization.conn_graph_ui import ConnGraphHandler
        import socket

        def find_unused_port():
            """Find an unused port starting from 8001."""
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        if port is None:
            port = find_unused_port()

        def handler(*args, **kwargs):
            """
            Return a ConnGraphHandler instance.

            Parameters
            ----------
            *args : list
                Positional arguments passed to the base handler.
            **kwargs : dict
                Keyword arguments passed to the base handler.

            Returns
            -------
            ConnGraphHandler
                A ConnGraphHandler instance.
            """
            return ConnGraphHandler(self, *args, **kwargs)

        print(f"🌐 Starting Connection Graph UI on port {port}")
        print(f"📱 Open your browser to: http://localhost:{port}")

        if open_browser:
            def open_browser():
                time.sleep(1)
                webbrowser.open(f'http://localhost:{port}')

            threading.Thread(target=open_browser, daemon=True).start()

        try:
            with HTTPServer(("", port), handler) as httpd:
                print(f"✅ Server running on http://localhost:{port}")
                print("Press Ctrl+C to stop")
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

    def copy_var_shape(self, from_node, to_node):
        """
        Copy shape info from from_node's metadata to to_node's metadata in the graph.

        Parameters
        ----------
        from_node : tuple
            Tuple containing the IO and name of the variable to copy shape info from.
        to_node : tuple
            Tuple containing the IO and name of the variable to copy shape info to.

        Returns
        -------
        tuple or None
            If the shape of the variable is known, return the shape.
            Otherwise, return None.
        """
        from_conn_meta = self.nodes[from_node]['attrs']
        if from_conn_meta.shape is None:
            return
        to_conn_meta = self.nodes[to_node]['attrs']

        from_io, from_name = from_node
        _, to_name = to_node

        # is this a connection internal to a component?
        internal = to_name.rpartition('.')[0] == from_name.rpartition('.')[0]

        if internal:  # if internal to a component, src_indices isn't used
            fwd = from_io == 'i'
            src_inds_list = None
        else:
            fwd = from_io == 'o'
            if fwd:
                src_inds_list = to_conn_meta.src_inds_list
            else:  # rev
                src_inds_list = from_conn_meta.src_inds_list

        is_full_slice = \
            src_inds_list and len(src_inds_list) == 1 and src_inds_list[0].is_full_slice()

        if self.comm.size > 1:
            dist_from = from_conn_meta.distributed
            dist_to = to_conn_meta.distributed
        else:
            dist_from = dist_to = False

        if fwd:
            dist_src = dist_from
            dist_tgt = dist_to
        else:
            dist_src = dist_to
            dist_tgt = dist_from

        fname = _shape_func_map[(dist_src, dist_tgt, fwd)]
        return getattr(self, fname)(from_node, to_node, src_inds_list, is_full_slice)

    def _get_var_existence(self):
        """
        Get the existence of a all continuous variables across all processes.

        Returns
        -------
        dict of the form {'input': np.ndarray, 'output': np.ndarray}
            A dictionary of variable existence from the model.
        """
        if self._var_existence is None:
            all_abs2meta = self._var_allprocs_abs2meta
            self._var_existence = {
                'input': np.zeros((self.comm.size, len(all_abs2meta['input'])), dtype=bool),
                'output': np.zeros((self.comm.size, len(all_abs2meta['output'])), dtype=bool),
            }

            iproc = self.comm.rank
            for io, existence in self._var_existence.items():
                abs2meta = self._var_abs2meta[io]
                if self.comm.size > 1:
                    for i, name in enumerate(all_abs2meta[io]):
                        if name in abs2meta:
                            existence[iproc, i] = True

                    row = existence[iproc, :].copy()
                    self.comm.Allgather(row, existence)
                else:
                    existence[:] = True

        return self._var_existence

    def serial2serialfwd(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a serial to serial connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : list of Indexers
            The source indices list.
        is_full_slice : any
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        shp = self.nodes[from_node]['attrs'].shape
        if src_inds_list:
            for inds in src_inds_list:
                inds.set_src_shape(shp)
                shp = inds.indexed_src_shape

        self.nodes[to_node]['attrs'].shape = shp
        return shp

    def serial2serialrev(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a serial to serial reverse connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : any
            The source indices list.
        is_full_slice : bool
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        if not src_inds_list or is_full_slice:
            shp = self.nodes[from_node]['attrs'].shape
        else:
            self._collect_error(f"Input '{from_node[1]}' has src_indices so the shape "
                                f"of connected output '{to_node[1]}' cannot be "
                                "determined.")
            return

        self.nodes[to_node]['attrs'].shape = shp
        return shp

    def serial2distfwd(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a serial to distributed forward connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : any
            The source indices list.
        is_full_slice : bool
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        from_shape = self.nodes[from_node]['attrs'].shape
        existence = self._get_var_existence()
        exist_outs = existence['output'][:, self._var_allprocs_abs2idx[from_node[1]]]
        exist_ins = existence['input'][:, self._var_allprocs_abs2idx[to_node[1]]]
        to_meta = self.nodes[to_node]['attrs']
        num_exist = np.count_nonzero(exist_outs)

        if not src_inds_list:
            shape = from_shape
        else:
            if len(from_shape) <= 1:
                global_src_shape = (num_exist * shape_to_len(from_shape),)
            else:
                shapelst = list(from_shape)
                # stack the shapes across procs
                first_dim = shapelst[0] * num_exist
                shapelst[0] = first_dim
                global_src_shape = tuple(shapelst)

            shp = global_src_shape
            for inds in src_inds_list:
                inds.set_src_shape(shp)
                shp = inds.indexed_src_shape

            shape = shp

        size = shape_to_len(shape)
        sizes = np.zeros(self.comm.size, dtype=int)
        sizes[exist_ins] = size
        dist_shapes = self.get_dist_shapes()
        dist_sizes = self.get_dist_sizes()
        dist_sizes[to_node] = sizes
        dist_shapes[to_node] = [shape if exist_ins[i] else None for i in range(self.comm.size)]
        to_meta.shape = shape
        nexist = len([i for i in exist_ins if i])
        if len(shape) > 1:
            firstdim = shape[0] * nexist
            gshape = (firstdim,) + shape[1:]
        else:
            gshape = (size * nexist,)
        to_meta.global_shape = gshape
        return shape

    def serial2distrev(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a serial to distributed reverse connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : list of Indexers
            The source indices list.
        is_full_slice : bool
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        # serial_out <-- dist_in
        dshapes = self.get_dist_shapes(from_node)
        dshapes = [s for s in dshapes if s is not None and shape_to_len(s) > 0]
        if len(dshapes) >= 1:
            shape0 = dshapes[0]
            for ds in dshapes:
                if ds != shape0:
                    to_io = 'input' if to_node[0] == 'i' else 'output'
                    from_io = 'input' if from_node[0] == 'i' else 'output'
                    dshapes = [_strip_np(s) for s in dshapes]
                    self._collect_error(
                        f"{self.msginfo}: dynamic sizing of non-distributed {to_io} '{to_node[1]}' "
                        f"from distributed {from_io} '{from_node[1]}' is not supported because not "
                        f"all '{from_node[1]}' ranks are the same shape "
                        f"(shapes={dshapes}).", ident=(from_node, to_node))
                    return

            self.nodes[to_node]['attrs'].shape = shape0
            return shape0

    def dist2serialfwd(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a distributed to serial forward connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : list of Indexers
            The source indices list.
        is_full_slice : bool
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        if src_inds_list:
            shp = self.compute_global_shape(from_node)
            for src_indices in src_inds_list:
                src_indices.set_src_shape(shp)
                shp = src_indices.indexed_src_shape

            self.nodes[to_node]['attrs'].shape = shp
            return shp
        else:
            # We don't allow this case because
            # serial variables must have the same value on all procs and the only way
            # this is possible is if the src_indices on each proc are identical, but that's not
            # possible if we assume 'always local' transfer (see POEM 46).
            self._collect_error(
                f"{self.msginfo}: dynamic sizing of non-distributed input '{to_node[1]}' "
                f"from distributed output '{from_node[1]}' without src_indices is not "
                "supported.")

    def dist2serialrev(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a distributed to serial reverse connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : list of Indexers
            The source indices list.
        is_full_slice : any
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        # dist_out <-- serial_in
        if is_full_slice:
            abs2idx = self._var_allprocs_abs2idx
            # serial input is using full slice here, so contains the full
            # distributed value of the distributed output (and serial input will
            # have the same value on all procs).

            # dist input may not exist on all procs, so distribute the serial
            # entries across only the procs where the dist output exists.
            exist_procs = self._get_var_existence()['output'][:, abs2idx[to_node[1]]]
            split_num = np.count_nonzero(exist_procs)

            dist_shapes = self.get_dist_shapes()
            dist_sizes = self.get_dist_sizes()

            from_shape = self.nodes[from_node]['attrs'].shape
            sz, _ = evenly_distrib_idxs(split_num, shape_to_len(from_shape))
            sizes = np.zeros(self.comm.size, dtype=int)
            sizes[exist_procs] = sz
            dist_sizes[to_node] = sizes.copy()
            to_meta = self.nodes[to_node]['attrs']
            if len(from_shape) > 1:
                if from_shape[0] != self.comm.size:
                    self._collect_error(f"Serial input '{from_node[1]}' has shape "
                                        f"{from_shape} but output '{to_node[1]}' "
                                        f"is distributed over {self.comm.size} "
                                        f"procs and {from_shape[0]} != "
                                        f"{self.comm.size}.")
                    return
                else:
                    dist_shapes[to_node] = [from_shape[1:]] * self.comm.size
                    shp = to_meta.shape = from_shape[1:]
            else:
                dist_shapes[to_node] = [(s,) for s in sizes]
                shp = to_meta.shape = (sizes[sizes != 0][0],)

            to_meta.global_shape = from_shape

            return shp

        else:
            self._collect_error(f"Input '{from_node[1]}' has src_indices so the shape "
                                f"of connected output '{to_node[1]}' cannot be "
                                "determined.")

    def dist2distfwd(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a distributed to distributed forward connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : list of Indexers
            The source indices list.
        is_full_slice : bool
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        if is_full_slice:
            self._collect_error(f"Using a full slice [:] as src_indices between"
                                f" distributed variables '{from_node[1]}' and "
                                f"'{to_node[1]}' is invalid.")
            return

        from_conn_meta = self.nodes[from_node]['attrs']
        to_conn_meta = self.nodes[to_node]['attrs']
        dist_shapes = self.get_dist_shapes()
        dist_sizes = self.get_dist_sizes()

        if not src_inds_list:
            shp = from_conn_meta.shape
            to_conn_meta.shape = shp
            dist_shapes[to_node] = dist_shapes[from_node].copy()
            dist_sizes[to_node] = dist_sizes[from_node].copy()
            to_conn_meta.global_shape = from_conn_meta.global_shape
        else:
            shp = self.compute_global_shape(from_node)
            for src_indices in src_inds_list:
                src_indices.set_src_shape(shp)
                shp = src_indices.indexed_src_shape

            to_conn_meta.shape = shp
            dist_shapes[to_node] = self.comm.allgather(shp)
            dist_sizes[to_node] = np.array([shape_to_len(s) for s in dist_shapes[to_node]])
            to_conn_meta.global_shape = self.compute_global_shape(to_node)

        return shp

    def dist2distrev(self, from_node, to_node, src_inds_list, is_full_slice):
        """
        Compute the shape for a distributed to distributed reverse connection.

        Parameters
        ----------
        from_node : tuple of the form ('i' or 'o', name)
            from node.
        to_node : tuple of the form ('i' or 'o', name)
            to node.
        src_inds_list : list of Indexers
            The source indices list.
        is_full_slice : bool
            Whether the source is a full slice.

        Returns
        -------
        tuple
            The shape.
        """
        if is_full_slice:
            self._collect_error(f"Using a full slice [:] as src_indices between"
                                f" distributed variables '{from_node[1]}' and "
                                f"'{to_node[1]}' is invalid.")
        elif not src_inds_list:
            shp = self.nodes[from_node]['attrs'].shape
            to_conn_meta = self.nodes[to_node]['attrs']
            to_conn_meta.shape = shp
            dist_shapes = self.get_dist_shapes()
            dist_sizes = self.get_dist_sizes()
            dist_shapes[to_node] = dist_shapes[from_node].copy()
            dist_sizes[to_node] = dist_sizes[from_node].copy()
            to_conn_meta.global_shape = self.compute_global_shape(to_node)
            return shp
        else:
            self._collect_error(f"Input '{from_node[1]}' has src_indices so the shape "
                                f"of connected output '{to_node[1]}' cannot be "
                                "determined.")

