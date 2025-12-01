from difflib import get_close_matches
import networkx as nx
from networkx import dfs_edges, dfs_postorder_nodes
import numpy as np
from numpy import isscalar, reshape
from numbers import Number
from collections import deque
from copy import deepcopy
import textwrap

import webbrowser
import threading
import time
from http.server import HTTPServer

from openmdao.visualization.graph_viewer import write_graph
from openmdao.utils.general_utils import common_subpath, is_undefined, truncate_str, all_ancestors
from openmdao.utils.array_utils import array_connection_compatible, shape_to_len
from openmdao.utils.graph_utils import dump_nodes, dump_edges
from openmdao.utils.units import is_compatible
from openmdao.utils.units import unit_conversion
from openmdao.utils.indexer import indexer, Indexer
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.units import _is_unitless, has_val_mismatch
from openmdao.visualization.tables.table_builder import generate_table


# use hex colors here because the using english names was sometimes causing failure to show
# proper colors in the help dialog.
GRAPH_COLORS = {
    'input': 'peachpuff3',
    'output': 'skyblue3',
    'highlight': '#66ff00',
    'ambiguous': '#FF0800',
    'boundary': '#D3D3D3',
}


_continuous_copy_meta = ['val', 'units', 'shape', 'discrete', 'remote', 'distributed',
                         'global_shape']
_discrete_copy_meta = ['val', 'discrete', 'remote']


def is_equal(a, b):
    if not (isinstance(b, type(a)) or isinstance(a, type(b))):
        return False

    if isinstance(a, np.ndarray):
        return a.size == b.size and np.all(np.squeeze(a) == np.squeeze(b))

    return a == b


def are_compatible_values(a, b, discrete, src_indices=None):
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
    __slots__ = ['val', 'units', 'src_shape']
    def __init__(self, val=None, units=None, src_shape=None):
        self.val = val
        self.units = units
        self.src_shape = src_shape

    def __iter__(self):
        yield self.val
        yield self.units
        yield self.src_shape


class NodeAttrs():
    __slots__ = ('pathname', 'rel_name', '_val', '_shape', '_global_shape', '_units', 'defaults',
                 '_src_inds_list', 'flags', '_meta', '_locmeta', 'copy_shape', 'compute_shape',
                 'copy_units', 'compute_units', 'distributed')

    def __init__(self):
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
        self.defaults = Defaults()
        self.copy_shape = None
        self.compute_shape = None
        self.copy_units = None
        self.compute_units = None
        self.distributed = None  # 3 states: None (unknown), False or True

    def __getattr__(self, key):
        return None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def update(self, kwargs):
        # networkx uses this to populate the initial node attributes.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
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
            if self._locmeta is not None:
                self._locmeta['val'] = value
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
                    self._val = reshape(value, self._shape)
                else:
                    self._val[:] = reshape(value, self._shape)

            if self._locmeta is not None:
                self._locmeta['val'] = self._val

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        if self._shape is None and shape is not None:
            val_updated = False
            self._shape = shape
            if self._shape != ():
                if self._val is not None and np.ndim(self._val) == 0:
                    # if val is a scalar, reshape it to the new shape
                    self._val = np.full(shape, self._val)
                    val_updated = True

            if self.distributed is False:
                self.global_shape = shape

            if self._meta is not None:
                self._meta['shape'] = shape
                size = shape_to_len(shape)
                self._meta['size'] = size
                if self._locmeta is not None:
                    if self._val is None:
                        self._val = np.ones(shape)
                        val_updated = True
                    if val_updated:
                        self._locmeta['val'] = self._val
                    self._locmeta['shape'] = shape
                    self._locmeta['size'] = size

    @property
    def size(self):
        return shape_to_len(self._shape)

    @property
    def global_shape(self):
        return self._global_shape

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
        return shape_to_len(self._global_shape)

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        if units is not None and self._units is None:
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
            return False

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
        if self.defaults.units is not None:
            return self.defaults.units

        if self.ambiguous_units:
            return None

        return self.units

    def shape_from_child(self, node_meta, src_indices):
        if self.defaults.src_shape is not None:
            return self.defaults.src_shape

        if src_indices is not None and not src_indices.is_full_slice():
            # can't determine the shape of the parent if src_indices are present
            return None

        if self.distributed and not node_meta.distributed:
            return None

        return self.shape

    def val_from_child(self, node_meta, src_indices):
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
        if comm.size > 1:
            return comm.allgather(self.shape)
        else:
            return [self.shape]

    def as_dict(self):
        skip = {'_meta', '_locmeta', '_val'}
        ret = {}
        for name in self.__slots__:
            if name not in skip:
                ret[name] = getattr(self, name)
        return ret


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
        A set of nodes that have multiple input connections.
    """
    node_attr_dict_factory = NodeAttrs

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
        self._dist_nodes = set()

    def find_node(self, pathname, varname, io=None):
        """
        Find a node in the graph.

        Parameters
        ----------
        system : System
            The current scoping system.
        varname : str
            The variable name to find.
        io : str
            The io type of the variable

        Returns
        -------
        tuple of the form (io, name), where io is either 'i' or 'o'.
            The node found.
        """
        name = pathname + '.' + varname if pathname else varname

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
        return self.base_error(f"shape {src_shape} of '{src[1]}' is incompatible with shape "
                               f"{tgt_shape} of '{tgt[1]}'.", src, tgt)

    def value_error(self, going_up, src, tgt, src_val, tgt_val):
        if going_up:
            src, tgt = tgt, src
            src_val, tgt_val = tgt_val, src_val

        if not self.nodes[tgt].discrete:
            sshp = np.shape(src_val)
            tshp = np.shape(tgt_val)
            if sshp != tshp:
                return self.shape_error(src, tgt, sshp, tshp)

        return self.base_error(f"value {truncate_str(src_val, max_len=50)} of '{src[1]}' is "
                               f"incompatible with value {truncate_str(tgt_val, max_len=50)} of "
                               f"'{tgt[1]}'.", src, tgt)

    def units_error(self, going_up, src, tgt, src_units, tgt_units):
        if going_up:
            src, tgt = tgt, src
            src_units, tgt_units = tgt_units, src_units

        return self.base_error(f"units '{src_units}' of '{src[1]}' are incompatible with units "
                               f"'{tgt_units}' of '{tgt[1]}'.", src, tgt, src_indices=False)

    def handle_error(self, model, going_up, src, tgt, exc):
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

        model._collect_error(f"{model.msginfo}: {excstr}", tback=exc.__traceback__,
                                ident=ident)

    def input_root(self, node):
        assert node[0] == 'i'
        ionode = None
        dangling = []
        for n in self.bfs_up_iter(node, include_self=True):
            if n[0] == 'i':
                if self.in_degree(n) == 0:  # over-promoted input or dangling input
                    dangling.append(n)
                else:
                    for p in self.predecessors(n):
                        if p[0] == 'o':
                            ionode = n
                            break

        if ionode is not None:
            return ionode

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
        if prefix:
            return node[1].startswith(prefix)

        return True

    def get_val(self, system, name, units=None, indices=None, get_remote=False, rank=None,
                vec_name='nonlinear', kind=None, flat=False, from_src=True):

        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]
        need_gr = system.comm.size > 1 and not get_remote
        # if need_gr and node_meta.distributed:
        #     raise RuntimeError(f"{system.msginfo}: Variable '{name}' is a distributed variable. "
        #                        "You can retrieve values from all processes using "
        #                        "`get_val(<name>, get_remote=True)` or from the local process using "
        #                        "`get_val(<name>, get_remote=False)`.")

        if node[0] == 'o':
            tgt_units, tgt_inds_list = None, ()
        else:
            tgt_inds_list = node_meta.src_inds_list
            tgt_units = node_meta.units
            # ambiguous units aren't fatal during setup, but if we're getting a specific promoted
            # input that has ambiguous units, it becomes fatal, so we need to check that here.
            if node_meta.ambiguous_units:
                raise ValueError(self.ambig_units_msg(node))

        if node[0] == 'o':
            src_node = node
        elif from_src:
            src_node = self.get_root(node)
        else:
            # getting a specific input
            # (must use absolute name or have only a single leaf node)
            leaves = list(self.leaf_input_iter(node))

            if leaves:
                if len(leaves) > 1:
                    raise ValueError(
                        f"{system.msginfo}: Promoted variable '{name}' refers to multiple "
                        "input variables so the choice of input is ambiguous.  Either "
                        "use the absolute name of the input or set 'from_src=True' to "
                        "retrieve the value from the connected output.")
                src_node = leaves[0]
            else:
                src_node = node

        src_meta = self.nodes[src_node]
        src_units = src_meta.units
        if need_gr and src_meta.distributed and not node_meta.distributed:
            raise RuntimeError(f"{self.msginfo}: Non-distributed variable '{node[1]}' has "
                                f"a distributed source, '{src_node[1]}', so you must retrieve its "
                                "value using 'get_remote=True'.")

        if system.has_vectors():
            model = system._problem_meta['model_ref']()
            if model._resolver.is_prom(src_node[1], 'input' if src_node[0] == 'i' else 'output'):
                abs_name = model._resolver.prom2abs(src_node[1])
            else:
                abs_name = src_node[1]
            val = system._abs_get_val(abs_name, get_remote, rank, vec_name, kind, flat,
                                      from_root=True)
        else:
            val = src_meta.val

            if is_undefined(val):
                raise ValueError(f"{system.msginfo}: Variable '{self.msgname(src_node)}' has not "
                                 "been initialized.")

        if indices is not None and not isinstance(indices, Indexer):
            indices = indexer(indices, flat_src=flat)
        try:
            val = self.convert_get(node, val, src_units, tgt_units, tgt_inds_list, units, indices,
                                   get_remote=get_remote)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't get value of '{node[1]}': {str(err)}")

        if flat:
            val = val.ravel()

        return val

    def set_val(self, system, name, val, units=None, indices=None):
        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]

        nodes = self.nodes
        src_node = self.get_root(node)
        src_meta = nodes[src_node]
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
                # model._initial_condition_cache[node] = (val, None, None)
                self.set_tree_val(model, src_node, val)

            return

        # every variable is continuous from here down
        if node[0] == 'o':
            tgt_units, tgt_inds_list = None, ()
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
            val = self.convert_set(val, src_units, tgt_units, (),  units)
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

        if system.has_vectors():
            srcval = model._abs_get_val(src, get_remote=False)
            if np.ndim(srcval) > 0:
                self.set_subarray(srcval, inds, val, node)
            else:
                srcval = val
                system._outputs._abs_set_val(src, srcval)
        else:
            # if src_node in model._initial_condition_cache:
            #     srcval, src_units, _ = model._initial_condition_cache[src_node]
            # else:
            #     srcval = src_meta.val
            srcval = src_meta.val

            if srcval is not None:
                if isinstance(srcval, Number):
                    if inds:
                        raise RuntimeError("Can't set a non-array using indices.")
                    src_meta.val = val
                    srcval = src_meta.val
                else:
                    self.set_subarray(srcval, inds, val, node)
            else:
                if inds:
                    raise RuntimeError(f"Shape of '{name}' isn't known yet so you can't use "
                                       f"indices to set it.")
                srcval = val

            # model._initial_condition_cache[src_node] = (srcval, src_units, None)

        # propagate shape and value down the tree
        self.set_tree_val(model, src_node, srcval)

    def set_tree_discrete_val(self, model, src_node, srcval):
        nodes = self.nodes
        for u, v in dfs_edges(self, src_node):
            if v[0] == 'i':
                vmeta = nodes[v]
                if vmeta.locmeta is not None:
                    vmeta.locmeta['val'] = srcval

    def set_tree_val(self, model, src_node, srcval):
        nodes = self.nodes
        src_meta = nodes[src_node]
        src_meta.val = srcval
        src_dist = src_meta.distributed

        if src_meta.discrete:
            return self.set_tree_discrete_val(model, src_node, srcval)

        for leaf in self.leaf_input_iter(src_node):
            tgt_meta = nodes[leaf]
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

        # for u, v in dfs_edges(self, src_node):
        #     vmeta = nodes[v]
        #     umeta = nodes[u]
        #     shape = umeta.shape
        #     if v[0] == 'i':
        #         if vmeta.locmeta is None or v in self._dist_nodes:
        #             continue

        #         src_indices = self.edges[(u, v)].get('src_indices', None)
        #         if src_indices is not None:
        #             if src_indices._src_shape is None:
        #                 src_indices.set_src_shape(shape)
        #             shape = src_indices.indexed_src_shape

        #         val = apply_idx_list(srcval, vmeta.src_inds_list)
        #         vmeta.val = val

        #     else:  # output node
        #         if vmeta.shape is None:
        #             vmeta.shape = shape

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
        if (src, tgt) in self.edges():
            return

        if src not in self or tgt not in self:
            raise ValueError(f"Node {src} or {tgt} not found in the graph.")

        if self.in_degree(tgt) != 0:
            self._mult_inconn_nodes.add(tgt)
            if src[0] == 'o':
                for p in self.pred[tgt]:
                    if p[0] == 'o':
                        group._collect_error(
                            f"{group.msginfo}: Target '{self.msgname(tgt)}' cannot be "
                            f"connected to '{self.msgname(src)}' because it's already "
                            f"connected to '{self.msgname(p)}'.", ident=(src, tgt))
                        return

        self.add_edge(src, tgt, **kwargs)

    def create_node_meta(self, system, name, io):
        key = (io[0], system.pathname + '.' + name if system.pathname else name)

        meta = {'pathname': system.pathname, 'rel_name': name}

        return key, meta

    def get_path_prom(self, node):
        meta = self.nodes[node]
        return meta.pathname, meta.rel_name

    def set_model_meta(self, model, node, meta, locmeta):
        node_meta = self.nodes[node]

        # this is only called on nodes corresponding to variables in the model, not on
        # nodes internal to the tree.
        if meta is not None:
            node_meta._meta = meta
            node_meta._locmeta = locmeta

            if locmeta is None:
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
                        self._distributed_vars.add(node)
                    if not node_meta.dyn_shape:
                        node_meta._shape = meta['shape']
                        if node_meta.distributed:
                            node_meta.global_shape = self.compute_global_shape(model, node)
                        else:
                            node_meta.global_shape = meta['shape']
                        if locmeta is not None:
                            node_meta._val = locmeta['val']

                if not node_meta.dyn_units:
                    node_meta._units = meta['units']

    def add_continuous_var(self, model, name, meta, locmeta, io):
        node = (io[0], name)

        if node not in self:
            node, node_meta = self.create_node_meta(model, name, io)
            self.add_node(node, **node_meta)

        node_meta = self.nodes[node]
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
        node = (io[0], name)
        if node not in self:
            node, node_meta = self.create_node_meta(model, name, io)
            self.add_node(node, **node_meta)

        node_meta = self.nodes[node]
        node_meta.discrete = True
        self.set_model_meta(model, node, meta, locmeta)

    def add_variable_meta(self, model):
        self._distributed_vars = set()
        for io in ['input', 'output']:
            loc = model._var_abs2meta[io]
            for name, meta in model._var_allprocs_abs2meta[io].items():
                locmeta = loc[name] if name in loc else None
                self.add_continuous_var(model, name, meta, locmeta, io)

            loc = model._var_discrete[io]
            for name, meta in model._var_allprocs_discrete[io].items():
                locmeta = loc[name] if name in loc else None
                self.add_discrete_var(model, name, meta, locmeta, io)

    def get_dist_shapes(self, model, node=None):
        if self._dist_shapes is None:
            if model.comm.size > 1:
                dshapes = {}
                allmeta = model._var_allprocs_abs2meta
                for io, name in self._distributed_vars:
                    if io == 'i':
                        meta = allmeta['input'][name]
                    else:
                        meta = allmeta['output'][name]
                    if meta['shape'] is not None:
                        dshapes[name] = meta['shape']

                all_dshapes = {}

                for rank, dshp in enumerate(model.comm.allgather(dshapes)):
                    for n, shp in dshp.items():
                        if n not in all_dshapes:
                            all_dshapes[n] = [None] * model.comm.size
                        all_dshapes[n][rank] = shp

                self._dist_shapes = all_dshapes
            else:
                self._dist_shapes = {}

        if node is None:
            return self._dist_shapes

        _, name = node
        if name in self._dist_shapes:
            return self._dist_shapes[name]

        if node in self._distributed_vars:
            if name in model._var_allprocs_abs2meta['output']:
                shape = model._var_allprocs_abs2meta['output'][name]['shape']
            else:  # must be in inputs
                shape = model._var_allprocs_abs2meta['input'][name]['shape']
            dist_shapes = model.comm.allgather(shape)
            self._dist_shapes[name] = dist_shapes
            return dist_shapes
        else:
            raise ValueError(f"Can't get distributed shapes for variable '{node[1]}' because it is "
                             "not a distributed variable in the model.")

    def compute_global_shape(self, model, node):
        return model.get_global_dist_shape(node[1], self.get_dist_shapes(model, node))
        # else:
        #     raise ValueError(f"Can't compute global shape for variable '{node[1]}' because it is "
        #                      "not a distributed variable in the model.")

        # meta = self.nodes[node]
        # if meta.distributed:
        #     return model.get_global_dist_shape(node[1], self._dist_shapes[node[1]])
        # else:
        #     shp = list(meta.shape)
        #     if shp:
        #         return (model.comm.size * shp[0],) + shp[1:]
        #     else:
        #         return (model.comm.size,)

    def add_promotion(self, io, group, prom_name, subsys, sub_prom, pinfo=None):
        # we invert the order here for inputs vs. outputs.  For inputs, the promoted name
        # is the source and the subsys name is the target.  For outputs, the promoted name is the
        # target and the subsys name is the source.  This gives us a nice tree that flows from
        # the absolute output to all of the connected absolute inputs which lets us use
        # dfs_postorder_nodes.
        if io == 'input':
            src, src_kwargs = self.create_node_meta(group, prom_name, io)
            tgt, tgt_kwargs = self.create_node_meta(subsys, sub_prom, io)
        else:
            src, src_kwargs = self.create_node_meta(subsys, sub_prom, io)
            tgt, tgt_kwargs = self.create_node_meta(group, prom_name, io)

        self.add_node(src, **src_kwargs)
        self.add_node(tgt, **tgt_kwargs)

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
            self.nodes[tgt].defaults.src_shape = src_shape

    def add_manual_connections(self, group):
        manual_connections = group._manual_connections
        resolver = group._resolver
        allprocs_discrete_in = group._var_allprocs_discrete['input']
        allprocs_discrete_out = group._var_allprocs_discrete['output']

        for prom_tgt, (prom_src, src_indices, flat) in manual_connections.items():
            src_io = resolver.get_iotype(prom_src)
            if src_io is None:
                guesses = get_close_matches(prom_src, list(resolver.prom_iter('output')) +
                                            list(allprocs_discrete_out.keys()))
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
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                    f"'{prom_tgt}', but '{prom_tgt}' is an output. All "
                                    "connections must be to an input.")
                continue

            if tgt_io is None:
                guesses = get_close_matches(prom_tgt, list(resolver.prom_iter('input')) +
                                            list(allprocs_discrete_in.keys()))
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                     f"'{prom_tgt}', but '{prom_tgt}' doesn't exist. Perhaps you "
                                     f"meant to connect to one of the following inputs: {guesses}.")
                continue

            out_comps = {n.rpartition('.')[0] for n in resolver.absnames(prom_src, src_io)}
            in_comps = {n.rpartition('.')[0] for n in resolver.absnames(prom_tgt, tgt_io)}

            if out_comps & in_comps:
                group._collect_error(f"{group.msginfo}: Source and target are in the same System "
                                     f"for connection from '{prom_src}' to '{prom_tgt}'.")
                continue

            src, src_kwargs = self.create_node_meta(group, prom_src, src_io)
            tgt, tgt_kwargs = self.create_node_meta(group, prom_tgt, tgt_io)

            if src not in self:
                self.add_node(src, **src_kwargs)
            if tgt not in self:
                self.add_node(tgt, **tgt_kwargs)

            if src_io == 'input' and tgt_io == 'input':
                self._input_input_conns.add((src, tgt))

            self.check_add_edge(group, src, tgt, type='manual', src_indices=src_indices,
                                flat_src_indices=flat)

    def add_group_input_defaults(self, group):
        notfound = []
        nodes = self.nodes
        for name, gin_meta in group._group_inputs.items():
            path = group.pathname + '.' + name if group.pathname else name
            node = ('i', path)
            if node not in self:
                if not group._resolver.is_prom(name, 'input'):
                    notfound.append(name)
                    continue

                node, kwargs = self.create_node_meta(group, name, 'input')
                self.add_node(node, **kwargs)

            node_meta = nodes[node]
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

        # start with outputs (the root node of each connection tree) and do a postorder traversal
        # that rolls up desired metada from bottom of the tree to either the top promoted input node
        # or, in the case of an auto_ivc source, all the way to the source node.
        in_degree = self.in_degree
        for node in self.nodes():
            if node[0] == 'o' and in_degree(node) == 0:  # absolute output node
                if self.out_degree(node) == 0:
                    continue

                self.resolve_conn_tree(model, node)

        self._first_pass = False

    def gather_data(self, model):
        # def include_edge(edge):
        #     # include any edges that are manual connections or have src_indices
        #     edge_meta = self.edges[edge]
        #     if edge_meta.get('type') == 'manual' or edge_meta.get('src_indices', None) is not None:
        #         return True

        myrank = model.comm.rank
        resolver = model._resolver
        vars_to_gather = model._vars_to_gather
        nodes_to_send = {}
        edges_to_send = {}
        for abs_out in resolver.abs_iter('output'):
            own_out = abs_out in vars_to_gather and vars_to_gather[abs_out] == myrank
            out_node = ('o', abs_out)
            if own_out:
                for u, v in nx.dfs_edges(self, out_node):
                    if v[0] == 'i':
                        break
                    nodes_to_send[u] = self.nodes[u].as_dict()
                    edges_to_send[u, v] = self.edges[u, v]

            for in_node in self.leaf_input_iter(out_node):
                own_in = in_node[1] in vars_to_gather and vars_to_gather[in_node[1]] == myrank
                if own_out or own_in:
                    path = nx.shortest_path(self, out_node, in_node)
                    for i, n in enumerate(path):
                        if n[0] == 'i':
                            opath = path[:i]
                            ipath = path[i:]
                            break

                    if own_in:
                        edges_to_send[opath[-1], ipath[0]] = self.edges[opath[-1], ipath[0]]
                        for i, p in enumerate(ipath):
                            nodes_to_send[p] = self.nodes[p].as_dict()
                            if i > 0:
                                edges_to_send[ipath[i-1], p] = self.edges[ipath[i-1], p]
                    else:  # own_out
                        if ipath:
                            edges_to_send[opath[-1], ipath[0]] = self.edges[opath[-1], ipath[0]]

        # for name, owner in model._vars_to_gather.items():
        #     if myrank == owner:
        #         seen = False
        #         io = resolver.get_iotype(name)
        #         if io == 'input':
        #             node = ('i', name)
        #             for n in self.bfs_up_iter(node):
        #                 if n[0] == 'o':
        #                     if not seen:
        #                         edge = (n, node)
        #                         edges_to_send[edge] = self.edges[edge]
        #                         seen = True
        #                     continue
        #                 nodes_to_send[n] = self.nodes[n].as_dict()
        #         else:
        #             node = ('o', name)
        #             for n in self.bfs_down_iter(node):
        #                 if n[0] == 'i':
        #                     break
        #                 nodes_to_send[n] = self.nodes[n].as_dict()

        # if nodes_to_send:

            # # read-only view of the subgraph
            # subview = self.subgraph(nodes_to_send)

            # includes = ('pathname', 'rel_name', 'shape', 'units', 'defaults', 'flags',
            #             'copy_shape', 'compute_shape', 'copy_units', 'compute_units', 'val',
            #             'compute_units')
            # subgraph = nx.DiGraph()
            # for node, nodeattrs in subview.nodes(data=True):
            #     dct = {key: getattr(nodeattrs, key) for key in includes}
            #     subgraph.add_node(node, **dct)
            # subgraph.add_edges_from(subview.edges(data=True))

            # TODO: for now just allow src_indices to be transferred.  Probably need to handle case
            #       where src_indices are a large array...
            # for _, data in subview.edges(data=True):
            #     src_indices = data.get('src_indices', None)
            #     if src_indices is not None:
            #         pass
            #     data['src_indices'] = None
            #     data['flat_src_indices'] = None
        # else:
        #     subgraph = None

        # subgraphs = [s for s in model.comm.allgather(subgraph) if s is not None]
        graph_info = model.comm.allgather((nodes_to_send, edges_to_send))

        all_abs2meta = model._var_allprocs_abs2meta
        all_discrete = model._var_allprocs_discrete
        for nodes, edges in graph_info:
            for node, data in nodes.items():
                if node not in self:
                    data['remote'] = True
                    self.add_node(node, **data)
                    if node[0] == 'i':
                        na2m = all_abs2meta['input']
                        ndisc = all_discrete['input']
                    else:
                        na2m = all_abs2meta['output']
                        ndisc = all_discrete['output']

                    if node[1] in na2m:
                        nodes[node].meta = na2m[node[1]]
                    elif node[1] in ndisc:
                        nodes[node].meta = ndisc[node[1]]

            for edge, data in edges.items():
                if edge in self.edges:
                    if data.get('src_indices', None) is not None:
                        if self.edges[edge].get('src_indices', None) is None:
                            self.edges[edge]['src_indices'] = data['src_indices']
                            self.edges[edge]['flat_src_indices'] = data['flat_src_indices']
                else:
                    self.add_edge(edge[0], edge[1], **data)

        # for sub in subgraphs:
        #     for node, data in sub.nodes(data=True):
        #         if node not in self:
        #             data['remote'] = True
        #             self.add_node(node, **data)
        #             if node[0] == 'i':
        #                 na2m = all_abs2meta['input']
        #                 ndisc = all_discrete['input']
        #             else:
        #                 na2m = all_abs2meta['output']
        #                 ndisc = all_discrete['output']

        #             if node[1] in na2m:
        #                 nodes[node].meta = na2m[node[1]]
        #             elif node[1] in ndisc:
        #                 nodes[node].meta = ndisc[node[1]]

        #     for u, v, data in sub.edges(data=True):
        #         edge = (u, v)
        #         if edge in edges:
        #             if data.get('src_indices', None) is not None:
        #                 if edges[edge].get('src_indices', None) is None:
        #                     edges[edge]['src_indices'] = data['src_indices']
        #                     edges[edge]['flat_src_indices'] = data['flat_src_indices']
        #         else:
        #             self.add_edge(edge[0], edge[1], **data)

    def resolve_from_children(self, model, src_node, node, auto=False):
        if self.out_degree(node) == 0:  # skip leaf nodes
            return

        try:
            children_meta = \
                [(self.nodes[child], self.edges[(node, child)].get('src_indices', None))
                 for child in self.succ[node]]

            node_meta = self.nodes[node]
            if auto or node[0] == 'i':
                remote = all(m.remote for m, _ in children_meta)
            else:
                remote = False  # dont' transfer 'remote' status to connected outputs

            node_meta.remote = remote
            node_meta.require_connection = any(m.require_connection for m, _ in children_meta)
            node_meta.discrete = discrete = self.get_discrete_from_children(model, node,
                                                                            children_meta)
            node_meta.distributed = \
                self.get_distributed_from_children(model, node, children_meta, auto,
                                                   self.nodes[src_node].distributed)

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
                # if node_meta.distributed and len(children_meta) == 1:
                #     cmeta, src_inds = children_meta[0]
                #     if src_inds is None:
                #         node_meta.global_shape = cmeta.global_shape
                #         chname = list(self.succ[node])[0][1]
                #         self._dist_shapes[node[1]] = self._dist_shapes[chname]

            if not ambig_val:
                val = self.get_val_from_children(model, node, children_meta, node_meta.defaults,
                                                 auto)
                if val is not None:
                    if node[1].startswith('_auto_ivc.'):
                        val = deepcopy(val)
                    node_meta.val = val

            if ambig_units:
                raise ConnError(self.ambig_units_msg(ambig_units))
            if ambig_val:
                raise ConnError(self.ambig_values_msg(ambig_val))

        # finally:
        #     pass
        except Exception as err:
            if isinstance(err, ConnError):
                model._collect_error(f"{model.msginfo}: {err}", tback=err.__traceback__,
                                     ident=node)
            else:
                model._collect_error(f"{model.msginfo}: While resolving children of '{node[1]}': "
                                     f"{err}", tback=err.__traceback__,
                                     ident=node)

    def ambig_units_msg(self, node, incompatible=False):
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
                    f"{self.top_name(node)}', units=?)' to remove the ambiguity.")
        return msg

    def ambig_shapes_msg(self, node, children_meta):
        node_meta = self.nodes[node]
        shapes = [m.shape_from_child(node_meta, src_indices) for m, src_indices in children_meta]
        children = [n for _, n in self.succ[node]]
        rows = sorted((n, s) for n, s in zip(children, shapes))
        table = textwrap.indent(str(generate_table(rows, tablefmt='plain')), '   ')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"incompatible shapes:\n{table}")

    def ambig_values_msg(self, node):
        causes = []
        self.find_ambiguous_causes(node, causes, 'val')
        children = [n for n, _ in causes]
        child_nodes = [('i', n) for n in children]
        causing_meta = [self.nodes[n] for n in child_nodes]
        units_list = [m.units for m in causing_meta]
        vals = [m.val for m in causing_meta]
        ulist = [u if u is not None else '' for u in units_list]
        vlist = [truncate_str(v, max_len=60) for v in vals]
        rows = sorted((n, u, v) for n, u, v in zip(children, ulist, vlist))
        table = textwrap.indent(str(generate_table(rows, tablefmt='plain')), '   ')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"values, so the value of '{node[1]}' is ambiguous:\n{table}\n   Call "
                f"model.set_input_defaults('"
                f"{self.top_name(node)}', val=?)' to remove the ambiguity.")

    def find_ambiguous_causes(self, node, causes, data_name):
        """
        Starting from an ambiguous node, find all of the nodes that are causing the ambiguity.

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
            child_meta = self.nodes[child]
            if getattr(child_meta, attr):
                self.find_ambiguous_causes(child, causes, data_name)
            else:
                causes.append((child[1], getattr(child_meta, data_name)))

    def get_units_from_children(self, model, node, children_meta, defaults):
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
                    if nodes[s].ambiguous_units:
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
            nodes[node].ambiguous_units = False
            return defaults.units

        if child_units_differ:
            nodes[node].ambiguous_units = True
        else:
            # if a child is ambiguous, this node is also ambiguous if the default units are not set
            if node_ambig:
                nodes[node].ambiguous_units = True
            return start

    def get_shape_from_children(self, node, children_meta, defaults):
        node_meta = self.nodes[node]
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
        node_meta = self.nodes[node]

        start = None
        unshaped_scalar = False
        for i, tup in enumerate(children_meta):
            ch_meta, src_indices = tup
            val = ch_meta.val_from_child(node_meta, src_indices)
            # if val is not None and ch_meta.defaults.val is not None:
            #     def_child_vals.append(ch_meta.defaults.val)

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

        # this isn't really correct behavior but putting it here for backwards compatibility so
        # that if any child has a default value, then the parent will use that value and any
        # ambiguity will be removed.
        # if def_child_vals and len(def_child_vals) == 1:
        #     node_meta.ambiguous_val = False
        #     return def_child_vals[0]

        if not node_meta.ambiguous_val:
            return start

    def get_discrete_from_children(self, group, node, children_meta):
        discretes = [m.discrete for m, _ in children_meta]
        dset = set(discretes)
        if len(dset) == 1:
            discrete = dset.pop()
            if discrete:
                node_meta = self.nodes[node]
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
        # A parent is only desginated as distributed if it has only one child and that child is
        # distributed.

        dist = set()
        has_src_indices = False
        for m, src_indices in children_meta:
            dist.add(m.distributed)
            if src_indices is not None:
                has_src_indices = True

        if auto and True in dist:
            # bad = [n[1] for n, (meta, _) in zip(self.succ[node], children_meta)
            #        if meta.distributed]
            # raise ConnError(f"'{self.get_root(node)[1]}' is connected to distributed inputs "
            #                 f"({sorted(bad)}), but distributed variables cannot be connected to "
            #                 "an auto_ivc output.  Declare an IndepVarComp and connect it to "
            #                 "these inputs to get rid of this error.")
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
        return meta.defaults.val, meta.defaults.units, meta.defaults.src_shape

    def get_parent_val_shape_units(self, parent, child):
        #TODO: factor in the value of 'distributed'
        parent_meta = self.nodes[parent]
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
        model : Model
            The model.
        src : tuple
            The source output node.
        tgt : tuple
            The target input node.
        """
        src_meta = self.nodes[src]
        tgt_meta = self.nodes[tgt]
        src_discrete = src_meta.discrete
        tgt_discrete = tgt_meta.discrete

        if src_discrete != tgt_discrete:
            dmap = {True: 'discrete', False: 'continuous'}
            raise TypeError(f"Can't connect {dmap[tgt_discrete]} variable "
                            f"'{tgt[1]}' to {dmap[src_discrete]} variable '{src[1]}'.")

        src_val = src_meta.val

        if src_discrete:
            if not src_meta.remote and not tgt_meta.remote:
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
        # check compatibility between nodes that are not directly connected
        # if src_val is not None:
        #     src_val = src_indices.indexed_val(np.atleast_1d(src_val))
        src_meta = self.nodes[src]
        tgt_meta = self.nodes[tgt]

        if src_shape is not None:
            if tgt_shape is not None:
                if not array_connection_compatible(src_shape, tgt_shape):
                    raise ConnError(self.shape_error(src, tgt, src_shape, tgt_shape))
            elif not tgt_meta.ambiguous_val:
                tgt_meta.shape = src_shape

        # if src_val is not None:
        #     if tgt_meta.val is not None:
        #         if not are_compatible_values(src_val, tgt_meta.val, src_discrete, src_indices):
        #             raise ConnError(self.value_error(False, src, tgt, src_val, tgt_meta.val))
        #     elif not tgt_meta.ambiguous_val:
        #         tgt_meta.val = src_val

        src_units = src_meta.units
        tgt_units = tgt_meta.units
        if src_units is not None:
            if tgt_units is not None:
                if tgt_units != 'ambiguous' and not is_compatible(src_units, tgt_units):
                    raise ConnError(self.units_error(False, src, tgt, src_units, tgt_units))
            elif tgt_meta.units_by_conn:
                tgt_meta.units = src_units

    # def tgt_val_from_src(self, system, src_node, tgt_node, get_remote=False, rank=None, flat=False):
    #     src_meta = self.nodes[src_node]
    #     tgt_meta = self.nodes[tgt_node]
    #     sdist = src_meta.distributed
    #     tdist = tgt_meta.distributed

    #     if system.has_vectors():
    #         pass
    #     else:
    #         pass

    def get_dist_offset(self, node, rank, flat):
        offset = 0
        for i, dshape in enumerate(self._dist_shapes[node[1]]):
            if i == rank:
                break
            if dshape is not None:
                if flat:
                    offset += shape_to_len(dshape)
                else:
                    offset += dshape[0] if len(dshape) > 0 else 1

        if dshape is None:
            return offset, None

        if flat:
            sz = shape_to_len(dshape)
        else:
            sz = dshape[0] if len(dshape) > 0 else 1

        return offset, sz

    def check_dist_connection(self, model, src_node):
        """
        Check a connection starting at src where src and/or a target is distributed.
        """
        nodes = self.nodes
        src_meta = nodes[src_node]
        src_dist = src_meta.distributed

        if src_dist:
            if src_meta.global_shape is None:
                src_meta.global_shape = self.compute_global_shape(model, src_node)
            src_inds_shape = src_meta.global_shape
        else:
            src_inds_shape = src_meta.shape

        leaves = list(self.leaf_input_iter(src_node))
        for tgt in leaves:
            tgt_meta = nodes[tgt]
            tgt_dist = tgt_meta.distributed

            src_inds_list = tgt_meta.src_inds_list

            if tgt_meta.distributed:
                if tgt_meta.global_shape is None:
                    tgt_meta.global_shape = self.compute_global_shape(model, tgt)

            if src_dist:
                if tgt_dist:  # dist --> dist
                    src_shape = src_meta.shape
                    tgt_shape = tgt_meta.shape
                    if not src_inds_list:
                        # no src_indices, so shape of dist src must match shape of dist tgt on
                        # each rank, and we must specify src_indices to match src and tgt on
                        # each rank.
                        if tgt_shape is not None:
                            offset, sz = self.get_dist_offset(tgt, model.comm.rank, False)
                            if sz is not None:
                                src_indices = \
                                    indexer(slice(offset, offset + sz),
                                                                        flat_src=False,
                                                                        src_shape=src_meta.global_shape)

                                path = nx.shortest_path(self, src_node, tgt)
                                self.edges[(path[-2], tgt)]['src_indices'] = src_indices
                                tgt_meta.src_inds_list = src_inds_list = [src_indices]

                else:  # dist --> serial
                    if not src_inds_list:
                        model._collect_error(f"Can't automatically determine src_indices for "
                                             f"connection from distributed variable '{src_node[1]}'"
                                             f" to serial variable '{tgt[1]}'.")
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
                            offset, sz = self.get_dist_offset(tgt, model.comm.rank, False)
                            src_indices = \
                                indexer(slice(offset, offset + sz),
                                                                    flat_src=False,
                                                                    src_shape=src_inds_shape)

                            path = nx.shortest_path(self, src_node, tgt)
                            self.edges[(path[-2], tgt)]['src_indices'] = src_indices
                            tgt_meta.src_inds_list = src_inds_list = [src_indices]
                else:  # serial --> serial
                    pass

            for i, src_inds in enumerate(src_inds_list):
                if i == 0:
                    src_inds.set_src_shape(src_inds_shape)
                else:
                    src_inds.set_src_shape(src_shape)
                src_shape = src_inds.indexed_src_shape

            self.check_src_to_tgt_indirect(model, src_node, tgt, src_shape, tgt_shape)

    def resolve_output_to_output_down(self, parent, child):
        """
        Resolve metadata for a target output node based on the metadata of a parent output node.

        Parameters
        ----------
        model : Model
            The model.
        parent : tuple
            The source output node.
        child : tuple
            The target output node.
        """
        child_meta = self.nodes[child]
        parent_meta = self.nodes[parent]

        if parent_meta['discrete']:
            for key in _discrete_copy_meta:
                setattr(child_meta, key, getattr(parent_meta, key))
        else:
            for key in _continuous_copy_meta:
                setattr(child_meta, key, getattr(parent_meta, key))
            if parent_meta.distributed:
                self._dist_shapes[child[1]] = self._dist_shapes[parent[1]]

    def resolve_input_to_input_down(self, model, parent, child, auto):
        """
        Resolve a connection between two input nodes.

        Parameters
        ----------
        model : Model
            The model.
        parent : tuple
            The parent input node.
        child : tuple
            The child input node.
        auto : bool
            Whether the source node of the connection tree is an auto_ivc node.
        """
        child_meta = self.nodes[child]
        parent_meta = self.nodes[parent]

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

        # finally:
        #     pass
        except Exception as err:
            self.handle_error(model, False, parent, child, exc=err)

        return True

    def resolve_conn_tree(self, model, src_node):
        """
        Resolve the connection tree rooted at src_node.

        Metadata is first propagated up the tree from the absolute input nodes up to the root
        input node.  For auto_ivc rooted trees, the propagation continues to the root auto_ivc node.
        Then, checking of compatability btwtween nodes is performed from parent to child down the
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
        src_meta = nodes[src_node]
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
                node_meta = nodes[node]
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
                    # self.check_dist_connection(model, edge)
                else:
                    self.resolve_down(model, u, v, auto)

            if has_dist:
                self.check_dist_connection(model, src_node)

        if first_pass and not dynamic:
            self._resolved.add(src_node)

    def add_auto_ivc_nodes(self, model):
        assert model.pathname == ''

        # this occurs before the auto_ivc variables actually exist
        dangling_inputs = [n for n in self.nodes() if n[0] == 'i' and self.in_degree(n) == 0]

        # because we can have manual connection to nodes inside of the input tree, we have to
        # traverse them all to make sure the root input node is actually dangling.
        skip = set()
        if self._mult_inconn_nodes:  # we can skip if no input nodes have > 1 predecessor
            for d in dangling_inputs:
                for _, v in dfs_edges(self, d):
                    if v in self._mult_inconn_nodes:
                        skip.add(d)
                        break

        if skip:
            dangling_inputs = [d for d in dangling_inputs if d not in skip]

        auto_nodes = []
        for i, n in enumerate(dangling_inputs):
            auto_node, meta = self.create_node_meta(model, f'_auto_ivc.v{i}', 'output')
            self.add_node(auto_node, **meta)
            self.add_edge(auto_node, n, type='manual')
            auto_nodes.append(auto_node)

        return auto_nodes

    def check(self, group):
        nodes = self.nodes
        in_degree = self.in_degree
        for node in nodes():
            if node[0] == 'o' and in_degree(node) == 0:  # a root output node

                auto = node[1].startswith('_auto_ivc.')

                for u, v in dfs_edges(self, node):
                    if u[0] == 'o' and v[0] == 'i':  # an output to input connection
                        uunits = nodes[u].units
                        vmeta = nodes[v]
                        if auto and vmeta.distributed:
                            raise ConnError(f"Distributed input '{v[1]}', is not connected.  "
                                            "Declare an IndepVarComp and connect it to this "
                                            "input to eliminate this error.")

                        vunits = vmeta.units
                        if uunits is None or vunits is None:
                            uunitless = _is_unitless(uunits)
                            vunitless = _is_unitless(vunits)
                            if uunitless and not vunitless:
                                issue_warning(f"{group.msginfo}: Input '{v[1]}' with units of "
                                            f"'{vunits}' is connected to output '{u[1]}' "
                                            f"which has no units.")
                            elif not uunitless and vunitless:
                                if not nodes[v].ambiguous_units:
                                    issue_warning(f"{group.msginfo}: Output '{u[1]}' with units of "
                                                f"'{uunits}' is connected to input '{v[1]}' "
                                                f"which has no units.")

        desvars = group.get_design_vars()
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
                group._collect_error(f"{group.msginfo}: Input '{req}'{promstr} requires a "
                                     f"connection but is not connected.")

    def add_implicit_connections(self, model, implicit_conn_vars):
        # implicit connections are added after all promotions are added, so any implicitly connected
        # nodes are guaranteed to already exist in the graph.
        for prom_name in implicit_conn_vars:
            self.check_add_edge(model, ('o', prom_name), ('i', prom_name), type='implicit')

    def update_src_inds_lists(self, model):
        # propagate src_indices down the tree, but don't update shapes because we don't
        # know all of the shapes at the root and leaves of the tree yet.
        edges = self.edges
        nodes = self.nodes
        for node in self.nodes():
            if self.in_degree(node) == 0:
                for u, v in dfs_edges(self, node):
                    if v[0] == 'i':
                        edge_meta = edges[u, v]
                        src_inds = edge_meta.get('src_indices', None)
                        src_inds_list = nodes[u].src_inds_list
                        if src_inds is not None:
                            src_inds_list = src_inds_list.copy()
                            src_inds_list.append(src_inds)

                        nodes[v].src_inds_list = src_inds_list

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

            model._collect_error(f"{model.msginfo}: The following inputs have multiple connections:"
                                 f" {', '.join(msg)}.")

        return conns

    def get_root(self, node):
        in_degree = self.in_degree
        for n in self.bfs_up_iter(node):
            if n[0] == 'o' and in_degree(n) == 0:
                return n

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
        if node[0] == 'i' and self.out_degree(node) == 0:
            yield node
        else:
            for _, node in dfs_edges(self, node):
                if node[0] == 'i' and self.out_degree(node) == 0:
                    yield node

    def leaf_units(self, node):
        return [self.nodes[n]['units'] for n in self.leaf_input_iter(node)]

    def absnames(self, node):
        if node[0] == 'i':
            return [n for _, n in self.leaf_input_iter(node)]
        else:
            return [self.get_root(node)[1]]

    def source_iter(self):
        """
        Iterate over all source nodes in the graph.
        """
        for node in self.nodes():
            if node[0] == 'o' and self.in_degree(node) == 0:
                yield node

    def io_conn_iter(self):
        """
        Iterate over all input-output connections in the graph.

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
        Applies a sequence of indexing operations to the input array.

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
        Sets the provided val into the positions of the original array corresponding to the final
        subarray after applying the sequence of indexing operations.

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
        node = ('i', abs_in)
        if node not in self:
            raise ValueError(f"Input '{abs_in}' not found.")
        src_inds_list = self.nodes[node]['src_inds_list']
        if not src_inds_list:
            return None
        elif len(src_inds_list) == 1:
            return src_inds_list[0].shaped_array()
        else:
            root = self.get_root(node)
            root_shape = self.nodes[root].shape
            arr = np.arange(shape_to_len(root_shape)).reshape(root_shape)
            for inds in src_inds_list:
                arr = inds.indexed_val(arr)
            return arr

    def convert_get(self, node, val, src_units, tgt_units, src_inds_list=(), units=None,
                    indices=None, get_remote=False):
        node_meta = self.nodes[node]

        if not node_meta.discrete:
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

        #if model.comm.size > 1:
            ## do a preliminary gathering of variable sizing data for variables that are not
            ## dynamically shaped so we know sizes of distributed variables.  This sizing info is
            ## only computed at the top level of the system tree.
            #model._setup_var_sizes()

        # add nodes for all absolute inputs and connected absolute outputs
        self.add_variable_meta(model)

        self.add_implicit_connections(model, model._get_implicit_connections())

        systems = list(model.system_iter(include_self=True, recurse=True))
        groups = [s for s in systems if isinstance(s, Group)]
        for g in groups:
            self.add_manual_connections(g)

        if model.comm.size > 1:
            self.gather_data(model)

        # check for cycles
        if not nx.is_directed_acyclic_graph(self):
            cycle_edges = nx.find_cycle(self, orientation='original')
            errmsg = '\n'.join([f'     {edge[0]} ---> {edge[1]}'
                                for edge in cycle_edges])
            model._collect_error('Cycle detected in input-to-input connections. '
                                f'This is not allowed.\n{errmsg}')

        self.update_src_inds_lists(model)
        self.add_auto_ivc_nodes(model)

        for g in groups:
            self.add_group_input_defaults(g)

        model._setup_auto_ivcs()

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

        def get_table_row(name, meta, mods=(), align='LEFT', max_width=None, show_always=False):
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
        meta = self.nodes[node]
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
        rows.append(get_table_row('compute_shape', meta,
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
            f' ALIGN=\"LEFT\"><FONT POINT-SIZE=\"12\"><b>{name}</b></FONT></TD></TR>{combined}</TABLE>>'

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
            if pathname and not self.startswith(pathname, node):
                continue
            newdata = {}
            if node[0] == 'i':
                newdata['fillcolor'] = GRAPH_COLORS['input']
            else:
                newdata['fillcolor'] = GRAPH_COLORS['output']

            if data.ambiguous:
                newdata['color'] = GRAPH_COLORS['ambiguous']
                newdata['penwidth'] = '4'  # Thick border

            newdata['label'] = self.create_node_label(node)
            newdata['tooltip'] = (data.pathname, data.rel_name)
            newdata['style'] = 'filled,rounded'
            newdata['shape'] = 'box'  # Use box shape with rounded corners
            newdata['pathname'] = data.pathname
            newdata['rel_name'] = data.rel_name
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
                    node_meta = nodes[node]
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
        return nx.drawing.nx_pydot.to_pydot(self.get_drawable_graph(pathname, varname,
                                                                    show_cross_boundary))

    def get_dot(self, pathname='', varname=None, show_cross_boundary=True):
        return self.get_pydot_graph(pathname, varname, show_cross_boundary).to_string()

    def get_svg(self, pathname='', varname=None, show_cross_boundary=True):
        return self.get_pydot_graph(pathname, varname,
                                    show_cross_boundary).create_svg().decode('utf-8')

    def display(self, pathname='', varname=None, show_cross_boundary=True, outfile=None):
        write_graph(self.get_drawable_graph(pathname, varname, show_cross_boundary),
                    outfile=outfile)

    def print_tree(self, name):
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
            meta = nodes[node]
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

    def dump_nodes(self):
        dump_nodes(self)

    def dump_edges(self):
        dump_edges(self)
