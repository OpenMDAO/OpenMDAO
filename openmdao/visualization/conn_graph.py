from pprint import pformat
from difflib import get_close_matches
import networkx as nx
from networkx import dfs_edges, dfs_postorder_nodes
import numpy as np
from numbers import Number
from collections import deque
from copy import deepcopy

import webbrowser
import threading
import time
from http.server import HTTPServer

from openmdao.visualization.graph_viewer import write_graph
from openmdao.utils.general_utils import common_subpath, is_undefined, truncate_str
from openmdao.utils.array_utils import array_connection_compatible, shape_to_len
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


_continuous_copy_meta = ['val', 'units', 'shape', 'discrete']
_discrete_copy_meta = ['val', 'discrete']


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


class Ambiguous():
    __slots__ = ['units', 'val']
    def __init__(self):
        self.units = False
        self.val = False

    def __bool__(self):
        return self.units or self.val

    def __contains__(self, item):
        return item == 'units' or item == 'val'

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def __iter__(self):
        yield self.units
        yield self.val

    def __repr__(self):
        return f"Ambiguous(units={self.units}, val={self.val})"


# _flagattrs = frozenset({'discrete', 'resolved', 'require_connection', 'shape_by_conn',
#                         'units_by_conn', 'ambig_units', 'ambig_val'})
# _updateattrs = frozenset({'val', 'units', 'shape'})
# _otherattrs = frozenset({'pathname', 'rel_name', 'absnames', 'src_inds_list', 'meta', 'defaults',
#                          'errors'})
# _allattrs = frozenset(_flagattrs | _updateattrs | _otherattrs)


# class NodeAttrDict():
#     def __init__(self, **kwargs):
#         self.update(kwargs)

#     def __getitem__(self, key):
#         return getattr(self, key)

#     def __setitem__(self, key, value):
#         setattr(self, key, value)

#     def __contains__(self, key):
#         return hasattr(self, key)

#     def update(self, kwargs):
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#     def get(self, key, default=None):
#         return getattr(self, key, default)


# class LocalVal():
#     __slots__ = ['val']

#     def __init__(self, val):
#         self.val = val

#     @property
#     def shape(self):
#         return np.shape(self.val)


# class RemoteVal():

#     __slots__ = ['comm', 'owner', 'val']

#     def __init__(self, comm, owner, val=None):
#         self.comm = comm
#         self.owner = owner
#         self.val = val

#     @property
#     def shape(self):
#         if self.owner == self.comm.rank:
#             shp = np.shape(self.val)
#             self.comm.bcast(shp, root=self.owner)
#             return shp
#         else:
#             return self.comm.bcast(None, root=self.owner)


# class DistributedVal():
#     def __init__(self, comm, val=None):
#         self.comm = comm
#         self.val = val


# class RemoteIndexer():
#     __slots__ = ['comm', 'owner', 'idxer']

#     def __init__(self, comm, owner, idxer=None):
#         self.comm = comm
#         self.owner = owner
#         self.idxer = idxer

#     def indexed_val(self, arr):
#         if self.idxer is None:
#             val = self.comm.bcast(None, root=self.owner)
#         elif self.owner == self.comm.rank:  # this proc is owner of the indexer
#             idxval = self.idxer.indexed_val(arr)
#             junk = self.comm.bcast(idxval, root=self.owner)
#             return idxval
#         else:  # indexer is local but this proc is not the owner
#             junk = self.comm.bcast(None, root=self.owner)
#             return self.idxer.indexed_val(arr)

#     @property
#     def indexed_src_shape(self):
#         if self.idxer is None:
#             return self.comm.bcast(None, root=self.owner)
#         elif self.owner == self.comm.rank:  # this proc is owner of the indexer
#             iss = self.idxer.indexed_src_shape
#             self.comm.bcast(iss, root=self.owner)
#             return iss
#         else:  # indexer is local but this proc is not the owner
#             return self.comm.bcast(None, root=self.owner)


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
    # node_attr_dict_factory = NodeAttrDict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mult_inconn_nodes = set()
        self._input_input_conns = set()
        self._first_pass = True
        self._required_conns = set()

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

        if not self.nodes[tgt]['discrete']:
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

        if node[0] == 'i':
            names = [n for _, n in self.leaf_input_iter(node)]
        else:
            names = [self.get_root(node)[1]]

        names = sorted(names)
        if len(names) == 1:
            names = names[0]

        if node[1] == names:
            return node[1]

        return f'{node[1]} ({names})'

    def combined_name(self, node):
        meta = self.nodes[node]
        if meta['pathname']:
            return f'{meta["pathname"]}.{meta["rel_name"]}'
        else:
            return meta["rel_name"]

    def startswith(self, prefix, node):
        if prefix:
            return self.combined_name(node).startswith(prefix)

        return True

    def get_val(self, system, name, units=None, indices=None, get_remote=False, rank=None,
                vec_name='nonlinear', kind=None, flat=False, from_src=True):

        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]
        if node[0] == 'o':
            tgt_units, tgt_inds_list = None, ()
        else:
            tgt_inds_list = node_meta['src_inds_list']
            tgt_units = node_meta['units']
            # ambiguous units aren't fatal during setup, but if we're getting a specific promoted
            # input that has ambiguous units, it becomes fatal, so we need to check that here.
            if node_meta['ambiguous'].units:
                raise ValueError(self.ambig_units_msg(node))

        if from_src:
            src_node = self.get_root(node)
        else:
            # getting a specific input
            # (must use absolute name or have only a single leaf node)
            leaves = list(self.leaf_input_iter(node))
            if len(leaves) > 1:
                raise ValueError(
                    f"{system.msginfo}: Promoted variable '{name}' refers to multiple "
                    "input variables so the choice of input is ambiguous.  Either "
                    "use the absolute name of the input or set 'from_src=True' to "
                    "retrieve the value from the connected output.")

            src_node = leaves[0]

        src_meta = self.nodes[src_node]
        src_units = src_meta['units']

        if system.has_vectors():
            val = system._abs_get_val(src_node[1], get_remote, rank, vec_name, kind, flat,
                                      from_root=True)
        else:
            model = system._problem_meta['model_ref']()
            if src_node in model._initial_condition_cache:
                val, src_units, tgt_inds_list = model._initial_condition_cache[src_node]
            else:
                val = src_meta['val']
                model._initial_condition_cache[src_node] = (val, src_units, None)

            if is_undefined(val):
                raise ValueError(f"{system.msginfo}: Variable '{self.msgname(src_node)}' has not "
                                 "been initialized.")

        if indices is not None and not isinstance(indices, Indexer):
            indices = indexer(indices, flat_src=flat)
        try:
            val = self.convert_get(node, val, src_units, tgt_units, tgt_inds_list, units, indices,
                                   flat=flat)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't get value of '{node[1]}': {str(err)}")

        return val

    def set_val(self, system, name, val, units=None, indices=None):
        # self.dump_vals("before set_val")
        node = self.find_node(system.pathname, name)
        node_meta = self.nodes[node]
        if node[0] == 'o':
            tgt_units, tgt_inds_list = None, ()
        else:
            tgt_units = node_meta['units']
            tgt_inds_list = node_meta['src_inds_list']

        nodes = self.nodes
        src_node = self.get_root(node)
        src_meta = nodes[src_node]
        src = src_node[1]

        model = system._problem_meta['model_ref']()

        if src_meta['discrete']:
            if system.has_vectors():
                if src in model._discrete_outputs:
                    model._discrete_outputs[src] = val
                if node[0] == 'i':
                    for abs_in_node in self.leaf_input_iter(node):
                        if abs_in_node[1] in model._discrete_inputs:
                            model._discrete_inputs[abs_in_node[1]] = val
            else:
                model._initial_condition_cache[node] = (val, None, None)

            return

        src_units = nodes[src_node]['units']

        if indices is None:
            inds = tgt_inds_list
        else:
            if not isinstance(indices, Indexer):
                indices = indexer(indices)
            inds = list(tgt_inds_list) + [(indices, None)]

        # do unit conversion on given val if needed
        try:
            val = self.convert_set(val, src_units, tgt_units, (),  units)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't set value of '{self.msgname(node)}': "
                             f"{str(err)}")

        if system.has_vectors():
            srcval = model._abs_get_val(src, get_remote=False)
            self.set_subarray(srcval, inds, val, node)
        else:
            if src_node in model._initial_condition_cache:
                srcval, src_units, _ = model._initial_condition_cache[src_node]
            else:
                srcval = nodes[src_node]['val']
                if isinstance(srcval, np.ndarray) and node != src_node:
                    # copy vals down the tree to avoid changing them when we update srcval
                    nval = nodes[node]['val']
                    if nval is srcval:
                        print(f"set_val: setting {node[1]} to {nval}")
                        nodes[node]['val'] = nval = nval.copy()
                        for _, v in dfs_edges(self, node):
                            vval = nodes[v]['val']
                            if vval is srcval:
                                nodes[v]['val'] = nval

            if srcval is not None:
                if isinstance(srcval, Number):
                    if inds:
                        raise RuntimeError("Can't set a non-array using indices.")
                    srcval = val
                else:
                    self.set_subarray(srcval, inds, val, node)
            else:
                if inds:
                    raise RuntimeError(f"Shape of '{name}' isn't known yet so you can't use "
                                       f"indices to set it.")
                srcval = val

            model._initial_condition_cache[src_node] = (srcval, src_units, None)

            if indices is None:
                all_meta_in = model._var_allprocs_abs2meta['input']
                loc_meta_in = model._var_abs2meta['input']
                node_meta = nodes[node]
                for abs_in_node in self.leaf_input_iter(node):
                    _, abs_name = abs_in_node
                    if abs_name in all_meta_in:
                        meta = all_meta_in[abs_name]
                        if 'shape_by_conn' in meta and meta['shape_by_conn']:
                            abs_in_meta = nodes[abs_in_node]
                            # get any src_indices applied downstream of the initial target node
                            src_inds_list = \
                                abs_in_meta['src_inds_list'][len(node_meta['src_inds_list']):]
                            if src_inds_list:
                                # compute final val shape
                                inval = self.get_subarray(val, src_inds_list)
                            else:
                                inval = val

                            # val = ic_cache[abs_name][0]
                            shape = () if np.isscalar(inval) else inval.shape
                            all_meta_in[abs_name]['shape'] = shape
                            if abs_name in loc_meta_in:
                                loc_meta_in[abs_name]['shape'] = shape

        # self.dump_vals(f"after set_val")

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
        mypreds = self.pred
        if include_self:
            yield node
        stack = [mypreds[node]]
        while stack:
            preds = stack.pop()
            for node in preds:
                yield node
                stack.append(mypreds[node])

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
        mysuccs = self.succ
        if include_self:
            yield node
        stack = [mysuccs[node]]
        while stack:
            succs = stack.pop()
            for node in succs:
                yield node
                stack.append(mysuccs[node])

    def check_add_edge(self, group, src, tgt, **kwargs):
        if (src, tgt) in self.edges():
            return

        if src not in self or tgt not in self:
            raise ValueError(f"Node {src} or {tgt} not found in the graph.")

        if self.in_degree(tgt) != 0:
            self._mult_inconn_nodes.add(tgt)
            iotypes = [p[0] for p in self.pred[tgt]]
            if 'o' in iotypes and src[0] == 'o':
                for p in self.pred[tgt]:
                    if p[0] == 'o':
                        group._collect_error(
                            f"{group.msginfo}: Target '{self.msgname(tgt)}' cannot be "
                            f"connected to '{self.msgname(src)}' because it's already "
                            f"connected to '{self.msgname(p)}'.", ident=(src, tgt))
                        return
                return

        self.add_edge(src, tgt, **kwargs)

    def create_node_meta(self, group, name, io):
        if not (name.startswith('_auto_ivc.') or group._resolver.is_prom(name, io) or
                group._resolver.is_abs(name, io)):
            raise KeyError(group._resolver._add_guesses(
                           name, f"{group.msginfo}: '{name}' not found."))

        key = (io[0], '.'.join((group.pathname, name)) if group.pathname else name)

        meta = {'pathname': group.pathname, 'rel_name': name, 'discrete': None, 'resolved': None,
                'units': None, 'val': None, 'shape': None, 'meta': None}

        if io == 'input':
            meta['require_connection'] = False
            meta['src_inds_list'] = []
            meta['ambiguous'] = Ambiguous()

        return key, meta

    def get_path_prom(self, node):
        meta = self.nodes[node]
        return meta['pathname'], meta['rel_name']

    def add_continuous_var(self, group, name, meta, locmeta, io):
        node = (io[0], name)

        if node not in self:
            node, node_meta = self.create_node_meta(group, name, io)
            self.add_node(node, **node_meta)

        node_meta = self.nodes[node]

        node_meta['shape_by_conn'] = meta['shape_by_conn']
        node_meta['shape'] = shape = None if meta['shape_by_conn'] else meta['shape']
        node_meta['units_by_conn'] = meta['units_by_conn']
        node_meta['units'] = units = None if meta['units_by_conn'] else meta['units']
        node_meta['discrete'] = False

        node_meta['resolved'] = shape is not None and units is not None

        if io == 'input':
            if meta['require_connection']:
                self._required_conns.add(name)
                node_meta['require_connection'] = True

        if name in locmeta:
            node_meta['val'] = None if meta['shape_by_conn'] else locmeta[name]['val']
            node_meta['meta'] = (meta, locmeta[name])
        else:
            node_meta['meta'] = (meta, None)

    def add_discrete_var(self, group, name, meta, locmeta, io):
        node = (io[0], name)
        if node not in self:
            node, node_meta = self.create_node_meta(group, name, io)
            self.add_node(node, **node_meta)

        node_meta = self.nodes[node]
        node_meta['discrete'] = True
        node_meta['resolved'] = True

        if name in locmeta:
            node_meta['val'] = locmeta[name]['val']
            node_meta['meta'] = (meta, locmeta[name])
        else:
            node_meta['meta'] = (meta, None)

    def add_abs_variable_meta(self, model):
        assert model.pathname == ''
        for io in ['input', 'output']:
            loc = model._var_abs2meta[io]
            for name, meta in model._var_allprocs_abs2meta[io].items():
                self.add_continuous_var(model, name, meta, loc, io)

            loc = model._var_discrete[io]
            for name, meta in model._var_allprocs_discrete[io].items():
                self.add_discrete_var(model, name, meta, loc, io)

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
                            flat_src_indices=flat_src_indices, style='dashed')

        if src_shape is not None:
            # group input defaults haven't been added yet, so just put src_shape there so we
            # can deal with it in the same way as the defaults.
            assert 'defaults' not in self.nodes[tgt], \
                "Group input defaults have already been added"
            self.nodes[tgt]['defaults'] = {'src_shape': src_shape}

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

            input_input = src_io == 'input' and tgt_io == 'input'
            if input_input:
                self._input_input_conns.add((src, tgt))

            self.check_add_edge(group, src, tgt, type='manual', src_indices=src_indices,
                                flat_src_indices=flat)

    def add_group_input_defaults(self, group):
        notfound = []
        for name, gin_meta in group._group_inputs.items():
            path = group.pathname + '.' + name if group.pathname else name
            node = ('i', path)
            if node not in self:
                if not group._resolver.is_prom(name, 'input'):
                    notfound.append(name)
                    continue

                node, kwargs = self.create_node_meta(group, name, 'input')
                self.add_node(node, **kwargs)

            if 'defaults' not in self.nodes[node]:
                self.nodes[node]['defaults'] = gin_meta.copy()
            else:
                defaults = self.nodes[node]['defaults']
                for k, v in gin_meta.items():
                    if k in defaults and defaults[k] is not None:
                        issue_warning(f"{group.msginfo}: skipping default input {k} for '{name}' "
                                      f"because it was already set to {truncate_str(defaults[k])}.")
                        continue
                    defaults[k] = v

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
        for node, node_meta in self.nodes(data=True):
            io, name = node
            if io == 'o' and in_degree(node) == 0:  # absolute output node
                if self.out_degree(node) == 0:
                    continue

                self.resolve_conn_tree(model, node)

        self._first_pass = False

    def get_upward_child_meta(self, model, parent, child):
        """
        Get the metadata for a child node.

        This happens when going up the tree.  If src_indices is present on the edge, the val
        and shape won't be propagated up the tree.  Also, defaults may override certain values.

        Parameters
        ----------
        model : Model
            The model.
        parent : tuple
            The parent node.
        child : tuple
            The child node.

        Returns
        -------
        tuple
            A tuple of the form: (units, shape, val, discrete, required, src_indices,
            shape_by_conn, units_by_conn, ambiguous)
        """
        child_meta = self.nodes[child]

        val = child_meta['val']
        required = child_meta.get('require_connection', False)
        discrete = child_meta['discrete']
        units = child_meta['units']
        shape = child_meta['shape']
        shape_by_conn = child_meta.get('shape_by_conn')
        units_by_conn = child_meta.get('units_by_conn')
        defval, defunits, defshape = self.get_defaults(child_meta)
        ambiguous = child_meta.get('ambiguous')
        errors = []

        if discrete:
            src_indices = None
            if defunits is not None:
                errors.append(f"Cannot set 'units={defunits}' for "
                              f"discrete variable '{child[1]}'.")
            if defshape is not None:
                errors.append(f"Cannot set 'shape={defshape}' for "
                              f"discrete variable '{child[1]}'.")
        else:
            src_indices = self.edges[(parent, child)].get('src_indices', None)
            if src_indices is not None:
                val = None
                shape = None

            if defshape is not None:
                shape = defshape
            if defunits is not None:
                units = defunits

        if defval is not None:
            val = defval
            if defshape is None:
                shape = np.shape(val)

        return units, shape, val, discrete, required, src_indices, \
            shape_by_conn, units_by_conn, ambiguous, errors

    def get_upward_children_meta(self, model, node):
        return [self.get_upward_child_meta(model, node, child) for child in self.succ[node]]

    def resolve_from_children(self, model, node, auto=False):
        outdeg = self.out_degree(node)

        if outdeg == 0:  # skip leaf nodes
            return

        try:
            children_meta = self.get_upward_children_meta(model, node)

            # each child_meta entry is a tuple of the form:
            # units, shape, val, discrete, required, src_indices, shape_by_conn, units_by_conn,
            # ambiguous
            node_meta = self.nodes[node]
            defaults = node_meta.get('defaults', None)
            node_meta['require_connection'] = any(m[4] for m in children_meta)
            node_meta['discrete'] = discrete = self.get_discrete_from_children(node, children_meta)

            for chmeta in children_meta:
                for i, e in enumerate(chmeta[9]):
                    model._collect_error(f"{model.msginfo}: {e}", ident=(i, node))

            if node[0] == 'o':
                for i, cm in enumerate(children_meta):
                    ambig = cm[8]
                    if ambig:
                        nlist = [s for s in self.succ[node]]
                        if ambig.units:
                            raise ConnError(self.ambig_units_msg(nlist[i]))
                        elif ambig.val:
                            raise ConnError(self.ambig_values_msg(nlist[i]))

            if not discrete:
                node_meta['units_by_conn'] = ubyconn =all(m[7] for m in children_meta)
                if not ubyconn or not self._first_pass:
                    node_meta['units'] = self.get_units_from_children(model, node, children_meta,
                                                                      defaults)
                node_meta['shape_by_conn'] = shbyconn =all(m[6] for m in children_meta)
                if not shbyconn or not self._first_pass:
                    node_meta['shape'] = self.get_shape_from_children(node, children_meta, defaults)

            val = self.get_val_from_children(model, node, children_meta, defaults, auto)
            if val is not None and node[1].startswith('_auto_ivc.'):
                val = deepcopy(val)
            node_meta['val'] = val

        except Exception as err:
            if isinstance(err, ConnError):
                model._collect_error(f"{model.msginfo}: {err}", tback=err.__traceback__,
                                     ident=node)
            else:
                model._collect_error(f"{model.msginfo}: While resolving children of '{node[1]}': "
                                     f"{err}", tback=err.__traceback__,
                                     ident=node)

    def ambig_units_msg(self, node):
        rows = []
        self.find_ambiguous_causes(node, rows, 'units')
        rows = sorted(rows, key=lambda x: x[0])
        table = generate_table(rows, tablefmt='plain')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"units:\n{table}\nCall model.set_input_defaults('"
                f"{self.top_name(node)}', units=?)' to remove the ambiguity.")

    def ambig_shapes_msg(self, node, shapes):
        children = [n for _, n in self.succ[node]]
        rows = sorted((n, s) for n, s in zip(children, shapes))
        table = generate_table(rows, tablefmt='plain')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"incompatible shapes:\n{table}")

    def ambig_values_msg(self, node):
        causes = []
        self.find_ambiguous_causes(node, causes, 'val')
        children = [n for n, _ in causes]
        child_nodes = [('i', n) for n in children]
        causing_meta = [self.nodes[n] for n in child_nodes]
        units_list = [m['units'] for m in causing_meta]
        vals = [m['val'] for m in causing_meta]
        ulist = [u if u is not None else '' for u in units_list]
        vlist = [truncate_str(v, max_len=60) for v in vals]
        rows = sorted((n, u, v) for n, u, v in zip(children, ulist, vlist))
        table = generate_table(rows, tablefmt='plain')
        return (f"The following inputs promoted to '{node[1]}' have different "
                f"values, so the value of '{node[1]}' is ambiguous:\n{table}\nCall "
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
        for child in self.succ[node]:
            child_meta = self.nodes[child]
            ambiguous = child_meta['ambiguous']
            if ambiguous[data_name]:
                self.find_ambiguous_causes(child, causes, data_name)
            else:
                causes.append((child[1], child_meta[data_name]))

    def get_units_from_children(self, model, node, children_meta, defaults):
        start = None
        if defaults:
            defunits = defaults.get('units')
        else:
            defunits = None

        nodes = self.nodes

        is_output = node[0] == 'o'
        node_ambig = False
        differ = False
        if children_meta:
            start = children_meta[0][0]
            node_ambig = children_meta[0][8].units

        for i, chmeta in enumerate(children_meta):
            if i == 0:
                continue
            u = chmeta[0]
            ch_ambig = chmeta[8]
            node_ambig |= ch_ambig.units
            if is_output and node_ambig:
                # we want the ambiguous child for error reporting
                for s in self.succ[node]:
                    if nodes[s]['ambiguous'].units:
                        raise ConnError(self.ambig_units_msg(s))

            #if u is not None:
            #if start is None:
                #start = u
            #else:
            mismatch = (start is not None and u is None) or (start is None and u is not None)

            if not mismatch and not is_compatible(start, u):
                slist = list(self.succ[node])
                raise ConnError(self.units_error(True, slist[i], node, start, u))

            differ |= start != u

        if defunits is not None:
            if not is_output:
                nodes[node]['ambiguous'].units = False
            return defunits

        if not differ:
            # if a child is ambiguous, this node is also ambiguous if the default units are not set
            if node_ambig and not is_output:
                nodes[node]['ambiguous'].units = True
            return start

        if not is_output:
            nodes[node]['ambiguous'].units = True

    def get_shape_from_children(self, node, children_meta, defaults):
        defval = defaults.get('val') if defaults else None
        start = None
        for chmeta in children_meta:
            shape = chmeta[1]
            val = chmeta[2]
            if shape is None and val is not None:
                shape = np.shape(val)

            if shape is not None:
                if start is None:
                    start = shape
                else:
                    if not array_connection_compatible(start, shape):
                        shapes = [m[1] for m in children_meta]
                        raise ConnError(self.ambig_shapes_msg(node, shapes))

        if defval is not None:
            return np.shape(defval)
        return start

    # def dump_vals(self, msg):
    #     if msg:
    #         print(msg)
    #     nodes = [('o', '_auto_ivc.v0'), ('i', 'x'), ('i', 'comp1.x'), ('i', 'comp2.x')]
    #     for node in nodes:
    #         node_meta = self.nodes[node]
    #         print(node, node_meta['val'])

    def get_val_from_children(self, model, node, children_meta, defaults, auto):
        if defaults:
            defval = defaults.get('val')
        else:
            defval = None

        ambiguous = self.nodes[node].get('ambiguous', Ambiguous())

        start = None
        for i, ch_meta in enumerate(children_meta):
            val = ch_meta[2]
            ch_ambig = ch_meta[8]
            if ch_ambig.val:
                ambiguous.val = True
                if auto:
                    continue

            if val is not None:
                if start is None:
                    start = val
                    start_type = type(start)
                    start_units = ch_meta[0]
                elif auto:  # values must be the same or value of auto_ivc will be ambiguous

                    if ch_meta[3]:
                        if start_type is not type(val):
                            ambiguous.val = True
                            continue

                        if isinstance(start, np.ndarray):
                            if not np.all(start == val):
                                ambiguous.val = True
                                continue

                        if start != val:
                            ambiguous.val = True

                    else:  # continuous
                        if has_val_mismatch(start_units, start, ch_meta[0], val):
                            ambiguous.val = True
                else:
                    if not are_compatible_values(start, val, ch_meta[3], src_indices=ch_meta[5]):
                        slist = list(self.succ[node])
                        raise ConnError(self.value_error(True, slist[i], node, start, val))


        if defval is not None:
            ambiguous.val = False
            return defval

        if not ambiguous.val:
            return start

    def get_discrete_from_children(self, node, children_meta):
        discretes = [m[3] for m in children_meta]
        dset = set(discretes)
        if len(dset) == 1:
            return dset.pop()
        else:
            slist = list(self.succ[node])
            discs = [s for s, d in zip(slist, discretes) if d]
            non_discs = [s for s, d in zip(slist, discretes) if not d]
            raise ConnError(f"'{node[1]}' has discrete ({sorted(discs)}) and non-discrete "
                            f"({sorted(non_discs)}) children.")

    def get_defaults(self, meta):
        defaults = meta.get('defaults', None)
        if defaults:
            return defaults.get('val'), defaults.get('units'), defaults.get('src_shape', None)
        return None, None, None

    def get_parent_val_shape_units(self, parent, child):
        parent_meta = self.nodes[parent]
        parent_shape = parent_meta['shape']
        parent_val = parent_meta['val']
        parent_units = parent_meta['units']
        src_indices = self.edges[(parent, child)].get('src_indices', None)
        if not (src_indices is None or parent_shape is None):
            src_indices.set_src_shape(parent_shape)
            parent_shape = src_indices.indexed_src_shape
            if parent_val is None:
                parent_val = np.ones(parent_shape)
            else:
                parent_val = src_indices.indexed_val(np.atleast_1d(parent_val))

        parent_ambiguous = parent_meta['ambiguous']
        if parent_ambiguous.val:
            parent_val = None
        if parent_ambiguous.units:
            parent_units = None

        return parent_val, parent_shape, parent_units, src_indices

    def resolve_output_input_connection(self, src, tgt, auto):
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
        src_discrete = src_meta['discrete']
        tgt_discrete = tgt_meta['discrete']

        if src_discrete != tgt_discrete:
            dmap = {True: 'discrete', False: 'continuous'}
            raise TypeError(f"Can't connect {dmap[tgt_discrete]} variable "
                            f"'{tgt[1]}' to {dmap[src_discrete]} variable '{src[1]}'.")

        src_val = src_meta['val']
        tgt_val = tgt_meta['val']

        if src_discrete:
            if not are_compatible_values(src_val, tgt_val, src_discrete):
                raise ConnError(self.value_error(False, src, tgt, src_val, tgt_val))
        else:
            src_units = src_meta['units']
            tgt_units = tgt_meta['units']
            src_shape = src_meta['shape']
            tgt_shape = tgt_meta['shape']

            tgt_ambiguous = tgt_meta['ambiguous']

            edge = (src, tgt)

            src_indices = self.edges[edge].get('src_indices', None)
            if src_indices is not None and src_shape is not None:
                src_indices.set_src_shape(src_shape)
                src_shape = src_indices.indexed_src_shape
                if src_val is not None:
                    src_val = src_indices.indexed_val(np.atleast_1d(src_val))

            if src_shape is not None:
                if tgt_shape is not None:
                    if not array_connection_compatible(src_shape, tgt_shape):
                        raise ConnError(self.shape_error(src, tgt, src_shape, tgt_shape))
                elif not tgt_ambiguous.val:
                    tgt_meta['shape'] = src_shape

            if src_val is not None:
                if tgt_val is not None:
                    if not are_compatible_values(src_val, tgt_val, src_discrete, src_indices):
                        raise ConnError(self.value_error(False, src, tgt, src_val, tgt_val))
                elif not tgt_ambiguous.val:
                    tgt_meta['val'] = src_val

            if src_units is not None:
                if tgt_units is not None:
                    if tgt_units != 'ambiguous' and not is_compatible(src_units, tgt_units):
                        raise ConnError(self.units_error(False, src, tgt, src_units, tgt_units))
                elif tgt_meta['units_by_conn']:
                    tgt_meta['units'] = src_units

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
                child_meta[key] = parent_meta[key]
        else:
            for key in _continuous_copy_meta:
                child_meta[key] = parent_meta[key]

        child_meta['resolved'] = True

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
        parent_discrete = parent_meta['discrete']
        if child_meta['resolved']:
            # force src_indices to have a source shape
            self.get_parent_val_shape_units(parent, child)
            return

        child_val = child_meta['val']

        val = None

        if parent_discrete:
            pass
        else:  # continuous parent
            shape = None
            units = def_units = None

            child_shape = child_meta['shape']
            child_units = child_meta['units']
            child_val = child_meta['val']

            val, shape, units, _ = self.get_parent_val_shape_units(parent, child)

            if val is not None:
                if child_val is not None and  not child_meta.get('shape_by_conn'):
                    if not are_compatible_values(val, child_val, parent_discrete):
                        raise ConnError(self.value_error(False, parent, child, val, child_val))

            if def_units is not None:
                units = def_units

            if units is not None:
                if child_units is not None:
                    if child_units != 'ambiguous' and units != 'ambiguous':
                        if not is_compatible(child_units, units):
                            raise ConnError(self.units_error(False, parent, child, child_units,
                                                            units))
                elif child_units is None and auto:
                    child_meta['units'] = units

            if shape is None:
                if val is not None:
                    shape = np.shape(val)

            if shape is not None:
                if child_shape is not None:
                        if not array_connection_compatible(shape, child_shape):
                            raise ConnError(self.shape_error(parent, child, shape, child_shape))
                else:
                    child_meta['shape'] = shape

        if val is not None and shape is not None:
            val = np.reshape(val, shape)

        # self.dump_vals(f"resolve_input_to_input_down: setting")
        if auto and val is not None:
            child_meta['val'] = val

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
                    self.resolve_output_input_connection(parent, child, auto)
                else:
                    self.resolve_output_to_output_down(parent, child)
            else:
                self.resolve_input_to_input_down(model, parent, child, auto)

        except Exception as err:
            self.handle_error(model, False, parent, child, exc=err)

    def resolve_conn_tree(self, model, src_node):
        """
        Resolve the connection tree rooted at src_node.

        Metadata is first propagated up the tree from the absolute input nodes up to the root
        input node.  For auto_ivc rooted trees, the propagation continues to the root auto_ivc node.
        Then, metadata is propagated down the tree from the root output node.

        Parameters
        ----------
        model : Model
            The model.
        src_node : tuple
            The source node. This is always an absolute output node.
        """
        nodes = self.nodes
        auto = src_node[1].startswith('_auto_ivc.')

        # first, resolve inputs up from the bottom of the tree to the root input node.
        for node in dfs_postorder_nodes(self, src_node):
            if node[0] == 'i':
                node_meta = nodes[node]
                if not node_meta['resolved']:
                    self.resolve_from_children(model, node, auto=auto)

        # resolve auto_ivc node  (these are never promoted so there is always only one output node)
        if auto:
            abs2meta = model._var_allprocs_abs2meta['output']
            discrete2meta = model._var_allprocs_discrete['output']
            loc2meta = model._var_abs2meta['output']
            locdiscrete2meta = model._var_discrete['output']
            src_meta = nodes[src_node]
            name = src_node[1]

            self.resolve_from_children(model, src_node, auto=auto)

            if src_meta['val'] is None:
                shape = src_meta['shape']
                if shape is not None:
                    src_meta['val'] = np.ones(shape)
                else:
                    for s in self.succ[src_node]:
                        if nodes[s]['ambiguous']:
                            break
                    else:
                        model._collect_error(f"Auto_ivc variable '{src_node[1]}' "
                                             "has no shape or value.")

            # for auto_ivcs, transfer graph metadata to the variable metadata
            if src_meta['discrete']:
                meta = discrete2meta[name]
                if src_meta['val'] is not None and name in locdiscrete2meta:
                    locdiscrete2meta[name]['val'] = src_meta['val']
            elif name in abs2meta:  # check name here because may be an uninitialized discrete var
                meta = abs2meta[name]
                if src_meta['shape'] is not None:
                    meta['shape'] = src_meta['shape']
                    meta['size'] = shape_to_len(src_meta['shape'])
                    if name in loc2meta:
                        loc2meta[name]['shape'] = meta['shape']
                        loc2meta[name]['size'] = meta['size']

                if src_meta['units'] is not None:
                    meta['units'] = src_meta['units']
                    if name in loc2meta:
                        loc2meta[name]['units'] = meta['units']

                if src_meta['val'] is not None and name in loc2meta:
                    loc2meta[name]['val'] = src_meta['val']

        # now try filling in any missing metadata going down the tree.  This can happen if
        # for example there are src_indices that block propagation of val and shape below a node
        # where shape or val has been set by set_input_defaults.  This will also fill in
        # missing metadata for promoted output nodes, and it can also set shapes for
        # shape_by_conn inputs.
        if not self._first_pass:
            for u, v in dfs_edges(self, src_node):
                self.resolve_down(model, u, v, auto)

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
        for u, v in self.edges():
            if u[0] == 'o' and v[0] == 'i':  # an output to input connection
                uunits = nodes[u]['units']
                vunits = nodes[v]['units']
                if uunits is None or vunits is None:
                    uunitless = _is_unitless(uunits)
                    vunitless = _is_unitless(vunits)
                    if uunitless and not vunitless:
                        issue_warning(f"{group.msginfo}: Input '{v[1]}' with units of "
                                      f"'{vunits}' is connected to output '{u[1]}' "
                                      f"which has no units.")
                    elif not uunitless and vunitless:
                        vambig = nodes[v]['ambiguous']
                        if not vambig.units:
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
            self.check_add_edge(model, ('o', prom_name), ('i', prom_name), style='dotted',
                                type='implicit')

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
                        src_inds_list = nodes[u]['src_inds_list'] if u[0] == 'i' else []
                        if src_inds is not None:
                            src_inds_list = src_inds_list.copy()
                            flat = edge_meta.get('flat_src_indices', True)
                            src_inds_list.append((src_inds, flat))

                        nodes[v]['src_inds_list'] = src_inds_list

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
            self.add_edge(root, inp_src, **edge_meta)

            tgt_syspath, tgt_prom = self.get_path_prom(tgt)
            if tgt_syspath:
                for abs_in_node in self.leaf_input_iter(tgt):
                    break
                tgt_prom = model._resolver.abs2prom(abs_in_node[1], 'input')
            del model._manual_connections[tgt_prom]

            _, abs_out = root
            if abs_out.startswith('_auto_ivc.'):
                src_prom = abs_out
            else:
                src_prom = model._resolver.abs2prom(abs_out, 'output')

            inp_src_syspath, inp_src_prom = self.get_path_prom(inp_src)
            for abs_in_node in self.leaf_input_iter(inp_src):
                break
            inp_src_prom = model._resolver.abs2prom(abs_in_node[1], 'input')

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
            for idx, flat in indices_list:
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
                for idx, _ in indices_list:
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
            idx, _ = indices_list[i]
            if sub.base is not prev:
                idx.indexed_val_set(prev, sub)

    def get_src_index_array(self, abs_in):
        node = ('i', abs_in)
        if node not in self:
            raise ValueError(f"Input '{abs_in}' not found.")
        src_inds_list = self.nodes[node]['src_inds_list']
        if len(src_inds_list) == 0:
            return None
        elif len(src_inds_list) == 1:
            return src_inds_list[0][0].shaped_array()
        else:
            root = self.get_root(node)
            root_shape = self.nodes[root]['shape']
            arr = np.arange(shape_to_len(root_shape)).reshape(root_shape)
            for inds, _ in src_inds_list:
                arr = inds.indexed_val(arr)
            return arr

    def convert_get(self, node, val, src_units, tgt_units, src_inds_list=(), units=None,
                    indices=None, flat=None):
        if src_inds_list:
            val = self.get_subarray(val, src_inds_list).reshape(self.nodes[node]['shape'])

        if indices:
            val = self.get_subarray(val, [(indices, flat)])

        if units is None:
            units = tgt_units

        if units is not None:
            if src_units is None:
                raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                f"'{units}'.")
            elif src_units != units:
                if units == 'ambiguous':
                    raise TypeError(f"The choice between units of {self.leaf_units(node)} for "
                                     f"input '{node[1]}' is ambiguous. Call "
                                     f"model.set_input_defaults('{self.top_name(node)}', "
                                     f"units=?) to remove the ambiguity.")
                try:
                    scale, offset = unit_conversion(src_units, units)
                except Exception:
                    raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                    f"'{units}'.")

                return (val + offset) * scale

        return val

    def convert_set(self, val, src_units, tgt_units, src_inds_list=(), units=None, indices=None):
        if indices:
            src_inds_list = list(src_inds_list) + [(indices, None)]

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

    def create_node_label(self, node):

        def get_table_row(name, meta, mods=(), align='LEFT', max_width=None):
            if '.' in name:
                parent, _, child = name.rpartition('.')
                meta = meta.get(parent, {})
            else:
                parent = None
                child = name

            content = meta.get(child, None)

            if content is None:
                ambig = meta.get('ambiguous', Ambiguous())
                if name in ambig:
                    ambig = ambig[name]
                    if ambig:
                        content = '?'

            if (content is not None and isinstance(content, np.ndarray)) or content:
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

            return ''

        name = self.combined_name(node)
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
        rows.append(get_table_row('units_by_conn', meta))
        rows.append(get_table_row('discrete', meta))
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

            ambiguous = data.get('ambiguous', Ambiguous())
            if ambiguous:
                newdata['color'] = GRAPH_COLORS['ambiguous']
                newdata['penwidth'] = '4'  # Thick border

            newdata['label'] = self.create_node_label(node)
            newdata['tooltip'] = (data['pathname'], data['rel_name'])
            newdata['style'] = 'filled,rounded'
            newdata['shape'] = 'box'  # Use box shape with rounded corners
            newdata['pathname'] = data['pathname']
            newdata['rel_name'] = data['rel_name']
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

    def display(self, pathname='', varname=None, show_cross_boundary=True):
        write_graph(self.get_drawable_graph(pathname, varname, show_cross_boundary))

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

        print(f"🌐 Starting Simple Connection Graph Web UI on port {port}")
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

    def dump(self):
        skip = {'style', 'tooltip', 'fillcolor', 'label'}
        for node, data in sorted(self.nodes(data=True),
                                 key=lambda x: (x[1]['pathname'].count('.'), x[1]['pathname'])):
            dct = {}
            dct.update({k: v for k, v in data.items() if k not in skip})
            print(node[1], pformat(dct))

        print()

    def dump_edges(self):
        skip = {'style', 'tooltip', 'fillcolor', 'label'}
        for u, v, data in sorted(self.edges(data=True), key=lambda x: (x[0], x[1])):
            dct = {k: v for k, v in data.items() if k not in skip and v is not None}
            print(u, '->', v, pformat(dct))

        print()
