from pprint import pformat
import sys
from difflib import get_close_matches
import networkx as nx
from networkx import dfs_edges, dfs_postorder_nodes
import numpy as np
from numbers import Number
from collections import deque

import webbrowser
import threading
import time
from http.server import HTTPServer

from openmdao.visualization.graph_viewer import write_graph
from openmdao.utils.general_utils import common_subpath, is_undefined, shape2tuple, \
    ensure_compatible, truncate_str
from openmdao.utils.array_utils import array_connection_compatible
from openmdao.utils.units import simplify_unit, is_compatible
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.units import unit_conversion
# from openmdao.utils.indexer import indexer, Indexer


# use hex colors here because the using english names was sometimes causing failure to show
# proper colors in the help dialog.
GRAPH_COLORS = {
    'input': 'peachpuff3',
    'output': 'skyblue3',
    'highlight': '#66ff00',
    'boundary': '#D3D3D3',
}


def is_equal(a, b):
    if not (isinstance(b, type(a)) or isinstance(a, type(b))):
        return False

    if isinstance(a, np.ndarray):
        return a.size == b.size and np.all(np.squeeze(a) == np.squeeze(b))

    return a == b


def are_compatible_values(a, b):
    if not (isinstance(b, type(a)) or isinstance(a, type(b))):
        return False

    if isinstance(a, np.ndarray) and not array_connection_compatible(a.shape, b.shape):
        return False

    return True


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mult_inconn_nodes = set()
        self._input_input_conns = set()

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

        raise KeyError(f"{pathname}: Variable '{varname}' not found in connection graph.")

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

        meta = self.nodes[node]
        absnames = sorted(meta['absnames'])
        if len(absnames) == 1:
            absnames = absnames[0]

        if node[1] == absnames:
            return node[1]

        return f'{node[1]} ({absnames})'

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
        if node[0] == 'o':
            tgt_units = None
            tgt_inds_list = ()
        else:
            tgt_units, tgt_inds_list = self.get_conversion_info(node)

        if from_src:
            src_node = self.get_root(node)
            src_units = self.nodes[src_node]['units']
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
            src_units = self.nodes[src_node]['units']

        if system.has_vectors():
            val = system._abs_get_val(src_node[1], get_remote, rank, vec_name, kind, flat,
                                      from_root=True)
        else:
            model = system._problem_meta['model_ref']()
            if src_node in model._initial_condition_cache:
                val, src_units, tgt_inds_list = model._initial_condition_cache[src_node]
            else:
                val = self.nodes[src_node]['val']
                model._initial_condition_cache[src_node] = \
                    (val, src_units, ())

            if is_undefined(val):
                raise ValueError(f"{system.msginfo}: Variable '{self.msgname(src_node)}' has not "
                                 "been initialized.")

        try:
            val = self.convert_get(val, src_units, tgt_units, tgt_inds_list, units, indices)
        except Exception as err:
            raise ValueError(f"{system.msginfo}: Can't get value of '{self.msgname(node)}': "
                             f"{str(err)}.")

        return val

    def set_val(self, system, name, val, units=None, indices=None):
        node = self.find_node(system.pathname, name)
        if node[0] == 'o':
            tgt_units = None
            tgt_inds_list = ()
        else:
            tgt_units, tgt_inds_list = self.get_conversion_info(node)

        src_node = self.get_root(node)
        src_meta = self.nodes[src_node]
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

        src_units = self.nodes[src_node]['units']

        if indices is None:
            inds = tgt_inds_list
        else:
            inds = list(tgt_inds_list) + [indices]

        # do unit conversion on given val if needed
        val = self.convert_set(val, src_units, tgt_units, (),  units)

        if system.has_vectors():
            srcval = model._abs_get_val(src, get_remote=False)
            self.set_subarray(srcval, inds, val, node)
        else:
            if src_node in model._initial_condition_cache:
                srcval, src_units, _ = model._initial_condition_cache[src_node]
            else:
                srcval = self.nodes[src_node]['val']

            self.set_subarray(srcval, inds, val, node)
            model._initial_condition_cache[src_node] = (srcval, src_units, ())

            if indices is None:
                all_meta_in = model._var_allprocs_abs2meta['input']
                loc_meta_in = model._var_abs2meta['input']
                node_meta = self.nodes[node]
                for abs_in_node in self.leaf_input_iter(node):
                    _, abs_name = abs_in_node
                    if abs_name in all_meta_in:
                        meta = all_meta_in[abs_name]
                        if 'shape_by_conn' in meta and meta['shape_by_conn']:
                            abs_in_meta = self.nodes[abs_in_node]
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

    def up_tree_iter(self, node):
        """
        Iterate up the tree from the given node.

        The given node is included in the iteration.

        Parameters
        ----------
        node : tuple of the form ('i' or 'o', name)
            The node to start from.
        """
        yield node
        stack = [self.predecessors(node)]
        while stack:
            preds = stack.pop()
            for node in preds:
                yield node
                stack.append(self.predecessors(node))

    def get_conversion_info(self, node):
        meta = self.nodes[node]
        return meta['units'], meta['src_inds_list']

    def check_add_edge(self, group, src, tgt, **kwargs):
        if (src, tgt) in self.edges():
            return True

        if src not in self or tgt not in self:
            raise ValueError(f"Node {src} or {tgt} not found in the graph.")

        if self.in_degree(tgt) != 0:
            self._mult_inconn_nodes.add(tgt)
            iotypes = [p[0] for p in self.predecessors(tgt)]
            if 'o' in iotypes and src[0] == 'o':
                for p in self.predecessors(tgt):
                    if p[0] == 'o':
                        group._collect_error(
                            f"{group.msginfo}: Target '{self.msgname(tgt)}' cannot be "
                            f"connected to '{self.msgname(src)}' because it's already "
                            f"connected to '{self.msgname(p)}'.", ident=(src, tgt))
                        return False
                return False

        self.add_edge(src, tgt, **kwargs)
        return True

    def create_node_meta(self, group, name, io):
        # abs2meta = group._var_abs2meta[io]
        # all_abs2meta = group._var_allprocs_abs2meta[io]
        shape = val = units = discrete = varmeta =None

        absnames = group._resolver.absnames(name, io, report_error=False)
        if absnames is None:
            # auto_ivcs may not have been added to the graph yet
            if name.startswith('_auto_ivc.'):
                absnames = [name]
            else:
                raise KeyError(f"{group.msginfo}: '{name} not found.")

        key = (io[0], '.'.join((group.pathname, name)) if group.pathname else name)

        return key, {'io': io[0], 'pathname': group.pathname,
                     'rel_name': name, 'absnames': absnames, 'src_inds_list': [],
                     'units': units, 'val': val, '_shape': shape, 'meta': varmeta,
                     'discrete': discrete}

    def get_path_prom(self, node):
        meta = self.nodes[node]
        return meta['pathname'], meta['rel_name']

    def add_continuous_var(self, group, name, meta, locmeta, io):
        node = (io[0], name)
        if node not in self:
            node, node_meta = self.create_node_meta(group, name, io)
            self.add_node(node, **node_meta)

        node_meta = self.nodes[node]

        node_meta['_shape'] = meta['shape']
        node_meta['units'] = meta['units']
        node_meta['discrete'] = None
        if name in locmeta:
            node_meta['val'] = locmeta[name]['val']
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
        if name in locmeta:
            node_meta['val'] = locmeta[name]['val']
            node_meta['meta'] = (meta, locmeta[name])
        else:
            node_meta['meta'] = (meta, None)

    def add_abs_variable_meta(self, group):
        assert group.pathname == ''
        for io in ['input', 'output']:
            loc = group._var_abs2meta[io]
            for name, meta in group._var_allprocs_abs2meta[io].items():
                self.add_continuous_var(group, name, meta, loc, io)

            loc = group._var_discrete[io]
            for name, meta in group._var_allprocs_discrete[io].items():
                self.add_discrete_var(group, name, meta, loc, io)

    def add_promotion(self, io, group, prom_name, subsys, sub_prom, pinfo=None):
        if io == 'input':
            src, src_kwargs = self.create_node_meta(group, prom_name, io)
            tgt, tgt_kwargs = self.create_node_meta(subsys, sub_prom, io)
        else:
            src, src_kwargs = self.create_node_meta(subsys, sub_prom, io)
            tgt, tgt_kwargs = self.create_node_meta(group, prom_name, io)

        self.add_node(src, **src_kwargs)
        self.add_node(tgt, **tgt_kwargs)

        if pinfo is None:
            src_indices = flat_src_indices = None
        else:
            src_indices = pinfo.src_indices
            flat_src_indices = pinfo.flat

        self.check_add_edge(group, src, tgt, src_indices=src_indices,
                            flat_src_indices=flat_src_indices, prom=True, style='dashed')

    def add_manual_connections(self, group):
        manual_connections = group._manual_connections
        resolver = group._resolver
        allprocs_discrete_in = group._var_allprocs_discrete['input']
        allprocs_discrete_out = group._var_allprocs_discrete['output']

        for prom_tgt, (prom_src, src_indices, flat) in manual_connections.items():
            src_io = resolver.get_iotype(prom_src)
            if src_io is None:
                group._bad_conn_vars.update((prom_tgt, prom_src))
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
                group._bad_conn_vars.update((prom_tgt, prom_src))
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                    f"'{prom_tgt}', but '{prom_tgt}' is an output. All "
                                    "connections must be to an input.")
                continue

            if tgt_io is None:
                group._bad_conn_vars.update((prom_tgt, prom_src))
                guesses = get_close_matches(prom_tgt, list(resolver.prom_iter('input')) +
                                            list(allprocs_discrete_in.keys()))
                group._collect_error(f"{group.msginfo}: Attempted to connect from '{prom_src}' to "
                                     f"'{prom_tgt}', but '{prom_tgt}' doesn't exist. Perhaps you "
                                     f"meant to connect to one of the following inputs: {guesses}.")
                continue

            out_comps = {n.rpartition('.')[0] for n in resolver.absnames(prom_src, src_io)}
            in_comps = {n.rpartition('.')[0] for n in resolver.absnames(prom_tgt, tgt_io)}

            if out_comps & in_comps:
                group._bad_conn_vars.update((prom_tgt, prom_src))
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

            # conn_dict = group._manual_connections if input_input else None
            self.check_add_edge(group, src, tgt, src_indices=src_indices,
                                flat_src_indices=flat)#, input_input=input_input,
                                # conn_dict=conn_dict)

    def add_group_input_defaults(self, group):
        for name, gin_meta in group._group_inputs.items():
            node = ('i', name)
            if node not in self:
                node, kwargs = self.create_node_meta(group, name, 'input')
                self.add_node(node, **kwargs)

            meta = self.nodes[node]
            defaults = {}
            meta['defaults'] = defaults
            src_shape = gin_meta['src_shape']
            val = gin_meta['val']
            units = gin_meta['units']

            if is_undefined(val):
                src_shape = shape2tuple(src_shape)
            else:
                if src_shape is not None:
                    # make sure value and src_shape are compatible
                    val, src_shape = ensure_compatible(name, val, src_shape)
                elif isinstance(val, np.ndarray):
                    src_shape = val.shape
                elif isinstance(val, Number):
                    src_shape = (1,)

                if not meta['discrete']:
                    if src_shape is None:
                        val, _ = ensure_compatible(name, val, None)
                    defaults['shape'] = val.shape

                defaults['val'] = val

            if units is not None:
                if not isinstance(units, str):
                    raise TypeError('%s: The units argument should be a str or None' % self.msginfo)
                defaults['units'] = simplify_unit(units, msginfo=group.msginfo)

            if src_shape is not None:
                defaults['src_shape'] = src_shape

    def rollup_discrete(self, group, node, node_meta, succs):
        if len(succs) == 1:
            node_meta['discrete'] = self.nodes[succs[0]]['discrete']
            return

        nodes = self.nodes
        base = self.nodes[succs[0]]['discrete']
        for succ in succs:
            discrete = nodes[succ]['discrete']
            if discrete != base:
                discvars = [n for n in succs if nodes[n]['discrete']]
                nondiscvars = [n for n in succs if not nodes[n]['discrete']]
                group._collect_error(f"{group.msginfo}: Variable '{node[1]}' "
                                     f"connects to discrete variables {sorted(discvars)} and "
                                     f"continuous variables {sorted(nondiscvars)}. Discrete and "
                                     "continuous variables cannot be connected.")
                return

        node_meta['discrete'] = base

    def rollup_units(self, group, node, node_meta, succs, default, auto_ivc):
        base_units = None
        same = True
        nodes = self.nodes
        for succ in succs:
            units = nodes[succ].get('units', None)
            if units is not None:
                if base_units is None:
                    base_units = units
                    basenode = succ
                else:
                    if not is_compatible(base_units, units):
                        group._collect_error(f"'{succ[1]}' units of '{units}' are "
                                             f"incompatible with '{basenode[1]}' units of "
                                             f"'{base_units}'.")
                    same &= base_units == units

        if default is None:
            if not same:
                if auto_ivc:
                    absname = self.nodes[node]['absnames'][0]
                    prom = group._resolver.abs2prom(absname, 'input')
                    units_list = [nodes[succ].get('units', None) for succ in succs]
                    units_list = [u for u in units_list if u is not None]
                    group._collect_error(f"{group.msginfo}: No default units have been set for "
                                         f"input '{self.msgname(node)}' so the choice of units "
                                         f"between {sorted(units_list)} is ambiguous. Call "
                                         f"model.set_input_defaults('{prom}', units=?) to remove "
                                         "the ambiguity.")
                    base_units = None  # don't propagate ambiguous units
                else:
                    base_units = 'ambiguous'
            node_meta['units'] = base_units
            return

        if base_units is None:
            node_meta['units'] = default
            return

        # base and default are not None
        if is_compatible(base_units, default):
            # default overrides any node value as long as it's compatible
            node_meta['units'] = default
            return

        group._collect_error(f"{group.msginfo}: '{self.msgname(node)}' default units "
                             f"'{default.name()}' and '{self.msgname(basenode)}' units of "
                             f"'{base_units.name()}' are incompatible.")

    def rollup_valshape(self, group, node, node_meta, succs, defaults, auto_ivc):
        shape_base = val_base = None
        same_val = True
        default_val = defaults.get('val', None)
        default_shape = None
        if default_val is not None:
            if isinstance(default_val, np.ndarray):
                default_shape = default_val.shape
            elif isinstance(default_val, Number):
                default_shape = (1,)
        discrete = node_meta['discrete']

        shape_good = False
        nodes = self.nodes
        for succ in succs:
            succmeta = nodes[succ]
            sdefaults = succmeta.get('defaults', {})
            src_inds = None if discrete else self.edges[node, succ].get('src_indices', None)
            if 'meta' in succmeta:
                metameta = succmeta['meta']
                if metameta is not None:
                    shape_by_conn = metameta[0].get('shape_by_conn', False)
                    if shape_by_conn:
                        continue

            if src_inds is None:
                val = succmeta.get('val', None)
                if discrete:
                    _shape = None
                else:
                    _shape = succmeta.get('_shape', None)
                    if _shape is None and val is not None:
                        if isinstance(val, np.ndarray):
                            _shape = val.shape
                        elif isinstance(val, Number):
                            _shape = (1,)
            else:
                # if succ connects to the parent with src_indices, we can't propagate the shape
                # up, but if src_shape is in the defaults, we can use that and compare
                # to '_shape' or defaults['src_shape'] in the other succs.
                _shape = sdefaults.get('src_shape', None)
                val = None  # can't propagate value up because of src_indices

            if _shape is not None:
                if shape_base is None:
                    shape_base = _shape
                    sbase_shape = succ
                else:
                    if not array_connection_compatible(shape_base, _shape):
                        group._collect_error(f"{group.msginfo}: '{self.msgname(succ)}' shape of "
                                             f"'{_shape}' is incompatible with "
                                             f"'{self.msgname(sbase_shape)}' "
                                             f"shape of '{shape_base}'.")

            if val is not None:
                if val_base is None:
                    val_base = val
                    sbase_val = succ
                elif auto_ivc:
                    same_val &= is_equal(val_base, val)

        if default_shape is None:
            node_meta['_shape'] = shape_base
            shape_good = True
        elif shape_base is None:
            node_meta['_shape'] = default_shape
            shape_good = True
        elif array_connection_compatible(shape_base, default_shape):
            node_meta['_shape'] = default_shape
            shape_good = True

        if default_val is None:
            if not same_val:
                absname = self.nodes[node]['absnames'][0]
                prom = group._resolver.abs2prom(absname, 'input')
                group._collect_error(f"{group.msginfo}: No default val has been set for input "
                                     f"'{self.msgname(node)}' but different values feed into it. "
                                     f"Call model.set_input_defaults('{prom}', val=?) to remove "
                                     "the ambiguity.")
            node_meta['val'] = val_base

        elif val_base is None:
            node_meta['val'] = default_val

        # val_base and default_val are not None
        else:
            if are_compatible_values(val_base, default_val):
                # default overrides any node value as long as it's compatible
                node_meta['val'] = default_val
            else:
                group._collect_error(f"{group.msginfo}: '{self.msgname(node)}' default val "
                                     f"'{default_val}' and '{self.msgname(sbase_val)}' val of "
                                     f"'{val_base}' are incompatible.")

        if not shape_good:
            group._collect_error(f"{group.msginfo}: '{self.msgname(node)}' default shape "
                                 f"'{default_shape}' and '{self.msgname(sbase_shape)}' shape of "
                                 f"'{shape_base}' are incompatible.")

    def rollup_to_node(self, group, start_node, auto_ivc=False):
            # do a postorder traversal from the source node to all connected inputs to rollup
            # various properties from the bottom of the tree, and flag any incompatibilities.
            for node in dfs_postorder_nodes(self, start_node):
                if not auto_ivc and node[0] == 'o':
                    continue
                branch_meta = self.nodes[node]
                defaults = branch_meta.get('defaults', {})
                if self.out_degree(node) > 0:
                    succs = list(self.successors(node))
                    self.rollup_discrete(group, node, branch_meta, succs)
                    if not branch_meta['discrete']:
                        self.rollup_units(group, node, branch_meta, succs,
                                          defaults.get('units', None), auto_ivc)
                    self.rollup_valshape(group, node, branch_meta, succs, defaults, auto_ivc)
                else:  # leaf node (absolute input)
                    units = defaults.get('units', None)
                    val = defaults.get('val', None)
                    if units is not None:
                        if branch_meta['units'] is None or is_compatible(units,
                                                                         branch_meta['units']):
                            branch_meta['units'] = units
                        else:
                            group._collect_error(
                                f"{group.msginfo}: '{self.msgname(node)}' "
                                f"default units '{units}' and "
                                f"units of '{branch_meta['units']}' are incompatible.")

                    if val is not None:
                        if branch_meta['val'] is None or are_compatible_values(val,
                                                                               branch_meta['val']):
                            branch_meta['val'] = val
                        else:
                            group._collect_error(f"{group.msginfo}: '{self.msgname(node)}' "
                                                 f"default val '{val}' and "
                                                 f"val of '{branch_meta['val']}' are incompatible.")

    def rollup_input_meta(self, group):
        # start with outputs (the root node of each connection tree) and do a postorder traversal
        # that rolls up desired input metada up to the top promoted input node.
        # Note that auto_ivcs are not in the graph yet.
        for name in group._var_allprocs_abs2meta['output']:
            node = ('o', name)
            if node in self:
                self.rollup_to_node(group, node, auto_ivc=False)

    def add_auto_ivcs(self, group):
        assert group.pathname == ''

        # this occurs before the auto_ivc variables actually exist
        dangling_inputs = [n for n in self.nodes() if n[0] == 'i' and self.in_degree(n) == 0]

        # because we can have manual connection to nodes inside of the input tree, we have to
        # travers them all to make sure the root input node is actually dangling.
        skip = set()
        if self._mult_inconn_nodes:  # we can skip if no input nodes have > 1 predecessor
            for d in dangling_inputs:
                for _, v in dfs_edges(self, d):
                    if v[1] in self._mult_inconn_nodes:
                        skip.add(d)
                        break

        if skip:
            dangling_inputs = [d for d in dangling_inputs if d not in skip]

        auto_nodes = []
        nodes = self.nodes
        for i, n in enumerate(dangling_inputs):
            for in_node in self.leaf_input_iter(n):  # get absolute inputs
                in_meta = nodes[in_node]
                if in_meta['meta'][0]['require_connection']:
                    promoted_as = f', promoted as "{n[1]}",' if in_node[1] != n[1] else ''
                    group._collect_error(f'{self.msginfo}: Input "{in_node[1]}"{promoted_as} '
                                         'requires a connection but is not connected.',
                                         ident=(in_node[1], n[1]))

            auto_node, meta = self.create_node_meta(group, f'_auto_ivc.v{i}', 'output')
            self.add_node(auto_node, **meta)
            self.add_edge(auto_node, n)
            auto_nodes.append(auto_node)

            self.rollup_to_node(group, auto_node, auto_ivc=True)

        return auto_nodes

    def add_implicit_connections(self, group):
        assert group.pathname == ''

        # implicit connections are added after all promotions are added, so any implicitly connected
        # nodes are guaranteed to already exist in the graph.
        for prom_name in group._resolver.get_implicit_conns():
            self.check_add_edge(group, ('o', prom_name), ('i', prom_name), style='dotted')

    def update_src_indices(self, src, group):
        """
        Update the src_indices list for all nodes in the tree rooted at src.

        Also updates shape metadata.

        Parameters
        ----------
        src : tuple
            The source node.
        group : Group
            The group to update the src_indices for.
        """
        abs2meta_in = group._var_allprocs_abs2meta['input']
        nodes = self.nodes
        edges = self.edges
        abs_ins = []

        for u, v in dfs_edges(self, src):
            umeta = nodes[u]
            vmeta = nodes[v]
            shape = umeta['_shape']
            src_inds_list = umeta['src_inds_list']

            if v[0] == 'i':  # target is an input
                edge_meta = edges[u, v]
                src_inds = edge_meta.get('src_indices', None)
                if src_inds is not None:
                    # if not isinstance(src_inds, Indexer):
                    #     src_inds = indexer(src_inds, edge_meta.get('flat_src_indices', None))
                    try:
                        src_inds.set_src_shape(shape)
                    except Exception:
                        type_exc, exc, tb = sys.exc_info()
                        group._collect_error(f"When connecting '{self.msgname(u)}' to "
                                             f"'{self.msgname(v)}': {exc}",
                                             exc_type=type_exc, tback=tb, ident=(src[1], v[1]))
                    shape = src_inds.indexed_src_shape

                    if not array_connection_compatible(shape, vmeta['_shape']):
                        group._collect_error(f"After applying index {truncate_str(str(src_inds))} "
                                             f"to '{self.msgname(u)}', shape {shape} != "
                                             f"{vmeta['_shape']} of '{self.msgname(v)}'.",
                                             ident=(u[1], v[1]))

                    # only make a copy if we are modifying the list
                    src_inds_list = src_inds_list.copy()
                    src_inds_list.append(src_inds)

                if v[1] in abs2meta_in:
                    abs_ins.append(v[1])

            vmeta['src_inds_list'] = src_inds_list

        if abs_ins:
            for abs_in in abs_ins:
                # check shape of absolute input
                abs_in_shape = abs2meta_in[abs_in]['shape']
                expected_shape = nodes[('i', abs_in)]['_shape']
                if not array_connection_compatible(abs_in_shape, expected_shape):
                    src_inds_list = nodes[('i', abs_in)]['src_inds_list']
                    if src_inds_list:
                        if len(src_inds_list) == 1:
                            s = str(src_inds_list[0])
                        else:
                            s = f"{[str(s) for s in src_inds_list]}"
                        sistr = f" after applying src_indices {s} "
                    else:
                        sistr = ""
                    group._collect_error(f"When connecting '{self.msgname(src)}' to "
                                        f"'{self.msgname(('i', abs_in))}'{sistr}: shape "
                                        f"{expected_shape} != {abs_in_shape}.",
                                        ident=(src[1], abs_in))

    def update_shapes(self, model):
        """
        Update the shape, src_shape and src_indices for all nodes in the graph.

        Parameters
        ----------
        model : Group
            The top level group.
        """
        abs2meta = model._var_allprocs_abs2meta['output']
        discrete_outs = model._var_allprocs_discrete['output']

        # loop over all absolute output names and traverse until we find an input node
        for node in self.nodes():
            if node[0] == 'o' and self.in_degree(node) == 0:  # absolute src node
                # only do the following for continuous variables
                if node[1] in abs2meta or node[1].startswith('_auto_ivc.'):
                    self.update_src_indices(node, model)
                elif node[1] in discrete_outs:
                    continue
                else:
                    raise RuntimeError(f"Promoted output node '{node[1]}' has no source node.")

    def transform_input_input_connections(self, model):
        """
        Transform input-to-input connections into input-to-output connections.

        Parameters
        ----------
        model : Group
            The top level group.
        """
        to_move = []
        for u, v in self._input_input_conns:
            src = self.get_root(u)
            to_move.append((u, v, src))

        for u, v, src in to_move:
            self.move_to_src(u, v, src, model)

    def move_to_src(self, inp_src, tgt, new_src, group):
        """
        Move the tgt node to the src node.

        This is called from the top level group.
        """
        tgt_syspath, tgt_prom = self.get_path_prom(tgt)
        edge_meta = {k: v for k, v in self.edges[inp_src, tgt].items() if k != 'input_input'}
        self.remove_edge(inp_src, tgt)
        self.add_edge(new_src, tgt, **edge_meta)
        if tgt_syspath:
            tgt_meta = self.nodes[tgt]
            tgt_prom = group._resolver.abs2prom(tgt_meta['absnames'][0], 'input')

        src_meta = self.nodes[new_src]
        absname0 = src_meta['absnames'][0]
        if absname0.startswith('_auto_ivc.'):
            src_prom = absname0
        else:
            src_prom = group._resolver.abs2prom(absname0, 'output')

        group._manual_connections[tgt_prom] = (src_prom, edge_meta.get('src_indices', None),
                                               edge_meta.get('flat_src_indices', None))

    def get_all_conns(self, model):
        """
        Get a dict of global connections 'owned' by a group, keyed by the group's pathname.

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
        conns = {}
        abs_ins = model._var_allprocs_abs2meta['input']
        discrete_ins = model._var_allprocs_discrete['input']
        for node in self.nodes():
            abs_out = node[1]
            if self.in_degree(node) == 0:  # root node
                for _, abs_in in self.leaf_input_iter(node):
                    assert abs_in in abs_ins or abs_in in discrete_ins
                    common = common_subpath((abs_out, abs_in))
                    if common not in conns:
                        conns[common] = {}
                    conns[common][abs_in] = abs_out

        return conns

    def get_root(self, node):
        in_degree = self.in_degree
        for n in self.up_tree_iter(node):
            if n[0] == 'o' and in_degree(n) == 0:
                return n

    def get_root_input(self, node):
        if node[0] != 'i':
            return None

        root = None
        for n in self.up_tree_iter(node):
            if n[0] == 'i':
                root = n

        return root

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
                current = current[idx]
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
        try:
            if indices_list:
                chain = [arr]
                for idx in indices_list:
                    chain.append(chain[-1][idx])

                if np.shape(val) != () and np.squeeze(val).shape != np.squeeze(chain[-1]).shape:
                    err = (f"Value shape {np.squeeze(val).shape} does not match shape "
                           f"{np.squeeze(chain[-1]).shape} of the destination")
            else:
                arr[:] = val
                return
        except Exception as err:
            msg = str(err)
        else:
            msg = ''

        if msg:
            raise ValueError(f"Failed to set value of '{node[1]}': {msg}.")

        if np.isscalar(chain[-1]):
            chain[-1] = val
        else:
            chain[-1][:] = val

        for i in range(len(chain) - 2, -1, -1):
            sub = chain[i + 1]
            prev = chain[i]
            idx = indices_list[i]
            if sub.base is not prev:
                prev[idx] = sub

    def convert_get(self, val, src_units, tgt_units, src_inds_list=(), units=None, indices=None):
        if indices is not None:
            src_inds_list = list(src_inds_list) + [indices]

        val = self.get_subarray(val, src_inds_list)

        if units is None:
            units = tgt_units

        if units is not None:
            if src_units is None:
                issue_warning(f"Value has no units so can't convert to units of '{units}'.")
                return val
            elif src_units != units:
                try:
                    scale, offset = unit_conversion(src_units, units)
                except Exception:
                    raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                    f"'{units}'.")

                return (val + offset) * scale

        return val

    def convert_set(self, val, src_units, tgt_units, src_inds_list=(), units=None, indices=None):
        if indices is not None:
            src_inds_list = list(src_inds_list) + [indices]

        val = self.get_subarray(val, src_inds_list)

        if units is None:
            units = tgt_units

        if units is not None:
            if src_units is None:
                issue_warning(f"Value has no units so can't convert to units of '{units}'.")
                return val
            elif src_units != units:
                try:
                    scale, offset = unit_conversion(units, src_units)
                except Exception:
                    raise TypeError(f"Can't express value with units of '{src_units}' in units of "
                                    f"'{units}'.")

                return (val + offset) * scale

        return val

    def create_node_label(self, node):
        meta = self.nodes[node]
        units = meta['units']
        if units is None:
            units = ''
        else:
            units = f"<TR><TD><b>{units}</b></TD></TR>"

        shape = meta['_shape']
        if shape is None:
            shape = ''
        else:
            shape = f"<TR><TD><i>{shape}</i></TD></TR>"

        # Get the combined name and check for custom formatting first
        name = self.combined_name(node)

        # return f"{name}\n{units}\n{shape}"

        if units or shape:
            return f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="1" CELLPADDING="0"><TR><TD>' \
                f'{name}</TD></TR>{units}{shape}</TABLE>>'
        else:
            return name

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
            if data['io'] == 'i':
                newdata['fillcolor'] = GRAPH_COLORS['input']
            else:
                newdata['fillcolor'] = GRAPH_COLORS['output']

            newdata['label'] = self.create_node_label(node)
            newdata['tooltip'] = (data['pathname'], data['rel_name'])
            newdata['style'] = 'filled,rounded'
            newdata['shape'] = 'box'  # Use box shape with rounded corners
            newdata['pathname'] = data['pathname']
            newdata['rel_name'] = data['rel_name']
            yield node, newdata

    def drawable_edge_iter(self, pathname='', show_cross_boundary=True):
        """
        Yield edges usable in a pydot graph.

        Yields
        ------
        tuple of the form (u, v, data)
            The edge and its metadata.
        """
        if pathname:
            pathname = pathname + '.'

        for u, v, data in self.edges(data=True):
            style = data.get('style')

            if pathname:
                u_internal = self.startswith(pathname, u)
                if not u_internal:  # show as an external connection
                    if not show_cross_boundary:
                        continue
                    style = 'dotted'

                v_internal = self.startswith(pathname, v)
                if not v_internal:  # show as an external connection
                    if not show_cross_boundary:
                        continue
                    style = 'dotted'

                if not (u_internal or v_internal):
                    continue

            newdata = {}
            if 'style' in data:
                newdata['style'] = style

            if 'src_indices' in data and data['src_indices'] is not None:
                newdata['label'] = str(data['src_indices'])

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
            # change color of the variable node to highlight it
            G.nodes[varnode]['fillcolor'] = GRAPH_COLORS['highlight']
            root = self.get_root(varnode)
            nodes = [root]
            nodes.extend(v for _, v in dfs_edges(G, root))
            G = nx.subgraph(G, nodes)

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
            mnames = ['units', 'discrete', '_shape']
            dismeta = {k: meta[k] for k in mnames if k in meta and meta[k] is not None}
            print(f"{indent}{node[1]}  {dismeta}")

    def serve(self, port=8001, open_browser=True):
        """Serve connection graph web UI."""
        from openmdao.visualization.conn_graph_ui import ConnGraphHandler

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
