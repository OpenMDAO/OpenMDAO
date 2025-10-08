from pprint import pformat
import sys
from difflib import get_close_matches
import networkx as nx
from networkx import dfs_edges, dfs_postorder_nodes
import numpy as np
from numbers import Number
from collections import deque

from openmdao.visualization.graph_viewer import write_graph
from openmdao.utils.general_utils import common_subpath, is_undefined, shape2tuple, \
    ensure_compatible
from openmdao.utils.array_utils import array_connection_compatible
from openmdao.utils.units import simplify_unit, is_compatible
from openmdao.utils.indexer import get_subarray, set_subarray
from openmdao.utils.om_warnings import issue_warning
from openmdao.utils.units import unit_conversion


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


def convert_get(val, src_units, tgt_units, src_inds_list=(), units=None, indices=None):
    if indices is not None:
        src_inds_list = list(src_inds_list) + [indices]

    val = get_subarray(val, src_inds_list)

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


def convert_set(val, src_units, tgt_units, src_inds_list=(), units=None, indices=None):
    if indices is not None:
        src_inds_list = list(src_inds_list) + [indices]

    val = get_subarray(val, src_inds_list)

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


class AllConnGraph(nx.DiGraph):
    """
    A graph for all connection info.  Covers manual, implicit, and all promotions.

    Every connection in the graph forms a tree structure with an absolute output name at its
    root and all connected absolute input names as the leaf nodes.

    Node keys are tuples of the form (io, name), where io is either 'i' or 'o', and name is the
    name (either promoted or absolute) of a variable in the model.

    src_indices are stored in the edges between nodes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fullname(self, node):
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
        absnames = meta['absnames']
        if len(absnames) == 1:
            absnames = absnames[0]

        if node[1] == absnames:
            return node[1]

        return f'{node[1]} ({sorted(absnames)})'

    def get_val_from_node(self, val, src_node, tgt_node=None, units=None, indices=None):
        src_units = self.nodes[src_node]['units']

        if tgt_node is None:
            tgt_units = None
            tgt_inds_list = ()
        else:
            tgt_units, tgt_inds_list = self.get_conversion_info(tgt_node)

        return convert_get(val, src_units, tgt_units, tgt_inds_list, units, indices)

    def set_val_from_node(self, model, tgt_node, val, units=None, indices=None):
        src_node = self.get_root(tgt_node)
        src_meta = self.nodes[src_node]
        if src_meta['discrete']:
            if src_node[1] in model._discrete_outputs:
                model._discrete_outputs[src_node[1]] = val
            return

        # variables are continuous, so deal with src_indices and units if needed
        src_units, _ = self.get_conversion_info(src_node)
        if tgt_node is None:
            tgt_units = None
            tgt_inds_list = ()
        else:
            tgt_units, tgt_inds_list = self.get_conversion_info(tgt_node)

        # do unit conversion on given val if needed
        val = convert_set(val, src_units, tgt_units, (),  units)

        if indices is None:
            inds = tgt_inds_list
        else:
            inds = list(tgt_inds_list) + [indices]

        srcval = model._abs_get_val(src_node[1], get_remote=False)
        set_subarray(srcval, inds, val)

    def find_node(self, system, varname, io=None):
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
        if io is None:
            node = ('i', varname)
            if node not in self:
                node = ('o', varname)
        else:
            node = (io[0], varname)

        if node in self:
            return node

        raise KeyError(f"{system.msginfo}: Variable '{varname}' not found in connection graph.")

        # # varname may be a promoted name that isn't in the graph, so try to find the nearest
        # # ancestor node if it's an input, or the absolute src node if it's an output.
        # if io is None:
        #     io = system._resolver.get_iotype(varname, report_error=True)

        # if io == 'input':
        #     node, _ = self.get_nearest_input_up(system, varname)
        # else:
        #     absnames = system._resolver.absnames(varname, io)
        #     node = ('o', absnames[0])
        #     assert node in self

        # return node, False

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
        while True:
            for node in self.predecessors(node):  # no node will have more than one predecessor
                yield node
                break
            else:
                break

    # def get_nearest_input_up(self, system, varname):
    #     """
    #     Get the nearest ancestor input node to the given variable.

    #     If varname is in the graph, just return that node.

    #     Parameters
    #     ----------
    #     system : System
    #         The system to get the nearest ancestor input node for.
    #     varname : str
    #         The variable name to get the nearest ancestor input node for.

    #     Returns
    #     -------
    #     tuple of the form ('i', name)
    #         The node for the nearest ancestor input node.
    #     bool
    #         True if the given node was already in the graph.
    #     """
    #     node = ('i', varname)

    #     if node in self:
    #         return node, True
    #     else:
    #         absnames = system._resolver.absnames(varname, 'input', report_error=True)
    #         node = ('i', absnames[0])
    #         assert node in self

    #         # traverse up the tree until we find a parent pathname
    #         mypathname = system.pathname
    #         for node in self.up_tree_iter(node):
    #             if node[0] == 'o':
    #                 break  # no more input ancestors left

    #             meta = self.nodes[node]
    #             par_prefix = meta['pathname'] + '.' if meta['pathname'] else ''

    #             if mypathname.startswith(par_prefix):
    #                 return node, False

    #         raise KeyError(f"{system.msginfo}: Input '{varname}' not found.")

    # def get_nearest_input_down(self, system, varname):
    #     """
    #     Get the nearest descendant input node to the given variable.

    #     If varname is in the graph, just return that node.

    #     Parameters
    #     ----------
    #     system : System
    #         The system to get the nearest descendant input node for.
    #     varname : str
    #         The variable name to get the nearest descendant input node for.

    #     Returns
    #     -------
    #     tuple of the form ('i', name)
    #         The node for the nearest descendant input node.
    #     bool
    #         True if the given node was already in the graph.
    #     """
    #     # first, find the nearest ancestor node
    #     node, exact = self.get_nearest_input_up(system, varname)
    #     if exact:
    #         return node, True

    #     prefix = system.pathname + '.' if system.pathname else ''
    #     nodes = self.nodes
    #     for succ in  self.successors(node):
    #         meta = nodes[succ]
    #         if meta['pathname'].startswith(prefix):
    #             return succ, False
    #     else:
    #         raise KeyError(f"{system.msginfo}: Input '{varname}' not found.")

    def get_conversion_info(self, node):
        meta = self.nodes[node]
        return meta['units'], meta['src_inds_list']

    def check_add_edge(self, group, src, tgt, **kwargs):
        if (src, tgt) in self.edges():
            return True

        if self.in_degree(tgt) != 0:
            preds = [self.fullname(p) for p in sorted(self.predecessors(tgt))]
            if len(preds) == 1:
                preds = preds[0]

            group._collect_error(
                f"{group.msginfo}: Target '{self.fullname(tgt)}' cannot be connected to "
                f"'{self.fullname(src)}' "
                f"because it's already connected to '{preds}'.",
                ident=(src, tgt))
            group._bad_conn_vars.update((tgt, src))
            return False

        self.add_edge(src, tgt, **kwargs)
        return True

    def create_node_meta(self, group, name, io):
        # abs2meta = group._var_abs2meta[io]
        # all_abs2meta = group._var_allprocs_abs2meta[io]
        shape = val = units = discrete = varmeta =None

        # if group._resolver.is_abs(name, io):
        #     try:
        #         varmeta = (all_abs2meta[name], abs2meta[name] if name in abs2meta else None)
        #         shape = all_abs2meta[name]['shape']
        #         units = all_abs2meta[name]['units']
        #         if varmeta[1] is not None:
        #             shape = varmeta[1]['shape']
        #         discrete = False
        #     except KeyError:
        #         # assume it's a discrete variable
        #         rel_name = name[len(group.pathname) + 1:] if group.pathname else name
        #         loc_meta = group._var_discrete[io][rel_name] \
        #             if rel_name in group._var_discrete[io] else None
        #         if loc_meta is not None:
        #             val = loc_meta['val']
        #         varmeta = (group._var_allprocs_discrete[io][name], loc_meta)
        #         discrete = True
        # else:
        #     varmeta = (None, None)

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

    def update_node_meta(self, group, name, io):
        abs2meta = group._var_abs2meta[io]
        all_abs2meta = group._var_allprocs_abs2meta[io]
        shape = val = units = None
        discrete = None
        if group._resolver.is_abs(name, io):
            try:
                varmeta = (all_abs2meta[name], abs2meta[name] if name in abs2meta else None)
                shape = all_abs2meta[name]['shape']
                units = all_abs2meta[name]['units']
                if varmeta[1] is not None:
                    shape = varmeta[1]['shape']
                discrete = False
            except KeyError:
                # assume it's a discrete variable
                rel_name = name[len(group.pathname) + 1:] if group.pathname else name
                loc_meta = group._var_discrete[io][rel_name] \
                    if rel_name in group._var_discrete[io] else None
                if loc_meta is not None:
                    val = loc_meta['val']
                varmeta = (group._var_allprocs_discrete[io][name], loc_meta)
                discrete = True
        else:
            varmeta = (None, None)

        key = (io[0], '.'.join((group.pathname, name)) if group.pathname else name)

        return key, {'src_inds_list': [],
                     'units': units, 'val': val, '_shape': shape, 'meta': varmeta,
                     'discrete': discrete}

    def add_drawing_info(self):
        """
        Add metadata for drawing the graph.
        """
        for _, data in self.nodes(data=True):
            if data['io'] == 'i':
                data['fillcolor'] = 'peachpuff3'
            else:
                data['fillcolor'] = 'skyblue3'
            data['label'] = \
                '.'.join((data['pathname'],
                          data['rel_name'])) if data['pathname'] else data['rel_name']
            data['tooltip'] = (data['io'], data['pathname'], data['rel_name'])
            data['style'] = 'filled'

    def get_path_prom(self, node):
        meta = self.nodes[node]
        return meta['pathname'], meta['rel_name']

    def add_abs_variable_meta(self, group):
        assert group.pathname == ''
        nodes = self.nodes
        for io in ['input', 'output']:
            loc = group._var_abs2meta[io]
            for name, meta in group._var_allprocs_abs2meta[io].items():
                # update/create all absolute variable nodes in the graph
                node =  (io[0], name)
                if node not in nodes:
                    node, node_meta = self.create_node_meta(group, name, io)
                    self.add_node(node, **node_meta)

                node_meta = nodes[node]

                if name in loc:
                    node_meta['val'] = loc[name]['val']

                node_meta['_shape'] = meta['shape']
                node_meta['units'] = meta['units']
                node_meta['discrete'] = False

                node_meta['meta'] = (meta, loc[name] if name in loc else None)

            loc = group._var_discrete[io]
            for name, meta in group._var_allprocs_discrete[io].items():
                node = (io[0], name)
                if node not in nodes:
                    if node[0] == 'i':  # only create missing input nodes
                        node, node_meta = self.create_node_meta(group, name, io)
                        self.add_node(node, **node_meta)

                node_meta = nodes[node]
                node_meta['discrete'] = True
                if name in loc:  # at top level, keys of _var_discrete are also abs names
                    node_meta['val'] = loc[name]['val']
                node_meta['meta'] = (meta, loc[name] if name in loc else None)

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
            conn_dict = group._manual_connections if input_input else None
            self.check_add_edge(group, src, tgt, src_indices=src_indices,
                                flat_src_indices=flat, input_input=input_input,
                                conn_dict=conn_dict)

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

                defaults['val'] = val
                defaults['shape'] = val.shape

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

        dmeta = [(succ[1],self.nodes[succ]['discrete']) for succ in succs]
        base = dmeta[0][1]
        for meta in dmeta:
            name, discrete = meta
            if discrete != base:
                discvars = [n for n, d in dmeta if d]
                nondiscvars = [n for n, d in dmeta if not d]
                group._collect_error(f"{group.msginfo}: Variable '{name}' "
                f"connects to discrete variables {sorted(discvars)} and continuous variables "
                f"{sorted(nondiscvars)}. Discrete and continuous variables cannot be connected.")
                return

        node_meta['discrete'] = base

    def rollup_units(self, group, node, node_meta, succs, default):
        base = None
        same = True
        units_list = []
        for succ in succs:
            units = self.nodes[succ].get('units', None)
            if units is not None:
                units_list.append(units)
                if base is None:
                    base = units
                    sbase = succ[1]
                else:
                    if not is_compatible(base, units):
                        raise ValueError(f"'{succ[1]}' units of '{units}' are incompatible "
                                         f"with '{sbase}' units of '{base}'.")
                    same &= base == units

        if default is None:
            if not same:
                absname = self.nodes[node]['absnames'][0]
                prom = group._resolver.abs2prom(absname, 'input')
                group._collect_error(f"{group.msginfo}: No default units have been set for input "
                                     f"'{self.fullname(node)}' so the choice of units between "
                                     f"'{sorted(units_list)}' is ambiguous. Call "
                                     f"model.set_input_defaults('{prom}', units=?) to remove "
                                     "the ambiguity.")
            node_meta['units'] = base
            return

        if base is None:
            node_meta['units'] = default
            return

        # base and default are not None
        if is_compatible(base, default):
            # default overrides any node value as long as it's compatible
            node_meta['units'] = default
            return

        group._collect_error(f"{group.msginfo}: Input '{self.fullname(node)}' default units "
                             f"'{default.name()}' and '{self.fullname(sbase)}' units of "
                             f"'{base.name()}' are incompatible.")

    def rollup_valshape(self, group, node, node_meta, succs, defaults):
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

        for succ in succs:
            sdefaults = self.nodes[succ].get('defaults', {})
            src_inds = None if discrete else self.edges[node, succ].get('src_indices', None)

            if src_inds is None:
                val = self.nodes[succ].get('val', None)
                if discrete:
                    _shape = None
                else:
                    _shape = self.nodes[succ].get('_shape', None)
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
                        group._collect_error(f"{group.msginfo}: '{self.fullname(succ)}' shape of "
                                             f"'{_shape}' is incompatible with "
                                             f"'{self.fullname(sbase_shape)}' "
                                             f"shape of '{shape_base}'.")

            if val is not None:
                if val_base is None:
                    val_base = val
                    sbase_val = succ
                else:
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
                                     f"'{self.fullname(node)}' but different values feed into it. "
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
                group._collect_error(f"{group.msginfo}: Input '{self.fullname(node)}' default val "
                                     f"'{default_val}' and '{self.fullname(sbase_val)}' val of "
                                     f"'{val_base}' are incompatible.")

        if not shape_good:
            group._collect_error(f"{group.msginfo}: Input '{self.fullname(node)}' default shape "
                                 f"'{default_shape}' and '{self.fullname(sbase_shape)}' shape of "
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
                                          defaults.get('units', None))
                    self.rollup_valshape(group, node, branch_meta, succs, defaults)
                else:  # leaf node (absolute input)
                    units = defaults.get('units', None)
                    val = defaults.get('val', None)
                    if units is not None:
                        if branch_meta['units'] is None or is_compatible(units,
                                                                         branch_meta['units']):
                            branch_meta['units'] = units
                        else:
                            group._collect_error(
                                f"{group.msginfo}: Input '{self.fullname(node)}' "
                                f"default units '{units}' and "
                                f"units of '{branch_meta['units']}' are incompatible.")

                    if val is not None:
                        if branch_meta['val'] is None or are_compatible_values(val,
                                                                               branch_meta['val']):
                            branch_meta['val'] = val
                        else:
                            group._collect_error(f"{group.msginfo}: Input '{self.fullname(node)}' "
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
        auto_nodes = []
        for i, n in enumerate(dangling_inputs):
            auto_node, meta = self.create_node_meta(group, f'_auto_ivc.v{i}', 'output')
            self.add_node(auto_node, **meta)
            self.add_edge(auto_node, n)
            auto_nodes.append(auto_node)

            self.rollup_to_node(group, auto_node, auto_ivc=True)

        return auto_nodes

    def add_implicit_connections(self, group):
        assert group.pathname == ''

        for prom_name in group._resolver.get_implicit_conns():
            src, src_kwargs = self.create_node_meta(group, prom_name, 'output')
            if src not in self:
                self.add_node(src, **src_kwargs)

            tgt, tgt_kwargs = self.create_node_meta(group, prom_name, 'input')
            if tgt not in self:
                self.add_node(tgt, **tgt_kwargs)

            self.check_add_edge(group, src, tgt, style='dotted', implicit=True)

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
        #abs2meta_out = group._var_allprocs_abs2meta['output']
        abs2meta_in = group._var_allprocs_abs2meta['input']
        nodes = self.nodes
        edges = self.edges
        #src_meta = nodes[src]
        # use '_shape' since 'shape' has meaning in graphviz and will alter the node appearance
        #if not src[1].startswith('_auto_ivc.'):
            #src_meta['_shape'] = abs2meta_out[src[1]]['shape']
        abs_ins = []

        for u, v in dfs_edges(self, src):
            umeta = nodes[u]
            vmeta = nodes[v]
            shape = umeta['_shape']
            src_inds_list = umeta['src_inds_list']

            if v[0] == 'i':  # target is an input
                src_inds = edges[u, v].get('src_indices', None)
                if src_inds is not None:
                    try:
                        src_inds.set_src_shape(shape)
                    except Exception:
                        type_exc, exc, tb = sys.exc_info()
                        group._collect_error(f"When connecting '{self.fullname(u)}' to "
                                             f"'{self.fullname(v)}': {exc}",
                                             exc_type=type_exc, tback=tb, ident=(src[1], v[1]))
                    shape = src_inds.indexed_src_shape
                    # only make a copy if we are modifying the list
                    src_inds_list = src_inds_list.copy()
                    src_inds_list.append(src_inds)

                if v[1] in abs2meta_in:
                    abs_ins.append(v[1])

            #vmeta['_shape'] = shape
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
                            sistr = f" after applying src_indices {str(src_inds_list[0])} "
                        else:
                            sistr = f" after applying src_indices {[str(s) for s in src_inds_list]} "
                    else:
                        sistr = ""
                    group._collect_error(f"When connecting '{self.fullname(src)}' to "
                                        f"'{self.fullname(('i', abs_in))}'{sistr}: shape "
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
        edges = self.edges
        to_move = []

        # loop over all absolute output names and traverse until we find an input node
        for node in self.nodes():
                # find each input branch off of the lowest level output node in this output tree
                stack = [node]
                while stack:
                    src = stack.pop()
                    for s in self.successors(src):
                        if s[0] == 'o':  # promoted output
                            stack.append(s)
                        else:
                            # traverse the input tree
                            for u, v in dfs_edges(self, s):
                                if edges[u, v].get('input_input'):
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
        if 'conn_dict' in edge_meta:
            del edge_meta['conn_dict'][tgt_prom]
        self.remove_edge(inp_src, tgt)
        self.add_edge(new_src, tgt, **edge_meta)
        if tgt_syspath:
            # sub = group._get_subsystem(tgt_syspath)
            # if sub is not None and sub._is_local:
            #     del sub._manual_connections[tgt_prom]
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
            if in_degree(n) == 0:
                return n

    def get_root_input(self, node):
        if node[0] != 'i':
            return None

        root = None
        for n in self.up_tree_iter(node):
            if n[0] == 'i':
                root = n
            else:
                break
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
        dq = deque([(self.get_root(node), 0)])

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

    def print_tree(self, name):
        if name in self:
            node = name
        else:
            node = ('i', name)
            if node not in self:
                node = ('o', name)
                if node not in self:
                    raise ValueError(f"Variable '{name}' not found in the graph.")

        for node, depth in self.get_tree_iter(node):
            indent = '  ' * depth
            print(f"{indent}{node[1]}")

    def display(self):
        self.add_drawing_info()
        write_graph(self)

    def dump_nodes(self):
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

