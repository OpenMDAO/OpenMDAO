from pprint import pformat
import sys
from difflib import get_close_matches
import networkx as nx
from networkx import dfs_edges

from openmdao.visualization.graph_viewer import write_graph
from openmdao.utils.general_utils import common_subpath
from openmdao.utils.array_utils import array_connection_compatible


class AllConnGraph(nx.DiGraph):
    """
    A graph for all connection info.  Covers manual, implicit, and all promotions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fullname(self, node):
        """
        Get full name of node, including absolute names if they differ from the promoted name.

        Parameters
        ----------
        node : tuple
            The node to get the full name of.  Tuple of the form (io, name), where io is either
            'i' or 'o'.

        Returns
        -------
        str
            The full name of the node.
        """
        meta = self.nodes[node]
        absnames = meta['absnames']
        if len(absnames) == 1:
            absnames = absnames[0]

        if node[1] == absnames:
            return node[1]

        return f'{node[1]} ({absnames})'

    def check_add_edge(self, group, src, tgt, **kwargs):
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

    def get_node_meta(self, group, prom_name, io):
        absnames = group._resolver.absnames(prom_name, io, report_error=False)
        if absnames is None:
            # auto_ivcs may not have been added to the graph yet
            if prom_name.startswith('_auto_ivc.'):
                absnames = [prom_name]
            else:
                raise KeyError(f"{group.msginfo}: '{prom_name} not found.")

        key = (io[0], '.'.join((group.pathname, prom_name)) if group.pathname else prom_name)

        return key, {'io': io[0], 'pathname': group.pathname,
                     'prom_name': prom_name, 'absnames': absnames, 'src_inds_list': []}

    def decorate_nodes(self):
        for _, data in self.nodes(data=True):
            if data['io'] == 'i':
                data['fillcolor'] = 'peachpuff3'
            else:
                data['fillcolor'] = 'skyblue3'
            data['label'] = \
                '.'.join((data['pathname'],
                          data['prom_name'])) if data['pathname'] else data['prom_name']
            data['tooltip'] = (data['io'], data['pathname'], data['prom_name'])
            data['style'] = 'filled'

    def get_path_prom(self, node):
        meta = self.nodes[node]
        return meta['pathname'], meta['prom_name']

    def add_promotion(self, io, group, prom_name, subsys, sub_prom, pinfo=None):
        if io == 'input':
            src, src_kwargs = self.get_node_meta(group, prom_name, io)
            tgt, tgt_kwargs = self.get_node_meta(subsys, sub_prom, io)
        else:
            src, src_kwargs = self.get_node_meta(subsys, sub_prom, io)
            tgt, tgt_kwargs = self.get_node_meta(group, prom_name, io)

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

            src, src_kwargs = self.get_node_meta(group, prom_src, src_io)
            tgt, tgt_kwargs = self.get_node_meta(group, prom_tgt, tgt_io)

            if src not in self:
                self.add_node(src, **src_kwargs)
            if tgt not in self:
                self.add_node(tgt, **tgt_kwargs)

            input_input = src_io == 'input' and tgt_io == 'input'
            self.check_add_edge(group, src, tgt, src_indices=src_indices,
                                flat_src_indices=flat, input_input=input_input)

    def add_auto_ivcs(self, group):
        assert group.pathname == ''

        # this occurs before the auto_ivc variables actually exist
        dangling_inputs = [n for n in self.nodes() if n[0] == 'i' and self.in_degree(n) == 0]
        for i, n in enumerate(dangling_inputs):
            node, meta = self.get_node_meta(group, f'_auto_ivc.v{i}', 'output')
            self.add_node(node, **meta)
            self.add_edge(node, n)

    def add_implicit_connections(self, group):
        assert group.pathname == ''

        for prom_name in group._resolver.get_implicit_conns():
            src, src_kwargs = self.get_node_meta(group, prom_name, 'output')
            if src not in self:
                self.add_node(src, **src_kwargs)

            tgt, tgt_kwargs = self.get_node_meta(group, prom_name, 'input')
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
        abs2meta_out = group._var_allprocs_abs2meta['output']
        abs2meta_in = group._var_allprocs_abs2meta['input']
        nodes = self.nodes
        edges = self.edges
        src_meta = nodes[src]
        # use '_shape' since 'shape' has meaning in graphviz and will alter the node appearance
        src_meta['_shape'] = abs2meta_out[src[1]]['shape']
        abs_in = None

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
                    abs_in = v[1]

            vmeta['_shape'] = shape
            vmeta['src_inds_list'] = src_inds_list

        if abs_in is not None:
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
                                     f"'{self.fullname(('i', abs_in))}'{sistr}: shape {expected_shape} "
                                     f"!= {abs_in_shape}.", ident=(src[1], abs_in))

    def update_shapes(self, model):
        """
        Update the shape, src_shape and src_indices for all nodes in the graph.

        Parameters
        ----------
        model : Group
            The top level group.
        """
        abs2meta = model._var_allprocs_abs2meta['output']

        # loop over all absolute output names and traverse until we find an input node
        for node, data in self.nodes(data=True):
            if node[0] == 'o' and self.in_degree(node) == 0:  # absolute src node
                if node[1] in abs2meta:  # only do the following for continuous variables
                    self.update_src_indices(node, model)

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
        edge_meta = {k: v for k, v in self.edges[inp_src, tgt].items() if k != 'input_input'}
        self.remove_edge(inp_src, tgt)
        self.add_edge(new_src, tgt, **edge_meta)
        tgt_syspath, tgt_prom = self.get_path_prom(tgt)
        if tgt_syspath:
            sub = group._get_subsystem(tgt_syspath)
            if sub is not None and sub._is_local:
                del sub._manual_connections[tgt_prom]
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
                for abs_in in self.get_leaf_inputs(node):
                    assert abs_in in abs_ins or abs_in in discrete_ins
                    common = common_subpath((abs_out, abs_in))
                    if common not in conns:
                        conns[common] = {}
                    conns[common][abs_in] = abs_out

        return conns

    def get_leaf_inputs(self, node):
        for _, node in dfs_edges(self, node):
            if node[0] == 'i' and self.out_degree(node) == 0:
                yield node[1]

    def display(self):
        self.decorate_nodes()
        write_graph(self)

    def dump(self):
        skip = {'style', 'tooltip', 'fillcolor', 'label'}
        print("\nNodes:")
        for node, data in sorted(self.nodes(data=True),
                                 key=lambda x: (x[1]['label'].count('.'), x[1]['label'])):
            dct = {'node': node}
            dct.update({k: v for k, v in data.items() if k not in skip})
            print(data['label'], pformat(dct))

        print("\nEdges:")
        for u, v, data in sorted(self.edges(data=True), key=lambda x: (x[0], x[1])):
            dct = {k: v for k, v in data.items() if k not in skip and v is not None}
            print(self.nodes[u]['label'], '->', self.nodes[v]['label'], pformat(dct))

        print()

