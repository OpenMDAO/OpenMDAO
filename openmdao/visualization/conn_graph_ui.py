#!/usr/bin/env python3
"""
Simple Conection Graph Web UI with Graphviz integration.

This creates a web interface that:
1. Uses Graphviz for proper graph layouts
2. Serves focused views of connection graphs
"""

import os
import json
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import networkx as nx

from openmdao.visualization.conn_graph import GRAPH_COLORS
from openmdao.utils.file_utils import _load_and_exec
import openmdao.utils.hooks as hooks


class ConnGraphHandler(SimpleHTTPRequestHandler):
    """Custom handler for serving the connection graph web interface."""

    def __init__(self, conn_graph, *args, **kwargs):
        self.conn_graph = conn_graph
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/':
            self.serve_main_page()
        elif parsed_path.path == '/api/graph_info':
            self.serve_graph_info()
        elif parsed_path.path.startswith('/api/subsystem/'):
            subsystem = unquote(parsed_path.path.replace('/api/subsystem/', ''))
            self.serve_subsystem_graph(subsystem)
        elif parsed_path.path.startswith('/api/variable/'):
            variable = unquote(parsed_path.path.replace('/api/variable/', ''))
            self.serve_variable_graph(variable)
        elif parsed_path.path == '/api/search':
            self.serve_search()
        else:
            super().do_GET()

    def serve_main_page(self):
        """Serve the main HTML page."""
        html_content = self.get_html_template()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())

    def serve_graph_info(self):
        """Serve basic graph information."""
        # Get all nodes with their data
        nodes_data = {}
        for node_id, node_data in self.conn_graph.nodes(data=True):
            # Convert tuple node_id to JSON-serializable format
            if isinstance(node_id, tuple) and len(node_id) == 2:
                node_id_json = [node_id[0], node_id[1]]
            else:
                node_id_json = str(node_id)

            nodes_data[json.dumps(node_id_json)] = {
                'rel_name': node_data.get('rel_name', ''),
                'pathname': node_data.get('pathname', ''),
                'io': node_id[0] if isinstance(node_id, tuple) and len(node_id) == 2 else '',
                'fillcolor': node_data.get('fillcolor', '')
            }

        info = {
            'success': True,
            'nodes': len(self.conn_graph.nodes()),
            'edges': len(self.conn_graph.edges()),
            'subsystems': self.get_subsystems(),
            'nodes_data': nodes_data,
            'graph_colors': GRAPH_COLORS
        }
        self.send_json_response(info)

    def serve_subsystem_graph(self, subsystem):
        """Serve graph for a specific subsystem."""
        try:
            subgraph = self.conn_graph.get_drawable_graph(subsystem)
            pydot_graph = nx.drawing.nx_pydot.to_pydot(subgraph)
            try:
                graphviz_svg = pydot_graph.create_svg().decode('utf-8')
            except Exception:
                graphviz_svg = self.create_text_graph(subgraph, f"System: {subsystem}")

            # get help colors from the subgraph (where fillcolor is set)
            incolor = outcolor = None
            for node_id, node_data in subgraph.nodes(data=True):
                # Look for input nodes (starting with 'i')
                if incolor is None and node_id[0] == 'i':
                    incolor = node_data.get('fillcolor', None)
                # Look for output nodes (starting with 'o')
                elif outcolor is None and node_id[0] == 'o':
                    outcolor = node_data.get('fillcolor', None)
                # Stop once we have both colors
                if incolor is not None and outcolor is not None:
                    break

            help_colors = {'i': incolor, 'o': outcolor}

            # Convert nodes to a format the frontend can use
            nodes_data = {}
            for node_id, node_data in subgraph.nodes(data=True):
                # Extract the actual variable information from the node metadata
                # The node_data contains the original rel_name and pathname
                rel_name = node_data.get('rel_name', '')
                pathname = node_data.get('pathname', '')
                io_type = node_id[0]

                # If io_type is not in node_data, extract from node_id tuple
                if not io_type and isinstance(node_id, tuple) and len(node_id) == 2:
                    io_type = node_id[0]  # 'i' or 'o'

                nodes_data[str(node_id)] = {
                    'rel_name': rel_name,
                    'pathname': pathname,
                    'io': io_type,
                    'fillcolor': node_data.get('fillcolor', '')
                }

            response = {
                'success': True,
                'subsystem': subsystem,
                'nodes': len(subgraph.nodes()),
                'edges': len(subgraph.edges()),
                'nodes_data': nodes_data,
                'help_colors': help_colors,
                'svg': graphviz_svg
            }
        except Exception as e:
            print(f"Error in serve_subsystem_graph: {e}")
            # Provide more helpful error message
            error_msg = str(e)
            if "not found" in error_msg.lower():
                error_msg = f"Subsystem '{subsystem}' not found. Try searching for a different " \
                    "subsystem name."
            response = {
                'success': False,
                'error': error_msg
            }

        self.send_json_response(response)

    def serve_variable_graph(self, variable):
        """Serve graph focused on a specific variable."""
        try:
            # First try the variable name as-is
            try:
                pydot_graph = self.conn_graph.get_pydot_graph(varname=variable)
            except Exception:
                raise ValueError(f"Variable '{variable}' not found in connection graph")

            # pydot_graph is already a pydot.Dot object, so we can use it directly
            try:
                graphviz_svg = pydot_graph.create_svg().decode('utf-8')
            except Exception:
                # If SVG generation fails, create a text representation
                # We need to convert the pydot graph to NetworkX for the text fallback
                try:
                    nx_graph = nx.drawing.nx_pydot.from_pydot(pydot_graph)
                    graphviz_svg = self.create_text_graph(nx_graph, f"Variable: {variable}")
                except Exception:
                    graphviz_svg = f"<div style='padding: 20px; color: red;'>Error generating " \
                        f"graph for variable: {variable}</div>"

            # Count nodes and edges from the pydot graph
            nodes_count = len(pydot_graph.get_nodes())
            edges_count = len(pydot_graph.get_edges())

            response = {
                'success': True,
                'variable': variable,
                'nodes': nodes_count,
                'edges': edges_count,
                'svg': graphviz_svg
            }
        except Exception as e:
            # Provide more helpful error message
            error_msg = str(e)
            if "not found" in error_msg.lower():
                error_msg = f"Variable '{variable}' not found. Try searching for a different " \
                    "variable name."
            response = {'success': False, 'error': error_msg}

        self.send_json_response(response)

    def serve_search(self):
        """Serve search results."""
        query = self.get_query_param('q', '').lower()
        if not query:
            self.send_json_response({'results': []})
            return

        results = []

        # Search nodes
        for node_id, node_data in self.conn_graph.nodes(data=True):
            # Try different name fields
            name = node_data['rel_name']
            if query in name.lower():
                results.append({
                    'type': 'variable',
                    'name': name,
                    'path': node_data['pathname'],
                    'io_type': node_id[0]
                })

        # Search subsystems
        subsystems = self.get_subsystems()
        for subsystem in subsystems:
            if query in subsystem.lower():
                results.append({
                    'type': 'subsystem',
                    'name': subsystem,
                    'path': subsystem
                })

        self.send_json_response({'results': results[:20]})

    def get_subsystems(self):
        """Get list of unique subsystems."""
        subsystems = {'model'}
        for _, node_data in self.conn_graph.nodes(data=True):
            pathname = node_data.get('pathname', '')
            if pathname:
                subsystems.add(pathname)

        return sorted(subsystems)

    def create_text_graph(self, subgraph, title):
        """Create a simple text-based graph representation when Graphviz is not available."""
        html = '<div style="font-family: monospace; padding: 20px;">'
        html += f'<h3>{title}</h3>'
        html += f'<p><strong>Nodes:</strong> {len(subgraph.nodes())}</p>'

        # Show nodes
        html += '<h4>Variables:</h4><ul>'
        for node_id, node_data in subgraph.nodes(data=True):
            name = node_data['rel_name']
            io_type = node_id[0]
            io_label = 'Input' if io_type == 'i' else 'Output'
            html += f'<li><strong>{name}</strong> ({io_label})</li>'
        html += '</ul>'

        # Show connections
        html += '<h4>Connections:</h4><ul>'
        for edge in subgraph.edges():
            source_data = subgraph.nodes[edge[0]]['attrs']
            target_data = subgraph.nodes[edge[1]]['attrs']
            source_name = source_data['rel_name']
            target_name = target_data['rel_name']
            html += f'<li>{source_name} → {target_name}</li>'
        html += '</ul>'

        html += '<p><em>Note: Using text representation due to graph generation error</em></p>'
        html += '</div>'

        return html

    def get_query_param(self, param, default=''):
        """Get query parameter value."""
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        return query_params.get(param, [default])[0]

    def send_json_response(self, data):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def get_html_template(self):
        """Get the HTML template."""
        fpath = os.path.join(os.path.dirname(__file__), 'conn_graph_ui_template.html')
        with open(fpath, "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content


def _conn_graph_setup_parser(parser):
    """
    Set up the openmdao subparser for the 'openmdao graph' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.description = ('This command requires pydot and graphviz to be installed. '
                          'It displays the connection graph for the specified variable or system.')
    parser.add_argument('file', nargs=1, help='Python file containing the model.')
    parser.add_argument('--problem', action='store', dest='problem', help='Problem name')
    parser.add_argument('--port', action='store', dest='port', help='Port number')


def _conn_graph_cmd(options, user_args):
    """
    Return the post_setup hook function for 'openmdao graph'.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    def _view_graph(model):
        model._get_conn_graph().serve(port=options.port)

    # register the hooks
    def _set_dyn_hook(prob):
        hooks._register_hook('_setup_part2', class_name='Group', inst_id='',
                             post=_view_graph, exit=True)
        hooks._setup_hooks(prob.model)

    # register the hooks
    hooks._register_hook('setup', 'Problem', pre=_set_dyn_hook, ncalls=1)
    _load_and_exec(options.file[0], user_args)


if __name__ == "__main__":
    # Example usage
    from example_conn_graph_ui import create_complex_engineering_model

    # Create model
    prob = create_complex_engineering_model()
    prob.setup()

    # Start web UI
    prob.model._get_conn_graph().serve(port=8001)

    # serve_simple_conn_graph_ui(conn_graph, port=8001)
