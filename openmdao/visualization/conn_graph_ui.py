#!/usr/bin/env python3
"""
Simple Conection Graph Web UI with Graphviz integration.

This creates a web interface that:
1. Uses Graphviz for proper graph layouts
2. Serves focused views of connection graphs
"""

import json
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import networkx as nx

from openmdao.visualization.conn_graph import GRAPH_COLORS


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
                # If that fails, try to find the variable in the graph
                try:
                    # Try to find the variable as an input
                    pydot_graph = self.conn_graph.get_pydot_graph(varname=variable)
                except Exception:
                    try:
                        # Try to find the variable as an output
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
            source_data = subgraph.nodes[edge[0]]
            target_data = subgraph.nodes[edge[1]]
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
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AllConnGraph Explorer</title>
    <style>
        * {
            box-sizing: border-box;
        }
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
        }
        .container {
            height: 100vh;
            display: flex;
            flex-direction: column;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            flex-shrink: 0;
            position: relative;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
        }
        .header button {
            position: absolute;
            right: 20px;
        }
        .content-area {
            display: flex;
            flex: 1;
            min-height: 0;
        }
        .controls {
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
            flex-shrink: 0;
        }
        .dropdown {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            background: white;
            cursor: pointer;
        }
        .dropdown:focus {
            outline: none;
            border-color: #2c3e50;
            box-shadow: 0 0 0 2px rgba(44, 62, 80, 0.2);
        }
        .search-results {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-top: none;
            background: white;
            display: none;
        }
        .search-result {
            padding: 10px;
            cursor: pointer;
            border-bottom: 1px solid #eee;
        }
        .search-result:hover {
            background: #f0f0f0;
        }
        .graph-container {
            flex: 1;
            padding: 20px;
            text-align: center;
            overflow: auto;
            display: flex;
            flex-direction: column;
            min-height: 0; /* Allow flex item to shrink */
        }
        .graph-svg {
            max-width: 100%;
            max-height: 100%;
            width: auto;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            flex: 1;
            object-fit: contain;
        }
        .graph-text {
            text-align: left;
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 20px;
            margin: 10px 0;
            flex: 1;
            overflow: auto;
        }
        .info {
            padding: 15px 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
            flex-shrink: 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }

        /* Modal styles */
        .modal {
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .modal-content {
            background-color: white;
            margin: auto;
            padding: 0;
            border-radius: 8px;
            width: 80%;
            max-width: 600px;
            max-height: 80%;
            overflow-y: auto;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }

        .modal-header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px 8px 0 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .modal-header h2 {
            margin: 0;
            font-size: 20px;
        }

        .close {
            color: white;
            font-size: 28px;
            font-weight: bold;
            cursor: pointer;
            line-height: 1;
        }

        .close:hover {
            opacity: 0.7;
        }

        .modal-body {
            padding: 20px;
        }

        .modal-body h3 {
            color: #2c3e50;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .modal-body h3:first-child {
            margin-top: 0;
        }

        .modal-body ol, .modal-body ul {
            margin: 10px 0;
            padding-left: 20px;
        }

        .modal-body li {
            margin: 5px 0;
        }

        /* Legend styles */
        .legend {
            margin: 15px 0;
        }

        .legend-item {
            display: flex;
            align-items: center;
            margin: 10px 0;
        }

        .legend-line {
            width: 40px;
            height: 3px;
            margin-right: 15px;
            background-color: transparent;
        }

        .legend-line.solid {
            border-top: 3px solid #333;
        }

        .legend-line.dashed {
            border-top: 3px dashed #333;
        }

        .legend-line.dotted {
            border-top: 3px dotted #333;
        }

        .legend-color {
            width: 20px;
            height: 20px;
            margin-right: 15px;
            border-radius: 3px;
        }

        .legend-color.input {
            /* Color will be set dynamically by JavaScript */
            background-color: #f0f0f0; /* Default gray */
        }

        .legend-color.output {
            /* Color will be set dynamically by JavaScript */
            background-color: #f0f0f0; /* Default gray */
        }

        .legend-color.highlight {
            /* Color will be set dynamically by JavaScript */
            background-color: #f0f0f0; /* Default gray */
        }

        .legend-color.boundary {
            /* Color will be set dynamically by JavaScript */
            background-color: #f0f0f0; /* Default gray */
        }

        .modal-body code {
            background-color: #f8f9fa;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }

        /* Tree structure styles */
        .sidebar {
            width: 300px;
            background: #f8f9fa;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            flex-shrink: 0;
            position: sticky;
            top: 0;
            height: 100vh;
        }

        .tree-header {
            background: #2c3e50;
            color: white;
            padding: 15px;
            font-weight: bold;
            text-align: center;
        }

        .tree-container {
            padding: 10px;
        }

        .tree-node {
            margin: 2px 0;
        }

        .tree-node-content {
            display: flex;
            align-items: center;
            padding: 5px 8px;
            cursor: pointer;
            border-radius: 3px;
            transition: background-color 0.2s;
        }

        .tree-node-content:hover {
            background-color: #e9ecef;
        }

        .tree-node-content.selected {
            background-color: #007bff;
            color: white;
        }

        .tree-toggle {
            width: 16px;
            height: 16px;
            margin-right: 8px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            color: #666;
        }

        .tree-toggle:hover {
            color: #333;
        }

        .tree-icon {
            width: 16px;
            height: 16px;
            margin-right: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
        }

        .tree-label {
            flex: 1;
            font-size: 14px;
        }

        .tree-children {
            margin-left: 20px;
            border-left: 1px solid #ddd;
            padding-left: 10px;
        }

        .tree-children.collapsed {
            display: none;
        }

        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Full-width header -->
        <div class="header">
            <h1>Connection Graph Explorer</h1>
            <button onclick="showHelp()" style="padding: 8px 16px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer;">Help</button>
        </div>

        <!-- Content area with sidebar and main content -->
        <div class="content-area">
            <!-- Sidebar with tree structure -->
            <div class="sidebar">
                <div class="tree-container" id="tree-container">
                    <div class="loading">Loading model structure...</div>
                </div>
            </div>

            <!-- Main content area -->
            <div class="main-content">

            <div class="graph-container">
                <div id="graph-content">
                    <div class="loading">
                        <h3>🔍 Select a system or variable from the tree to explore</h3>
                        <p>Use the tree on the left to:</p>
                        <ul style="text-align: left; display: inline-block;">
                            <li>Navigate through your OpenMDAO model hierarchy</li>
                            <li>View connection graphs for specific systems</li>
                            <li>Focus on individual variables and their connections</li>
                        </ul>
                    </div>
                </div>
            </div>

            <div class="info" id="graph-info">
                <strong></strong> Select a system or variable from the tree to view its connection graph.
            </div>
            </div>
        </div>
    </div>

    <!-- Help Modal -->
    <div id="help-modal" class="modal" style="display: none;">
        <div class="modal-content">
            <div class="modal-header">
                <h2>Connection Graph Explorer - Help</h2>
                <span class="close" onclick="hideHelp()">&times;</span>
            </div>
            <div class="modal-body">
                <p>A tool for visualizing and understanding how variables are connected and
                promoted within your OpenMDAO model by examining their connection graphs.
                </p>

                <ol>
                    <li>Click on any system (📁) to see all connection graphs involving that
                    system</li>
                    <li>Click on any variable (⬇️ or ⬆️) to see only the connection graph for that
                    variable</li>
                </ol>

                <h3>Connection Graph Edge Types</h3>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-line solid"></div>
                        <span><strong>Solid Line:</strong> Manual connection (from a connect()
                        call)</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line dashed"></div>
                        <span><strong>Dashed Line:</strong> Variable promotion</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-line dotted"></div>
                        <span><strong>Dotted Line:</strong> Automatic connection when promoted
                        input and output names match</span>
                    </div>
                </div>

                <h3>Connection Graph Node Colors</h3>
                <div class="legend">
                    <div class="legend-item">
                        <div class="legend-color input" id="input-color-swatch"></div>
                        <span id="input-color-text"><strong>Input variables:</strong></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color output" id="output-color-swatch"></div>
                        <span id="output-color-text"><strong>Output variables:</strong></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color highlight" id="highlight-color-swatch"></div>
                        <span id="highlight-color-text"><strong>Highlighted variables:</strong></span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color boundary" id="boundary-color-swatch"></div>
                        <span id="boundary-color-text"><strong>Variables outside of the current system:</strong></span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentGraph = null;
        let currentSubsystem = 'model';
        let cachedColors = {
            input: '#3498db',
            output: '#e74c3c',
            highlight: '#ffd700',
            boundary: '#ff6b6b'
        }; // Cache the colors
        let colorsLoaded = false; // Track if colors have been loaded

        // Color mapping function to convert color names to hex values
        function getColorHex(colorName) {
            const colorMap = {
                'peachpuff3': '#cdaf95',
                'skyblue3': '#87ceeb',
                'peachpuff': '#ffdab9',
                'skyblue': '#87ceeb'
            };
            // If it's already a hex color (starts with #), return it as-is
            if (colorName && colorName.startsWith('#')) {
                return colorName;
            }
            return colorMap[colorName] || colorName;
        }

        // UI elements
        const treeContainer = document.getElementById('tree-container');
        const graphContent = document.getElementById('graph-content');
        const graphInfo = document.getElementById('graph-info');

        // Tree data structure
        let treeData = {};
        let selectedNode = null;
        let globalGraphColors = null; // Store graph colors from initial load

        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            loadTreeStructure();
        });

        function loadTreeStructure() {
            // First get the basic graph info
            fetch('/api/graph_info')
                .then(response => response.json())
                .then(data => {
                    // Store graph colors globally
                    if (data.graph_colors) {
                        globalGraphColors = data.graph_colors;
                        loadHelpColors();
                    } else {
                        console.error('No graph_colors in API response');
                    }
                    // Build tree structure from subsystems
                    buildTreeFromSubsystems(data.subsystems);
                })
                .catch(error => {
                    console.error('Error loading tree structure:', error);
                    treeContainer.innerHTML = '<div class="loading" style="color: red;">Error loading model structure</div>';
                });
        }

        function buildTreeFromSubsystems(subsystems) {
            // Create a hierarchical structure from variable names
            const tree = {};

            // Add model as root
            tree['model'] = {
                name: 'Model',
                path: 'model',
                type: 'system',
                children: {},
                variables: []
            };

            // Load variables and build tree from variable names
            loadVariablesForTree(tree);
        }

        function loadVariablesForTree(tree) {
            // Load all variables and build tree structure from variable names

            fetch('/api/graph_info')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        buildTreeFromVariables(data.nodes_data, tree);
                    } else {
                        console.error('Error loading graph info:', data.error);
                        treeContainer.innerHTML = '<div class="loading" style="color: red;">Error loading variables</div>';
                    }
                })
                .catch(error => {
                    console.error('Error loading variables:', error);
                    treeContainer.innerHTML = '<div class="loading" style="color: red;">Error loading variables</div>';
                });
        }

        function buildTreeFromVariables(nodesData, tree) {
            // First, get the list of actual subsystems and their ancestors
            const actualSubsystems = new Set();
            Object.entries(nodesData).forEach(([nodeIdStr, nodeData]) => {
                const pathname = nodeData.pathname || '';
                if (pathname) {
                    actualSubsystems.add(pathname);
                    // Add all ancestor paths
                    const parts = pathname.split('.');
                    for (let i = 1; i < parts.length; i++) {
                        const ancestorPath = parts.slice(0, i).join('.');
                        actualSubsystems.add(ancestorPath);
                    }
                }
            });

            // Process each node to extract variable information
            const allVariables = new Map();

            Object.entries(nodesData).forEach(([nodeIdStr, nodeData]) => {
                // Parse the node ID string back to tuple format
                const nodeId = JSON.parse(nodeIdStr);
                const [io, varName] = nodeId;
                const pathname = nodeData.pathname || '';

                if (!allVariables.has(varName)) {
                    // Extract just the final part of the variable name for display
                    const displayName = varName.split('.').pop();
                    allVariables.set(varName, {
                        name: displayName,
                        fullName: varName,
                        path: pathname,
                        type: io,
                        io: io,
                        nodeId: nodeId,
                        nodeData: nodeData,
                        isBidirectional: false
                    });
                } else {
                    // Variable already exists - check if it's bidirectional
                    const existingVar = allVariables.get(varName);
                    if (existingVar.io !== io) {
                        // This is a bidirectional variable (has both input and output)
                        existingVar.isBidirectional = true;
                        existingVar.type = 'bidirectional';
                        existingVar.io = 'bidirectional';
                    }
                }
            });

            // Now build the tree structure from variable names
            allVariables.forEach((variable, key) => {

                // Use the full variable name to build the tree structure
                const fullVariableName = variable.fullName;
                const nameParts = fullVariableName.split('.');
                let current = tree['model'];

                // Create hierarchy based on variable name parts
                for (let i = 0; i < nameParts.length; i++) {
                    const part = nameParts[i];
                    const path = nameParts.slice(0, i + 1).join('.');

                    if (i === nameParts.length - 1) {
                        // This is the final part - add the variable
                        if (!current.variables) {
                            current.variables = [];
                        }
                        // Update the variable's path to match the tree hierarchy
                        variable.path = path;
                        current.variables.push(variable);
                    } else {
                        // This is an intermediate part - create or find the container
                        if (!current.children[part]) {
                            // Check if this path corresponds to an actual subsystem
                            // or if it's a prefix of any actual subsystem
                            const isActualSubsystem = actualSubsystems.has(path) ||
                                Array.from(actualSubsystems).some(subsys => subsys.startsWith(path + '.'));
                            current.children[part] = {
                                name: part,
                                path: path,
                                type: isActualSubsystem ? 'system' : 'container',
                                children: {},
                                variables: []
                            };
                        }
                        current = current.children[part];
                    }
                }
            });

            renderTree(tree);
        }

        function loadVariablesForAllSystems(tree) {
            // First, collect all variables from all systems
            const systemPaths = Object.keys(tree);
            let loadedCount = 0;
            const allVariables = new Map(); // Map to track variables and their owning systems

            systemPaths.forEach(systemPath => {
                const pathForAPI = systemPath === 'model' ? '' : systemPath;

                fetch(`/api/subsystem/${encodeURIComponent(pathForAPI)}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.success && data.nodes_data && tree[systemPath]) {
                            // Collect variables from this system
                            for (const [nodeId, nodeData] of Object.entries(data.nodes_data)) {
                                if (nodeData && nodeData.rel_name && !nodeData.rel_name.startsWith('_auto_ivc.')) {
                                    const variableKey = `${nodeData.rel_name}_${nodeData.io}`;

                                    // Determine the correct system path from the pathname
                                    const correctSystemPath = nodeData.pathname || 'model';

                                    // Only add if we haven't seen this variable before
                                    // or if this is the correct system for this variable
                                    if (!allVariables.has(variableKey) ||
                                        allVariables.get(variableKey).systemPath !== correctSystemPath) {

                                        allVariables.set(variableKey, {
                                            name: nodeData.rel_name,
                                            path: nodeData.pathname + '.' + nodeData.rel_name,
                                            type: nodeData.io === 'i' ? 'input' : 'output',
                                            io: nodeData.io,
                                            systemPath: correctSystemPath
                                        });
                                    }
                                }
                            }
                        }

                        loadedCount++;
                        if (loadedCount === systemPaths.length) {
                            // Now assign variables to their correct systems
                            assignVariablesToSystems(tree, allVariables);
                            renderTree(tree);
                        }
                    })
                    .catch(error => {
                        console.error(`Error loading variables for ${systemPath}:`, error);
                        loadedCount++;
                        if (loadedCount === systemPaths.length) {
                            assignVariablesToSystems(tree, allVariables);
                            renderTree(tree);
                        }
                    });
            });
        }

        function assignVariablesToSystems(tree, allVariables) {
            // Clear existing variables
            Object.keys(tree).forEach(systemPath => {
                tree[systemPath].variables = [];
            });

            // Assign each variable to its owning system
            allVariables.forEach(variable => {
                const systemPath = variable.systemPath;
                if (tree[systemPath]) {
                    tree[systemPath].variables.push({
                        name: variable.name,
                        fullName: variable.fullName,
                        path: variable.path,
                        type: variable.type,
                        io: variable.io
                    });
                }
            });
        }

        function renderTree(tree) {
            treeData = tree;
            treeContainer.innerHTML = '';

            // Render the model root
            const modelNode = tree['model'];
            if (modelNode) {
                const nodeElement = createTreeNode(modelNode, 0);
                treeContainer.appendChild(nodeElement);
            }
        }

        function createTreeNode(node, depth) {
            const nodeDiv = document.createElement('div');
            nodeDiv.className = 'tree-node';
            nodeDiv.setAttribute('data-path', node.path);
            nodeDiv.setAttribute('data-type', node.type);

            const contentDiv = document.createElement('div');
            contentDiv.className = 'tree-node-content';
            contentDiv.onclick = () => selectNode(node);

            // Ensure children and variables exist
            const children = node.children || {};
            const variables = node.variables || [];

            // Toggle button for systems/containers with children
            if ((node.type === 'system' || node.type === 'container') && (Object.keys(children).length > 0 || variables.length > 0)) {
                const toggle = document.createElement('div');
                toggle.className = 'tree-toggle';
                // Show expanded state for first level (depth 0)
                toggle.innerHTML = depth === 0 ? '▼' : '▶';
                toggle.onclick = (e) => {
                    e.stopPropagation();
                    toggleNode(nodeDiv, toggle);
                };
                contentDiv.appendChild(toggle);
            } else {
                const spacer = document.createElement('div');
                spacer.className = 'tree-toggle';
                contentDiv.appendChild(spacer);
            }

            // Icon
            const icon = document.createElement('div');
            icon.className = 'tree-icon';
            if (node.type === 'system') {
                icon.innerHTML = '⚙️';
            } else if (node.type === 'container') {
                icon.innerHTML = '📁';
            } else if (node.type === 'i') {
                icon.innerHTML = '⬇️';
            } else if (node.type === 'o') {
                icon.innerHTML = '⬆️';
            } else if (node.type === 'bidirectional') {
                icon.innerHTML = '⬆️⬇️';
            } else {
                icon.innerHTML = '⬆️⬇️';
            }
            contentDiv.appendChild(icon);

            // Label
            const label = document.createElement('div');
            label.className = 'tree-label';
            label.textContent = node.name;
            contentDiv.appendChild(label);

            nodeDiv.appendChild(contentDiv);

            // Children container
            const childrenDiv = document.createElement('div');
            // Expand the first level (depth 0) of the model tree at startup
            childrenDiv.className = depth === 0 ? 'tree-children' : 'tree-children collapsed';

            // Add child systems (sorted alphabetically)
            Object.values(children)
                .sort((a, b) => a.name.localeCompare(b.name))
                .forEach(child => {
                    childrenDiv.appendChild(createTreeNode(child, depth + 1));
                });

            // Add variables (sorted alphabetically)
            variables
                .sort((a, b) => a.name.localeCompare(b.name))
                .forEach(variable => {
                    const varNode = {
                        name: variable.name,
                        fullName: variable.fullName,
                        path: variable.path,
                        type: variable.io, // Use io as the type for variables
                        io: variable.io
                    };
                    childrenDiv.appendChild(createTreeNode(varNode, depth + 1));
                });

            nodeDiv.appendChild(childrenDiv);
            return nodeDiv;
        }

        function toggleNode(nodeElement, toggle) {
            const children = nodeElement.querySelector('.tree-children');
            if (children.classList.contains('collapsed')) {
                children.classList.remove('collapsed');
                toggle.innerHTML = '▼';
            } else {
                children.classList.add('collapsed');
                toggle.innerHTML = '▶';
            }
        }

        function selectNode(node) {
            // Remove previous selection
            document.querySelectorAll('.tree-node-content.selected').forEach(el => {
                el.classList.remove('selected');
            });

            // Add selection to current node
            event.currentTarget.classList.add('selected');
            selectedNode = node;

            // Load the appropriate graph
            if (node.type === 'system' || node.type === 'container') {
                updateCurrentSubsystem(node.path);
                loadSubsystemGraph(node.path);
            } else if (node.type === 'i' || node.type === 'o' || node.type === 'bidirectional') {
                // For variables, load the variable-specific graph
                updateCurrentSubsystem(node.path);
                loadVariableGraph(node.fullName);
            }
        }

        // Update currentSubsystem when a system is selected
        function updateCurrentSubsystem(systemPath) {
            currentSubsystem = systemPath;
            // No need to reload colors since we're using global graph colors
        }

        function loadVariableGraph(variable) {
            showLoading('Loading variable graph...');

            fetch(`/api/variable/${encodeURIComponent(variable)}`)
                .then(response => {
                    return response.json();
                })
                .then(data => {
                    if (data.success) {
                        displayGraph(data.svg, `Variable: ${variable}`, data.nodes, data.edges);
                    } else {
                        console.error('Variable API error:', data.error);
                        showError(`Error loading variable: ${data.error}`);
                    }
                })
                .catch(error => {
                    console.error('Variable API fetch error:', error);
                    showError(`Error: ${error.message}`);
                });
        }

        function loadSubsystemGraph(subsystem) {
            showLoading('Loading subsystem graph...');

            // Convert 'model' to empty string for top-level system
            const subsystemPath = subsystem === 'model' ? '' : subsystem;

            fetch(`/api/subsystem/${encodeURIComponent(subsystemPath)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayGraph(data.svg, `System: ${subsystem}`, data.nodes, data.edges);
                    } else {
                        showError(`Error loading subsystem: ${data.error}`);
                    }
                })
                .catch(error => {
                    console.error('System load error:', error);
                    showError(`Error: ${error.message}`);
                });
        }

        function displayGraph(svgContent, title, nodes, edges) {
            // Check if content is SVG or HTML
            if (svgContent.trim().startsWith('<svg')) {
                graphContent.innerHTML = `
                    <div class="graph-svg" style="flex: 1; display: flex; align-items: center; justify-content: center;">${svgContent}</div>
                `;
            } else {
                graphContent.innerHTML = `
                    <div class="graph-text" style="flex: 1;">${svgContent}</div>
                `;
            }

            // Determine if this is a variable or subsystem view
            let label, name;
            if (title.startsWith('Variable:')) {
                label = 'Variable:';
                name = title.replace('Variable: ', '');
            } else if (title.startsWith('System:')) {
                label = 'System:';
                name = title.replace('System: ', '');
            } else {
                label = 'View:';
                name = title;
            }

            graphInfo.innerHTML = `
                <div><strong>${label}</strong> ${name}</div>
                <div><strong>Nodes:</strong> ${nodes}</div>
            `;
        }

        function showLoading(message) {
            graphContent.innerHTML = `<div class="loading">${message}</div>`;
        }

        function showError(message) {
            graphContent.innerHTML = `<div class="loading" style="color: red;">${message}</div>`;
        }

        function loadHelpColors() {
            // Use global graph colors if available
            if (globalGraphColors) {
                // Cache the colors from global graph colors
                if (globalGraphColors.input) {
                    cachedColors.input = getColorHex(globalGraphColors.input);
                }
                if (globalGraphColors.output) {
                    cachedColors.output = getColorHex(globalGraphColors.output);
                }
                if (globalGraphColors.highlight) {
                    cachedColors.highlight = getColorHex(globalGraphColors.highlight);
                } else {
                    console.error('No highlight color in globalGraphColors');
                }
                if (globalGraphColors.boundary) {
                    cachedColors.boundary = getColorHex(globalGraphColors.boundary);
                } else {
                    console.error('No boundary color in globalGraphColors');
                }
                colorsLoaded = true; // Mark as loaded
            } else {
                // Fallback to old method if global colors not available
                const subsystemPath = currentSubsystem === 'model' ? '' : currentSubsystem;
                colorsLoaded = false; // Reset flag

                fetch(`/api/subsystem/${encodeURIComponent(subsystemPath)}`)
                    .then(response => response.json())
                    .then(subsystemData => {
                        if (subsystemData.success && subsystemData.help_colors) {
                            // Cache the colors
                            if (subsystemData.help_colors.i) {
                                cachedColors.input = getColorHex(subsystemData.help_colors.i);
                            }
                            if (subsystemData.help_colors.o) {
                                cachedColors.output = getColorHex(subsystemData.help_colors.o);
                            }
                        }
                        colorsLoaded = true; // Mark as loaded
                    })
                    .catch(error => {
                        // Keep default colors if fetch fails
                        colorsLoaded = true; // Still mark as loaded even if failed
                    });
            }
        }

        function showHelp() {
            // Wait for colors to load if they haven't loaded yet
            if (!colorsLoaded) {
                // Wait a bit and try again
                setTimeout(showHelp, 50);
                return;
            }

            // Use cached colors - no delay!
            const inputColorEl = document.getElementById('input-color-swatch');
            const outputColorEl = document.getElementById('output-color-swatch');
            const highlightColorEl = document.getElementById('highlight-color-swatch');
            const boundaryColorEl = document.getElementById('boundary-color-swatch');

            if (inputColorEl) {
                inputColorEl.style.backgroundColor = cachedColors.input;
            }
            if (outputColorEl) {
                outputColorEl.style.backgroundColor = cachedColors.output;
            }
            if (highlightColorEl && cachedColors.highlight) {
                highlightColorEl.style.backgroundColor = cachedColors.highlight;
            } else {
                console.error('Highlight color element or color not available:', highlightColorEl,
                              cachedColors.highlight);
            }
            if (boundaryColorEl && cachedColors.boundary) {
                boundaryColorEl.style.backgroundColor = cachedColors.boundary;
            } else {
                console.error('Boundary color element or color not available:', boundaryColorEl,
                              cachedColors.boundary);
            }

            document.getElementById('help-modal').style.display = 'flex';
        }


        function hideHelp() {
            document.getElementById('help-modal').style.display = 'none';
        }

        // Close modal when clicking outside of it
        window.onclick = function(event) {
            const modal = document.getElementById('help-modal');
            if (event.target === modal) {
                hideHelp();
            }
        }

    </script>
</body>
</html>
        '''



if __name__ == "__main__":
    # Example usage
    from example_conn_graph_ui import create_complex_engineering_model

    # Create model
    prob = create_complex_engineering_model()
    prob.setup()

    # Start web UI
    prob.model._get_all_conn_graph().serve(port=8001)

    # serve_simple_conn_graph_ui(conn_graph, port=8001)
