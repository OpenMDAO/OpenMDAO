#!/usr/bin/env python3
"""
Simple AllConnGraph Web UI with better Graphviz integration.

This creates a web interface that:
1. Uses Graphviz for proper graph layouts
2. Serves focused views of connection graphs
3. Provides search and navigation
4. Avoids complex D3.js force simulations
"""

import json
from http.server import SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote
import networkx as nx


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
        info = {
            'nodes': len(self.conn_graph.nodes()),
            'edges': len(self.conn_graph.edges()),
            'subsystems': self.get_subsystems()
        }
        self.send_json_response(info)

    def serve_subsystem_graph(self, subsystem):
        """Serve graph for a specific subsystem."""
        try:
            print(f"Serving subsystem graph for: '{subsystem}'")
            subgraph = self.conn_graph.get_drawable_graph(subsystem)
            print(f"Subgraph has {len(subgraph.nodes())} nodes and {len(subgraph.edges())} edges")

            pydot_graph = nx.drawing.nx_pydot.to_pydot(subgraph)
            try:
                graphviz_svg = pydot_graph.create_svg().decode('utf-8')
            except Exception:
                graphviz_svg = self.create_text_graph(subgraph, f"Subsystem: {subsystem}")

            # Convert nodes to a format the frontend can use
            nodes_data = {}
            for node_id, node_data in subgraph.nodes(data=True):
                print(f"Raw node {node_id}: {node_data}")

                # Extract the actual variable information from the node metadata
                # The node_data contains the original rel_name and pathname
                rel_name = node_data.get('rel_name', '')
                pathname = node_data.get('pathname', '')
                io_type = node_data.get('io', '')

                # If io_type is not in node_data, extract from node_id tuple
                if not io_type and isinstance(node_id, tuple) and len(node_id) == 2:
                    io_type = node_id[0]  # 'i' or 'o'

                nodes_data[str(node_id)] = {
                    'rel_name': rel_name,
                    'pathname': pathname,
                    'io': io_type
                }
                print(f"Processed node {node_id}: rel_name='{rel_name}', pathname='{pathname}', io='{io_type}'")

            response = {
                'success': True,
                'subsystem': subsystem,
                'nodes': len(subgraph.nodes()),
                'edges': len(subgraph.edges()),
                'nodes_data': nodes_data,
                'svg': graphviz_svg
            }
            print(f"Returning {len(nodes_data)} nodes_data entries")
        except Exception as e:
            print(f"Error in serve_subsystem_graph: {e}")
            # Provide more helpful error message
            error_msg = str(e)
            if "not found" in error_msg.lower():
                error_msg = f"Subsystem '{subsystem}' not found. Try searching for a different subsystem name."
            response = {'success': False, 'error': error_msg}

        self.send_json_response(response)

    def serve_variable_graph(self, variable):
        """Serve graph focused on a specific variable."""
        try:
            # First try the variable name as-is
            try:
                pydot_graph = self.conn_graph.get_pydot_graph(varname=variable)
            except Exception:
                # If that fails, try to find the full pathname for this variable
                full_variable = self.find_full_variable_path(variable)
                if full_variable:
                    pydot_graph = self.conn_graph.get_pydot_graph(varname=full_variable)
                else:
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
                    graphviz_svg = f"<div style='padding: 20px; color: red;'>Error generating graph for variable: {variable}</div>"

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
                error_msg = f"Variable '{variable}' not found. Try searching for a different variable name."
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
        for node_id, node_data in self.conn_graph.nodes(data=True):
            pathname = node_data.get('pathname', '')
            if pathname:
                subsystems.add(pathname)

        print(f"Found subsystems: {sorted(subsystems)}")
        return sorted(subsystems)

    def create_text_graph(self, subgraph, title):
        """Create a simple text-based graph representation when Graphviz is not available."""
        html = '<div style="font-family: monospace; padding: 20px;">'
        html += f'<h3>{title}</h3>'
        html += f'<p><strong>Nodes:</strong> {len(subgraph.nodes())} | <strong>Edges:</strong> {len(subgraph.edges())}</p>'

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
            text-align: center;
            flex-shrink: 0;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
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
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Connection Graph Explorer</h1>
        </div>

        <div class="controls">
            <div style="display: flex; gap: 20px; align-items: center;">
                <div style="flex: 1;">
                    <label for="subsystem-select" style="display: block; margin-bottom: 5px; font-weight: bold;">System:</label>
                    <select id="subsystem-select" class="dropdown" onchange="onSubsystemChange()">
                        <option value="model">Model (Top Level)</option>
                    </select>
                </div>
                <div style="flex: 1;">
                    <label for="variable-select" style="display: block; margin-bottom: 5px; font-weight: bold;">Variable:</label>
                    <select id="variable-select" class="dropdown" onchange="onVariableChange()">
                        <option value="">Select a variable...</option>
                    </select>
                </div>
                <div style="flex: 0;">
                    <button onclick="showCurrentSubsystem()" style="padding: 10px 20px; background: #2c3e50; color: white; border: none; border-radius: 4px; cursor: pointer;">Show Subsystem</button>
                </div>
            </div>
        </div>

        <div class="graph-container">
            <div id="graph-content">
                <div class="loading">
                    <h3>🔍 Select a subsystem and variable to explore</h3>
                    <p>Use the dropdowns above to:</p>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Choose a subsystem to see its connection graph</li>
                        <li>Select a variable to focus on its connections</li>
                        <li>Navigate through your OpenMDAO model hierarchy</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="info" id="graph-info">
            <strong>Ready to explore!</strong> Use the dropdowns above to navigate your OpenMDAO model.
        </div>
    </div>

    <script>
        let currentGraph = null;
        let currentSubsystem = 'model';

        // UI elements
        const subsystemSelect = document.getElementById('subsystem-select');
        const variableSelect = document.getElementById('variable-select');
        const graphContent = document.getElementById('graph-content');
        const graphInfo = document.getElementById('graph-info');

        // Initialize the interface
        document.addEventListener('DOMContentLoaded', function() {
            loadSubsystems();
            loadVariablesForSubsystem('model');
        });

        function loadSubsystems() {
            fetch('/api/graph_info')
                .then(response => response.json())
                .then(data => {
                    // Clear existing options except the first one
                    subsystemSelect.innerHTML = '<option value="model">Model (Top Level)</option>';

                    // Add subsystem options, filtering out 'model' if it exists
                    const filteredSubsystems = data.subsystems.filter(s => s !== 'model');
                    filteredSubsystems.forEach(subsystem => {
                        const option = document.createElement('option');
                        option.value = subsystem;
                        option.textContent = subsystem;
                        subsystemSelect.appendChild(option);
                    });
                })
                .catch(error => {
                    console.error('Error loading subsystems:', error);
                });
        }

        function loadVariablesForSubsystem(subsystem) {
            const subsystemPath = subsystem === 'model' ? '' : subsystem;
            console.log('Loading variables for subsystem:', subsystem, 'path:', subsystemPath);

            fetch(`/api/subsystem/${encodeURIComponent(subsystemPath)}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Full API response:', data);
                    if (data.success) {
                        // Clear existing options
                        variableSelect.innerHTML = '<option value="">Select a variable...</option>';

                        // Get all variables in this subsystem
                        const variables = new Set();

                        console.log('nodes_data:', data.nodes_data);
                        console.log('nodes_data type:', typeof data.nodes_data);
                        console.log('nodes_data keys:', data.nodes_data ? Object.keys(data.nodes_data) : 'none');

                        if (data.nodes_data) {
                            for (const [nodeId, nodeData] of Object.entries(data.nodes_data)) {
                                console.log('Processing node:', nodeId, nodeData);
                                if (nodeData && nodeData.rel_name) {
                                    // Create the combined name: pathname + '.' + rel_name (or just rel_name if no pathname)
                                    let combinedName = nodeData.rel_name;
                                    if (nodeData.pathname && nodeData.pathname !== '') {
                                        combinedName = nodeData.pathname + '.' + nodeData.rel_name;
                                    }

                                    // Filter out internal OpenMDAO variables
                                    if (!combinedName.startsWith('_auto_ivc.')) {
                                        console.log('Adding variable:', combinedName);
                                        variables.add(combinedName);
                                    } else {
                                        console.log('Filtering out internal variable:', combinedName);
                                    }
                                }
                            }
                        }

                        console.log('Found variables:', Array.from(variables));

                        // Add variable options
                        Array.from(variables).sort().forEach(variable => {
                            const option = document.createElement('option');
                            option.value = variable;
                            option.textContent = variable;
                            variableSelect.appendChild(option);
                        });

                        console.log(`Loaded ${variables.size} variables for subsystem: ${subsystem}`);
                    } else {
                        console.error('Failed to load subsystem:', data.error);
                    }
                })
                .catch(error => {
                    console.error('Error loading variables:', error);
                });
        }

        function onSubsystemChange() {
            const selectedSubsystem = subsystemSelect.value;
            currentSubsystem = selectedSubsystem;
            console.log('Subsystem changed to:', selectedSubsystem);
            loadVariablesForSubsystem(selectedSubsystem);
            showCurrentSubsystem();
        }

        function onVariableChange() {
            const selectedVariable = variableSelect.value;
            if (selectedVariable) {
                loadVariableGraph(selectedVariable);
            }
        }

        function showCurrentSubsystem() {
            const subsystemPath = currentSubsystem === 'model' ? '' : currentSubsystem;
            console.log('Showing subsystem:', subsystemPath);
            loadSubsystemGraph(subsystemPath);
        }

        function loadVariableGraph(variable) {
            showLoading('Loading variable graph...');

            fetch(`/api/variable/${encodeURIComponent(variable)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayGraph(data.svg, `Variable: ${variable}`, data.nodes, data.edges);
                    } else {
                        showError(`Error loading variable: ${data.error}`);
                    }
                })
                .catch(error => {
                    showError(`Error: ${error.message}`);
                });
        }

        function loadSubsystemGraph(subsystem) {
            showLoading('Loading subsystem graph...');
            console.log('Loading subsystem graph for:', subsystem);

            fetch(`/api/subsystem/${encodeURIComponent(subsystem)}`)
                .then(response => response.json())
                .then(data => {
                    console.log('Subsystem response:', data);
                    if (data.success) {
                        displayGraph(data.svg, `Subsystem: ${subsystem}`, data.nodes, data.edges);
                    } else {
                        showError(`Error loading subsystem: ${data.error}`);
                    }
                })
                .catch(error => {
                    console.error('Subsystem load error:', error);
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

            graphInfo.innerHTML = `
                <strong>${title}</strong> | <strong>Nodes:</strong> ${nodes} | <strong>Edges:</strong> ${edges}
            `;
        }

        function showLoading(message) {
            graphContent.innerHTML = `<div class="loading">${message}</div>`;
        }

        function showError(message) {
            graphContent.innerHTML = `<div class="loading" style="color: red;">${message}</div>`;
        }



        // Hide search results when clicking outside
        document.addEventListener('click', function(e) {
            if (!searchBox.contains(e.target) && !searchResults.contains(e.target)) {
                searchResults.style.display = 'none';
            }
        });
    </script>
</body>
</html>
        '''


# def create_simple_conn_graph_ui(conn_graph):
#     """Create a simple web UI for the connection graph."""
#     return conn_graph


# def serve_simple_conn_graph_ui(conn_graph, port=8001, open_browser=True):
#     """Serve the simple connection graph web UI."""
#     def handler(*args, **kwargs):
#         return ConnGraphHandler(conn_graph, *args, **kwargs)

#     print(f"🌐 Starting Simple AllConnGraph Web UI on port {port}")
#     print(f"📱 Open your browser to: http://localhost:{port}")

#     if open_browser:
#         def open_browser():
#             time.sleep(1)
#             webbrowser.open(f'http://localhost:{port}')

#         threading.Thread(target=open_browser, daemon=True).start()

#     try:
#         with HTTPServer(("", port), handler) as httpd:
#             print(f"✅ Server running on http://localhost:{port}")
#             print("Press Ctrl+C to stop")
#             httpd.serve_forever()
#     except KeyboardInterrupt:
#         print("\n🛑 Server stopped")


if __name__ == "__main__":
    # Example usage
    from example_conn_graph_ui import create_complex_engineering_model

    # Create model
    prob = create_complex_engineering_model()
    prob.setup()

    # Start web UI
    prob.model._get_all_conn_graph().serve(port=8001)

    # serve_simple_conn_graph_ui(conn_graph, port=8001)
