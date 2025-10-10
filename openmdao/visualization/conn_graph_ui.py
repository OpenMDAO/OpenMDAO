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
from urllib.parse import urlparse, parse_qs
import pydot


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
            subsystem = parsed_path.path.replace('/api/subsystem/', '')
            self.serve_subsystem_graph(subsystem)
        elif parsed_path.path.startswith('/api/variable/'):
            variable = parsed_path.path.replace('/api/variable/', '')
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
            subgraph = self.get_subsystem_graph(subsystem)
            graphviz_svg = self.generate_graphviz_svg(subgraph, f"Subsystem: {subsystem}")

            response = {
                'success': True,
                'subsystem': subsystem,
                'nodes': len(subgraph.nodes()),
                'edges': len(subgraph.edges()),
                'svg': graphviz_svg
            }
        except Exception as e:
            response = {'success': False, 'error': str(e)}

        self.send_json_response(response)

    def serve_variable_graph(self, variable):
        """Serve graph focused on a specific variable."""
        try:
            subgraph = self.get_variable_graph(variable)
            graphviz_svg = self.generate_graphviz_svg(subgraph, f"Variable: {variable}")

            response = {
                'success': True,
                'variable': variable,
                'nodes': len(subgraph.nodes()),
                'edges': len(subgraph.edges()),
                'svg': graphviz_svg
            }
        except Exception as e:
            response = {'success': False, 'error': str(e)}

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
            name = node_data.get('name') or node_data.get('rel_name', '')
            if query in name.lower():
                results.append({
                    'type': 'variable',
                    'name': name,
                    'path': node_data.get('pathname', ''),
                    'io_type': node_data.get('io', '')
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
        subsystems = set()
        for node_id, node_data in self.conn_graph.nodes(data=True):
            pathname = node_data.get('pathname', '')
            if pathname:
                parts = pathname.split('.')
                if len(parts) > 1:
                    subsystem = '.'.join(parts[:-1])
                    subsystems.add(subsystem)
        return sorted(list(subsystems))

    def get_subsystem_graph(self, subsystem_path):
        """Get subgraph for a specific subsystem."""
        subsystem_nodes = []
        for node_id, node_data in self.conn_graph.nodes(data=True):
            pathname = node_data.get('pathname', '')
            if pathname.startswith(subsystem_path + '.') or pathname == subsystem_path:
                subsystem_nodes.append(node_id)

        return self.conn_graph.subgraph(subsystem_nodes)

    def get_variable_graph(self, variable_name):
        """Get subgraph focused on a specific variable."""
        target_node = None
        for node_id, node_data in self.conn_graph.nodes(data=True):
            name = node_data.get('name') or node_data.get('rel_name', '')
            if name == variable_name:
                target_node = node_id
                break

        if not target_node:
            raise ValueError(f"Variable '{variable_name}' not found")

        # Get connected nodes (up to 2 degrees of separation)
        connected_nodes = {target_node}

        # First degree connections
        for edge in self.conn_graph.edges():
            if edge[0] == target_node or edge[1] == target_node:
                connected_nodes.add(edge[0])
                connected_nodes.add(edge[1])

        # Second degree connections
        for edge in self.conn_graph.edges():
            if edge[0] in connected_nodes or edge[1] in connected_nodes:
                connected_nodes.add(edge[0])
                connected_nodes.add(edge[1])

        return self.conn_graph.subgraph(list(connected_nodes))

    def generate_graphviz_svg(self, subgraph, title):
        """Generate SVG using pydot."""
        try:
            # Create a pydot graph
            graph = pydot.Dot(graph_type='digraph')
            graph.set_rankdir('TB')
            graph.set_node_defaults(shape='box', style='filled', fontname='Arial', fontsize='10')
            graph.set_edge_defaults(color='gray', fontname='Arial', fontsize='8')

            # Add nodes
            for node_id, node_data in subgraph.nodes(data=True):
                name = node_data.get('name') or node_data.get('rel_name', str(node_id))
                io_type = node_data.get('io', '')
                color = '#ffeb3b' if io_type == 'i' else '#4caf50'

                # Create pydot node
                node = pydot.Node(str(node_id), label=name, fillcolor=color)
                graph.add_node(node)

            # Add edges
            for edge in subgraph.edges():
                edge_obj = pydot.Edge(str(edge[0]), str(edge[1]))
                graph.add_edge(edge_obj)

            # Generate SVG
            svg_content = graph.create_svg().decode('utf-8')
            return svg_content

        except Exception:
            # Fallback to text representation
            return self.create_text_graph(subgraph, title)

    def create_text_graph(self, subgraph, title):
        """Create a simple text-based graph representation when Graphviz is not available."""
        html = '<div style="font-family: monospace; padding: 20px;">'
        html += f'<h3>{title}</h3>'
        html += f'<p><strong>Nodes:</strong> {len(subgraph.nodes())} | <strong>Edges:</strong> {len(subgraph.edges())}</p>'

        # Show nodes
        html += '<h4>Variables:</h4><ul>'
        for node_id, node_data in subgraph.nodes(data=True):
            name = node_data.get('name') or node_data.get('rel_name', str(node_id))
            io_type = node_data.get('io', '')
            io_label = 'Input' if io_type == 'i' else 'Output'
            html += f'<li><strong>{name}</strong> ({io_label})</li>'
        html += '</ul>'

        # Show connections
        html += '<h4>Connections:</h4><ul>'
        for edge in subgraph.edges():
            source_data = subgraph.nodes[edge[0]]
            target_data = subgraph.nodes[edge[1]]
            source_name = source_data.get('name') or source_data.get('rel_name', str(edge[0]))
            target_name = target_data.get('name') or target_data.get('rel_name', str(edge[1]))
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
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .controls {
            padding: 20px;
            border-bottom: 1px solid #eee;
            background: #f8f9fa;
        }
        .search-box {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
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
            padding: 20px;
            text-align: center;
        }
        .graph-svg {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .graph-text {
            text-align: left;
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 20px;
            margin: 10px 0;
        }
        .info {
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #eee;
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
            <h1>🔗 AllConnGraph Explorer</h1>
            <p>Explore OpenMDAO connection graphs with Graphviz layouts</p>
        </div>

        <div class="controls">
            <input type="text" id="search-box" class="search-box" placeholder="Search for variables or subsystems...">
            <div id="search-results" class="search-results"></div>
        </div>

        <div class="graph-container">
            <div id="graph-content">
                <div class="loading">
                    <h3>🔍 Search for a variable or subsystem to get started</h3>
                    <p>Try searching for:</p>
                    <ul style="text-align: left; display: inline-block;">
                        <li>Variable names (e.g., 'wing_span', 'engine_power')</li>
                        <li>Subsystem names (e.g., 'aerodynamics', 'propulsion')</li>
                        <li>Partial matches work too!</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="info" id="graph-info">
            <strong>Ready to explore!</strong> Use the search box above to find variables or subsystems.
        </div>
    </div>

    <script>
        let currentGraph = null;

        // Search functionality
        const searchBox = document.getElementById('search-box');
        const searchResults = document.getElementById('search-results');
        const graphContent = document.getElementById('graph-content');
        const graphInfo = document.getElementById('graph-info');

        searchBox.addEventListener('input', function() {
            const query = this.value.trim();
            if (query.length < 2) {
                searchResults.style.display = 'none';
                return;
            }

            fetch(`/api/search?q=${encodeURIComponent(query)}`)
                .then(response => response.json())
                .then(data => {
                    displaySearchResults(data.results);
                })
                .catch(error => {
                    console.error('Search error:', error);
                });
        });

        function displaySearchResults(results) {
            if (results.length === 0) {
                searchResults.innerHTML = '<div class="search-result">No results found</div>';
            } else {
                searchResults.innerHTML = results.map(result =>
                    `<div class="search-result" onclick="selectResult('${result.type}', '${result.name}')">
                        <strong>${result.name}</strong> (${result.type})
                        ${result.path ? `<br><small>${result.path}</small>` : ''}
                    </div>`
                ).join('');
            }
            searchResults.style.display = 'block';
        }

        function selectResult(type, name) {
            searchBox.value = name;
            searchResults.style.display = 'none';

            if (type === 'variable') {
                loadVariableGraph(name);
            } else if (type === 'subsystem') {
                loadSubsystemGraph(name);
            }
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

            fetch(`/api/subsystem/${encodeURIComponent(subsystem)}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        displayGraph(data.svg, `Subsystem: ${subsystem}`, data.nodes, data.edges);
                    } else {
                        showError(`Error loading subsystem: ${data.error}`);
                    }
                })
                .catch(error => {
                    showError(`Error: ${error.message}`);
                });
        }

        function displayGraph(svgContent, title, nodes, edges) {
            // Check if content is SVG or HTML
            if (svgContent.trim().startsWith('<svg')) {
                graphContent.innerHTML = `
                    <h3>${title}</h3>
                    <div class="graph-svg">${svgContent}</div>
                `;
            } else {
                graphContent.innerHTML = `
                    <h3>${title}</h3>
                    <div class="graph-text">${svgContent}</div>
                `;
            }

            graphInfo.innerHTML = `
                <strong>${title}</strong><br>
                <strong>Nodes:</strong> ${nodes} | <strong>Edges:</strong> ${edges}
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
