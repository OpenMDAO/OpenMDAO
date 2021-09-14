import openmdao.api as om
import json

from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import (Input, Output, State)
# import dash_daq as daq
import plotly.graph_objs as go
from dash import Dash
import os
import signal


class DashDB():

    def __init__(self):
        self.running = False
        self.app = JupyterDash(__name__)

        self.app.layout = html.Div([
            dcc.Graph(id='demo-live'),
            ## for every 0.5 seconds the layout updates
            dcc.Interval(id='output-update', interval=500),
            html.Div(id='output-container-button',
                    children='Press to shutdown'),
            html.Button('Shutdown Server', id='shutdown'),
        ])

        @self.app.callback(
            Output('output-container-button', 'children'),
            [Input('shutdown', 'n_clicks')],
        )
        def update_output(on):
            if self.running:
                print("Status: Shutting Down")
                try:
                    # iterating through each instance of the process
                    for line in os.popen("ps ax | grep openmdao/visualization/dash_server.py | grep -v grep"):
                        fields = line.split()

                        # extracting Process ID from the output
                        pid = fields[0]

                        # terminating process
                        os.kill(int(pid), signal.SIGKILL)
                    print("Process Successfully terminated")

                except:
                    print("Error Encountered while running script")

            self.running = True

        @self.app.callback(
            Output(component_id='demo-live', component_property='figure'),
            Input(component_id='output-update', component_property='n_intervals'),
        )
        def update_graph(_):
            opt_data = self.parse()
            if opt_data:
                x = opt_data["nMajor"]
                y = opt_data["feasibility"]
            else:
                x = [0.0]
                y = [0.0]


            fig = go.Figure()
            x_color = 'blue'
            fig.add_trace(go.Scatter(x=x, y=x,
                        mode='lines',
                        name='x',
                        marker_color=x_color
                        ))

            fig.update_layout(
                    title="Optimization Results",
                    title_x=0.5,
                    title_xanchor='center',
                    xaxis_title="Iterations",
                    yaxis_title="Feasibility",
                    legend_title="Vars",
            )

            return fig

        self.app.run_server(mode='inline', debug=True, port=8100)
        self.running = True

    def parse(self):
        opt_data = None
        if os.path.exists("cases.sql"):
            cr = om.CaseReader("cases.sql")
            cases = cr.get_cases()

            opt_data = {}
            for case in cases:
                if hasattr(case, 'opt_progress') and "{}" not in case.opt_progress:
                    data = json.loads(case.opt_progress)
                    for key, val in data.items():
                        if key not in opt_data:
                            opt_data[key] = [val]
                        else:
                            opt_data[key].append(val)

        return opt_data


if __name__ == '__main__':
    server = DashDB()











# def parse():
#     opt_data = None
#     if os.path.exists("cases.sql"):
#         cr = om.CaseReader("cases.sql")
#         cases = cr.get_cases()

#         opt_data = {}
#         for case in cases:
#             if hasattr(case, 'opt_progress') and "{}" not in case.opt_progress:
#                 data = json.loads(case.opt_progress)
#                 for key, val in data.items():
#                     if key not in opt_data:
#                         opt_data[key] = [val]
#                     else:
#                         opt_data[key].append(val)

#     return opt_data



# @app.callback(
#     Output(component_id='demo-live', component_property='figure'),
#     Input(component_id='output-update', component_property='n_intervals'),
# )
# def update_graph(_):
#     opt_data = parse()
#     if opt_data:
#         x = opt_data["nMajor"]
#         y = opt_data["feasibility"]
#     else:
#         x = [0.0]
#         y = [0.0]


#     fig = go.Figure()
#     x_color = 'blue'
#     fig.add_trace(go.Scatter(x=x, y=x,
#                 mode='lines',
#                 name='x',
#                 marker_color=x_color
#                 ))

#     fig.update_layout(
#             title="Optimization Results",
#             title_x=0.5,
#             title_xanchor='center',
#             xaxis_title="Iterations",
#             yaxis_title="Feasibility",
#             legend_title="Vars",
#     )

#     return fig

# app.run_server(debug=True, port=8100)



