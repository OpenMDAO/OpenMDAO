import openmdao.api as om
import json

from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import (Input, Output)
import plotly.graph_objs as go
from dash import Dash
from flask import request
import os
import time

app = Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='demo-live'),
    ## for every 0.5 seconds the layout updates
    dcc.Interval(id='output-update', interval=500),
    html.Button('Submit', id='submit-val', n_clicks=0),
])

def parse():
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

@app.callback(
    Output(component_id='demo-live', component_property='figure'),
    Input(component_id='output-update', component_property='n_intervals'),
)
def update_graph(_):
    opt_data = parse()
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

app.run_server(debug=False, port=8100)