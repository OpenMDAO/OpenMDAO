from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import (Input, Output)
import plotly.graph_objs as go

import pandas as pd
import time
import random
from csv import writer
import os



class OptimizationProgressViz(object):

    def __init__(self, data=None):

        self.data = data
        self.create_layout()

    def create_layout(self):
        app = JupyterDash(__name__)

        app.layout = html.Div([
            dcc.Graph(id='demo-live'),
            ## for every 2 secs the layout updates
            dcc.Interval(id='output-update', interval=2*1000)
        ])

    def generate_streaming_data(self):
        # rand_vals = [random.randint(20, 50), random.randint(20, 50)]
        data = self.data

        file_name = 'simple_demo.csv'
        if os.path.exists(path=file_name):
            pass
        else:
            with open(file_name, 'w') as wo:
                csv_writer = writer(wo)
                csv_writer.writerow(['col1', 'col2'])

        with open(file_name, 'a+', newline='\n') as wo:
            csv_writer = writer(wo)
            csv_writer.writerow(rand_vals)
        return None

# app = OptimizationProgressViz()

# @app.callback(
#     Output(component_id='demo-live', component_property='figure'),
#     [Input(component_id='output-update', component_property='n_intervals')]
# )
# def get_live_updates(n_intervals):
#     # app.generate_streaming_data()
#     time.sleep(1)
#     df = pd.read_csv('simple_demo.csv')
#     x = sorted(df['col1'].to_list())
#     y = sorted(df['col2'].to_list())
#     data = go.Scatter(
#         x=x, y=y, mode='lines+markers'
#     )

#     layout = go.Layout(
#         autosize=False,
#         width=780,
#         height=500,
#         margin=dict(l=20, r=20, t=20, b=20),
#         paper_bgcolor="LightSteelBlue",
#         plot_bgcolor='#ebf3fc'
#     )
#     fig = {'data' : [data], 'layout' : layout}
#     return fig

# app.run_server(mode='inline')