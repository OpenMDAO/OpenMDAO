import json
import os

import openmdao.api as om

from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from threading import Thread
import dymos as dm

class OptViewer():

    def __init__(self, prob):
        print("Bokeh App Entered")
        self.prob = prob
        thread = Thread(target = self.run_problem)
        thread.start()

    def start_visualization(self):
        io_loop = IOLoop.current()
        server = Server(applications = {'/optimizer_progress': Application(FunctionHandler(self.make_document))}, io_loop = io_loop, port = 5003)
        server.start()
        server.show('/optimizer_progress')
        io_loop.start()

    def run_problem(self):
        dm.run_problem(self.prob)

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

    def make_document(self, doc):
        self.source = ColumnDataSource(dict(
            iterations=[], feasibility=[]
        ))
        p = figure(title="Iterations vs Feasibility", x_axis_label='Iterations', y_axis_label='Feasibility')
        p.line(x="iterations", y="feasibility",line_width=2, source=self.source)

        doc.add_root(p)
        doc.add_periodic_callback(self.update, 1000)

    def update(self):
        opt_data = self.parse()

        if opt_data:
            new_data = dict(
                iterations=opt_data["nMajor"],
                feasibility=opt_data["feasibility"],
            )
        else:
            new_data = dict(
                iterations=[],
                feasibility=[],
            )

        self.source.stream(new_data, 20)
