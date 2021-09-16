import json
import os

import openmdao.api as om

from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from threading import Thread

class OptViewer(object):

    def __init__(self, port=5003):
        """
        Initialize threading.

        port : int
            What port to host Bokeh server on.
        """
        self.port = port
        thread = Thread(target = self._start_visualization)
        thread.start()

    def _start_visualization(self):
        """
        Start Bokeh server.
        """
        self.io_loop = IOLoop()
        server = Server(applications = {'/optimizer_progress': Application(FunctionHandler(self._make_document))}, io_loop = self.io_loop, port = self.port)
        server.start()
        server.show('/optimizer_progress')
        self.io_loop.start()

    def _parse(self, case_file="cases.sql"):
        """
        Parse the case recorder.
        """
        opt_data = None
        if os.path.exists(case_file):
            cr = om.CaseReader(case_file)
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

    def _make_document(self, doc):
        """
        Setup the Bokeh plot layout and set callback to update with new values.
        """
        self.source = ColumnDataSource(dict(
            iterations=[], feasibility=[]
        ))

        p = figure(title="Iterations vs Feasibility", x_axis_label='Iterations', y_axis_label='Feasibility')
        p.line(x="iterations", y="feasibility", line_width=2, source=self.source)

        doc.add_root(p)
        doc.add_periodic_callback(self._update, 1000)

    def _update(self):
        """
        Parse and update the source data if new data is present.
        """
        opt_data = self._parse()

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

        if opt_data and len(self.source.data['iterations']) != len(new_data['iterations']):
            self.source.data = new_data