import json

import openmdao.api as om

from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import row, column
from bokeh.models import Select
from threading import Thread

class OptViewer(object):

    def __init__(self, data, update_freq=1000, port=5003):
        """
        Initialize threading.

        port : int
            What port to host Bokeh server on.
        data : CaseRecorder or str
            A path to the recorder file or CaseRecorder.
            Currently only sqlite database files recorded via SqliteRecorder are supported.
        """
        self.data = data
        self.port = port
        self.update_freq = update_freq
        self.live_updating = None
        self.init_call_made = False
        self.opt_complete = False

        thread = Thread(target = self._start_visualization)
        thread.start()

    def _start_visualization(self):
        """
        Start Bokeh server.
        """
        self.io_loop = IOLoop()
        server = Server(applications = {
            '/optimizer_progress': Application(FunctionHandler(self._make_document))},
            io_loop = self.io_loop,
            port = self.port)
        server.start()
        server.show('/optimizer_progress')
        self.io_loop.start()

    def _parse_cases_for_variables(self):
        if not self.opt_complete:
            self.case_select.options = list(self.cr.list_cases(out_stream=None))
            self.variable_select.options = self.cr.list_source_vars('driver', out_stream=None)['outputs']

        if not self.init_call_made:
            self.case_select.value = self.case_select.options[0]
            self.variable_select.value = self.variable_select.options[0]

    def _parse_case(self):
        if isinstance(self.data, str):
            self.cr = om.CaseReader(self.data)
        if isinstance(self.data, om.SqliteRecorder):
            self.cr = self.data

    def _parse(self):
        """
        Parse the case recorder.
        """
        opt_data = None

        self._parse_case()
        cases = self.cr.get_cases()

        opt_data = {}
        for case in cases:
            if hasattr(case, 'opt_progress') and "{}" not in case.opt_progress:
                data = json.loads(case.opt_progress)
                for key, val in data.items():
                    if key not in opt_data:
                        opt_data[key] = [val]
                    else:
                        opt_data[key].append(val)

        if opt_data and opt_data['sol_status'][-1] != -1 and self.live_updating is not None:
            self.live_updating = self.doc.remove_periodic_callback(self.live_updating)
            self._parse_cases_for_variables()
            # self.opt_complete = True

        return opt_data

    def _make_document(self, doc):
        """
        Setup the Bokeh plot layout and set callback to update with new values.
        """
        self.source = ColumnDataSource(dict(
            x_vals=[], y_vals=[]
        ))
        self.var_data = ColumnDataSource(dict(
            x_vals=[], y_vals=[]
        ))

        # Convergence Plot and Dropdown
        self.convergence_plot_y_input_select = Select(title="Metric:", value="feasibility",
                                     options=["feasibility", "optimality"])
        self.convergence_plot_y_input_select.on_change('value', self._y_input_update)

        self.convergence_plot = figure(title="Optimization Parameter Progress",
                                       x_axis_label='Iterations',
                                       y_axis_label=self.convergence_plot_y_input_select.value, y_axis_type="log")
        self.convergence_plot.line(x="x_vals", y="y_vals", line_width=2, source=self.source)

        # Case select a design variables
        self.case_select = Select(title="Case:")
        self.case_select.on_change('value', self._case_select_update)
        self.case_select.width = 500
        self.case_select.align = "end"

        self.variable_select = Select(title="Variable:")
        self.variable_select.on_change('value', self._variable_select_update)
        self.variable_select.width = 500
        self.variable_select.align = "end"

        self.prob_variables_plot = figure(title="Problem Variables",
                                       x_axis_label="Variable Length",
                                       y_axis_label=self.variable_select.value)
        self.prob_variables_plot.line(x="x_vals", y="y_vals", line_width=2, source=self.var_data)
        self.prob_variables_plot.min_border_left = 100


        # Doc layout
        self.doc = doc
        self.doc.title = "Optimization Progress Visualization"
        plot1 = column(self.convergence_plot, self.convergence_plot_y_input_select)
        plot2 = column(self.prob_variables_plot, self.case_select, self.variable_select)

        self.doc.add_root(row(plot1, plot2))
        # self.doc.sizing_mode = 'scale_both'
        # self.doc.add_root(column(row(self.convergence_plot, self.convergence_plot_y_input_select),
        #                     row(self.prob_variables_plot, column(self.case_select, self.variable_select))))
        self.live_updating = doc.add_periodic_callback(self._update, self.update_freq)

    def _case_select_update(self, attr, old, new):
        self.case_select.value = new
        self._update_variables_plot()

    def _variable_select_update(self, attr, old, new):
        self.variable_select.value = new
        self.prob_variables_plot.yaxis.axis_label = new
        self._update_variables_plot()

    def _y_input_update(self, attr, old, new):
        self.convergence_plot_y_input_select.value = new
        self.convergence_plot.yaxis.axis_label = new
        self._update()

    def _update_variables_plot(self):
        case = self.cr.get_case(self.case_select.value)
        variable = case.outputs[self.variable_select.value]

        new_data = dict(
            x_vals=list(range(len(variable))),
            y_vals=variable,
        )
        self.var_data.data = new_data

    def _update(self):
        """
        Parse and update the source data if new data is present.
        """
        opt_data = self._parse()

        if opt_data:
            new_data = dict(
                x_vals=opt_data["nMajor"],
                y_vals=opt_data[self.convergence_plot_y_input_select.value],
            )
            try:
                self._parse_cases_for_variables()
                self.init_call_made = True
            except KeyError:
                pass
        else:
            new_data = dict(
                x_vals=[],
                y_vals=[],
            )

        self.source.data = new_data