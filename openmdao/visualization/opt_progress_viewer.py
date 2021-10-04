import json
from openmdao.utils.notebook_utils import notebook
from ipywidgets import interact
from bokeh.io import show, curdoc
from bokeh.models import Select, CustomJS
from bokeh.layouts import row, column
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

import openmdao.api as om

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, ColumnDataSource

class OptViewer(object):

    def __init__(self, data, port=8888):
        """
        Initialize threading.

        port : int
            What port to host Bokeh server on.
        data : CaseRecorder or str
            A path to the recorder file or CaseRecorder.
            Currently only sqlite database files recorded via SqliteRecorder are supported.
        """
        if not notebook:
            raise RuntimeError("OptView must be run in a notebook environment")

        self.data = data

        self.var_data = ColumnDataSource(dict(
            x_vals=[], y_vals=[]
        ))

        output_notebook()

        show(self._make_plot, notebook_handle=True, notebook_url=("http://localhost:" + str(port)))

    def _parse_cases(self):
        if isinstance(self.data, str):
            self.cr = om.CaseReader(self.data)
        if isinstance(self.data, om.SqliteRecorder):
            self.cr = self.data

    def _parse(self):
        """
        Parse the case recorder.
        """
        opt_data = None

        self._parse_cases()
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

        return opt_data

    def _make_plot(self, doc):
        self._parse()

        self.doc = doc
        source_options = self.cr.list_sources(out_stream=None)
        case_options = list(self.cr.list_cases(out_stream=None))
        io_options = self.cr.list_source_vars(source_options[0], out_stream=None)
        for key, val in io_options.items():
            if val:
                io_starting_option = val[0]
                break


        self.variables_plot = figure(title="Problem Variables", x_axis_label="Variable Length",
                                     y_axis_label="Variable X")
        self.variables_plot.line(x="x_vals", y="y_vals", line_width=2,
                                          source=self.var_data)

        self.source_select = Select(title="Source:", value=source_options[0],
                                    options=self.cr.list_sources(out_stream=None))
        # self.source_select.on_change('value', self._variable_select_update)

        self.case_select = Select(title="Case:", value=case_options[0],
                                  options=list(self.cr.list_cases(out_stream=None)))
        # self.case_select.on_change('value', self._variable_select_update)

        self.io_select = Select(title="IO Data:", value=io_starting_option,
                                options=self.cr.list_source_vars(self.source_select.options[0],
                                                                 out_stream=None))
        self.io_select.on_change('value', self.io_var_select)

        layout = row(self.variables_plot, column(self.source_select,
                                                 self.case_select,
                                                 self.io_select))

        self.doc.add_root(layout)
        self.update()


    def io_var_select(self, attr, old, new):
        self.variables_plot.yaxis.axis_label = new
        self.update()

    def update(self,):
        case = self.cr.get_case(self.case_select.value)
        variable = case.outputs[self.io_select.value]

        new_data = dict(
            x_vals=list(range(len(variable))),
            y_vals=variable,
        )
        self.var_data.data = new_data

        # push_notebook()

