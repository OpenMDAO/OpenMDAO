from ipywidgets import interact
import json

import openmdao.api as om

from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, ColumnDataSource

class OptViewer(object):

    def __init__(self, data):
        """
        Initialize threading.

        port : int
            What port to host Bokeh server on.
        data : CaseRecorder or str
            A path to the recorder file or CaseRecorder.
            Currently only sqlite database files recorded via SqliteRecorder are supported.
        """
        self.data = data

        self.var_data = ColumnDataSource(dict(
            x_vals=[], y_vals=[]
        ))

        output_notebook()

        self._make_plot()

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

    def _make_plot(self):
        self._parse()

        cases = list(self.cr.list_cases(out_stream=None))
        variables = self.cr.list_source_vars('driver', out_stream=None)['outputs']

        variables_plot = figure(title="Problem Variables", x_axis_label="Variable Length", y_axis_label="Variable X")
        v_line = variables_plot.line(x="x_vals", y="y_vals", line_width=2, source=self.var_data)

        show(variables_plot, notebook_handle=True)

        interact(self.update, case=cases, variable=variables)

    def update(self, case, variable):
        case_val = self.cr.get_case(case)
        variable_val = case_val.outputs[variable]

        new_data = dict(
            x_vals=list(range(len(variable_val))),
            y_vals=variable_val,
        )
        self.var_data.data = new_data

        push_notebook()