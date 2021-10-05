import json
from openmdao.utils.notebook_utils import notebook
from bokeh.io import show
from bokeh.models import Select, HoverTool
from bokeh.layouts import row, column
import openmdao.api as om

from bokeh.io import show, output_notebook
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

        self.line_data = ColumnDataSource(dict(
            x_vals=[], y_vals=[]
        ))

        self.circle_data = ColumnDataSource(dict(
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
        for key in io_options:
            io_options[key].append("segment_length")

        for val in io_options.values():
            if val and val[0] != "segment_length":
                io_starting_option = val[0]
                break

        # Need to have a checkbox that triggers log scale
        self.variables_plot = figure(title="Problem Variables", x_axis_label="Variable Length",
                                     y_axis_label="Variable X")

        circle_plot = self.variables_plot.circle(x="x_vals", y="y_vals", source=self.circle_data)
        line_plot = self.variables_plot.line(x="x_vals", y="y_vals", line_width=2,
                                          source=self.line_data)

        # Hover tool needs to be adjusted
        ht = HoverTool(renderers=[line_plot, circle_plot],
            tooltips=[
                ( 'x',  '@x_vals'),
                ( 'y',  '@y_vals' )
            ],

            mode='vline',
        )
        self.variables_plot.add_tools(ht)


        self.source_select = Select(title="Source:", value=source_options[0],
                                    options=self.cr.list_sources(out_stream=None))

        self.case_select = Select(title="Case:", value=case_options[0],
                                  options=list(self.cr.list_cases(out_stream=None)))
        self.case_select.on_change('value', self._case_select_update)

        self.io_select_y = Select(title="Y Value:", value=io_starting_option, options=io_options)
        self.io_select_y.on_change('value', self._io_var_select_y_update)
        self.variables_plot.yaxis.axis_label = io_starting_option
        self.variables_plot.xaxis.axis_label = io_starting_option

        self.io_select_x = Select(title="X Value:", value=io_starting_option, options=io_options)
        self.io_select_x.on_change('value', self._io_var_select_x_update)

        layout = row(self.variables_plot, column(self.source_select,
                                                 self.case_select,
                                                 self.io_select_y,
                                                 self.io_select_x))

        self.doc.add_root(layout)
        self.update()

    def _case_select_update(self, attr, old, new):
        self.update()

    def _io_var_select_y_update(self, attr, old, new):
        self.variables_plot.yaxis.axis_label = new
        self.update()

    def _io_var_select_x_update(self, attr, old, new):
        self.variables_plot.xaxis.axis_label = new
        self.update()

    def update(self):
        case = self.cr.get_case(self.case_select.value)
        # Need to adjust this to pull from inputs, outputs, and residuals
        if self.io_select_y.value == "segment_length":
            x_variable = case.outputs[self.io_select_x.value]
            y_variable = list(range(len(x_variable)))
        elif self.io_select_x.value == "segment_length":
            y_variable = case.outputs[self.io_select_y.value]
            x_variable = list(range(len(y_variable)))
        else:
            x_variable = case.outputs[self.io_select_x.value]
            y_variable = case.outputs[self.io_select_y.value]


        new_data = dict(
            x_vals=x_variable,
            y_vals=y_variable,
        )

        if len(new_data['x_vals']) >= 2:
            self.line_data.data = new_data
            self.circle_data.data = {"x_vals": [], "y_vals": []}
        else:
            self.circle_data.data = new_data
            self.line_data.data = {"x_vals": [], "y_vals": []}
