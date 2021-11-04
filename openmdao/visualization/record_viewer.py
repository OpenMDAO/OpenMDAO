"""Define the CaseViewer class."""

import json
import warnings

try:
    from bokeh.io import show, output_notebook
    from bokeh.models import Select, HoverTool, MultiSelect, Paragraph
    from bokeh.layouts import row, column
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.palettes import Category20, Turbo256
    from bokeh.util.warnings import BokehUserWarning
except ImportError:
    bokeh = None
    print("Bokeh not found")

from openmdao.utils.notebook_utils import notebook
from openmdao.recorders.case_reader import CaseReader
from openmdao.recorders.sqlite_reader import SqliteCaseReader

import numpy as np


class CaseViewer(object):
    """
    Visualizer to plot variables vs cases, variables vs variables, and more.

    Parameters
    ----------
    data : CaseRecorder or str
        A path to the recorder file or CaseRecorder.
        Currently only sqlite database files recorded via SqliteCaseReader are supported.
    port : int
        What port to host Bokeh server on.

    Attributes
    ----------
    circle_data : ColumnDataSource
        A Bokeh ColumnDataSource for non vectorized cases.
    multi_line_data : ColumnDataSource
        A Bokeh ColumnDataSource for vectorized cases.
    cr : CaseReader
        Recorded data.
    _case_iter_str : str
        Frequently used string.
    _num_points_str : str
        Frequently used string.
    """

    def __init__(self, data, port=8888):
        """
        Initialize attributes.
        """
        if not notebook:
            raise RuntimeError(f"{self.__class__.__name__} must be run in a notebook environment")

        warnings.simplefilter(action='ignore', category=BokehUserWarning)

        self.circle_data = ColumnDataSource(dict(x_vals=[], y_vals=[], color=[], cases=[]))
        self.multi_line_data = ColumnDataSource(dict(x_vals=[], y_vals=[], color=[], cases=[]))

        if isinstance(data, str):
            self.cr = CaseReader(data)
        elif isinstance(data, SqliteCaseReader):
            self.cr = data

        self._case_iter_str = "Case Iterations"
        self._num_points_str = "Number of Points"

        output_notebook()

        show(self._make_plot, notebook_handle=True, notebook_url=("http://localhost:" + str(port)))

    def _var_compatability_check(self, variables, var_to_compare):
        """
        Check and filter variables with same vector length as var_to_compare.

        Parameters
        ----------
        variables : dict
            Dictionary of variables to check the vector length.
        var_to_compare : str
            String to compare variable vector lengths for compatibility.

        Returns
        -------
        List
            List of variables with identical variable vector lengths.
        """
        variables = list(set(variables['inputs'] + variables['outputs'] + variables['residuals']))
        var_list = []
        case_vars = []
        special_case_vars = [self._num_points_str, self._case_iter_str]

        for i in [self.case.outputs, self.case.inputs, self.case.residuals]:
            if i is not None:
                case_vars += list(i.keys())

        if var_to_compare in special_case_vars:
            return self.io_options_x

        for variable in variables:
            if variable in case_vars and variable not in special_case_vars:
                if len(self.case[variable].flatten()) == len(self.case[var_to_compare].flatten()):
                    var_list.append(variable)

        if var_list:
            return sorted(var_list) + [self._num_points_str] + [self._case_iter_str]
        elif variables != [self._num_points_str, self._case_iter_str]:
            return [self._num_points_str, self._case_iter_str]

    def _make_plot(self, doc):
        """
        Create the plot and elements.

        Parameters
        ----------
        doc : Document
            The bokeh document to build.
        """
        self.doc = doc
        source_options = self.cr.list_sources(out_stream=None)
        self.case_options = [(str(i), case) for i, case in
                             enumerate(self.cr.list_cases(source_options[0], out_stream=None))]
        self.io_options_x = self.cr.list_source_vars(source_options[0], out_stream=None)

        for key in self.io_options_x:
            if self.io_options_x[key]:
                io_starting_option = self.io_options_x[key][0]
            self.io_options_x[key] = sorted(self.io_options_x[key])

        self.variables_plot = figure(title="Problem Variables", x_axis_label="Variable Length",
                                     y_axis_label="Variable X")
        self.variables_plot.circle(x="x_vals", y="y_vals", source=self.circle_data)

        self.line_plot = self.variables_plot.multi_line(xs="x_vals", ys="y_vals", line_width=2,
                                                        line_color='color',
                                                        source=self.multi_line_data)
        self.ht = HoverTool(renderers=[self.line_plot], tooltips=[('Case', '@cases')], mode='mouse')

        self.variables_plot.add_tools(self.ht)

        self.source_select = Select(title="Source:", value=source_options[0],
                                    options=source_options)
        self.source_select.on_change('value', self._source_update)

        self.case_select = MultiSelect(title="Case:", value=["0"], options=self.case_options)
        self.case_select.on_change('value', self._case_select_update)
        self.case_select.height = 300

        self.io_select_x = Select(title="X Value:", value=io_starting_option,
                                  options=self.io_options_x)
        self.io_select_x.on_change('value', self._io_var_select_x_update)

        self.io_select_y = Select(title="Y Value:", value=self._num_points_str,
                                  options=[self._num_points_str])
        self.io_select_y.on_change('value', self._io_var_select_y_update)

        self.case_iter_options = ['Min/Max', "Norm", "Vector Lines"]
        self.case_iter_select = Select(title="Case Iteration Plot Options", value="N/A",
                                       options=["N/A"])
        self.case_iter_select.on_change('value', self._case_iter_select_update)

        self.variables_plot.yaxis.axis_label = self.io_select_y.value
        self.variables_plot.xaxis.axis_label = self.io_select_x.value

        self.warning_box = Paragraph(text="""""", width=350, height=100)

        self.layout = row(self.variables_plot, column(self.source_select,
                                                      self.case_select,
                                                      self.io_select_x,
                                                      self.io_select_y,
                                                      self.case_iter_select,
                                                      self.warning_box
                                                      ))

        self._update()

        self.io_select_y.options = self._var_compatability_check(self.io_options_x,
                                                                 self.io_select_x.value)
        for key, val in self.io_select_x.options.items():
            self.io_select_x.options[key] = val + [self._num_points_str] + [self._case_iter_str]

        self.doc.add_root(self.layout)

    def _source_update(self, attr, old, new):
        """
        Update function for when the source dropdown is updated.
        """
        self.case_select.options = [(str(i), case) for i, case in
                                    enumerate(self.cr.list_cases(new, out_stream=None))]
        self.source_select.value = new
        self.case_select.value = ['0']
        self._update()

    def _case_select_update(self, attr, old, new):
        """
        Update function for when one or more cases are added by the user.
        """
        self._update()

    def _case_iter_select_update(self, attr, old, new):
        self._update()

    def _tooltip_management(self, pop=True):
        if isinstance(self.variables_plot.tools[-1], HoverTool) and pop:
            self.variables_plot.tools.pop(-1)
        elif not isinstance(self.variables_plot.tools[-1], HoverTool) and not pop:
            self.variables_plot.add_tools(self.ht)

    def _case_plot_calc(self, data, case_array):
        """
        Convert input arrays into correct format for case iterations.

        Parameters
        ----------
        data : np.array
            Array of variable data.
        case_array : np.array
            Aranged array of selected cases for multi_line plot.

        Returns
        -------
        np.array
            Array of variable data.
        np.array
            Aranged array of selected cases for multi_line plot
        """
        num_of_cases = data.shape[0]
        if self.case_iter_select.value == "Norm":
            data = np.linalg.norm(data, axis=1).reshape(1, num_of_cases)
            case_array = np.arange(num_of_cases).reshape(1, num_of_cases)
            self._tooltip_management()
            return data, case_array

        elif self.case_iter_select.value == "Vector Lines":
            data = data.T
            case_array = case_array.T
            self._tooltip_management()
            return data, case_array

        elif set(case_array.flatten()) == {0.} or set(data.flatten()) == {0.}:
            self.warning_box.text = ("NOTE: One or more variables are 0 arrays. Select a different "
                                     "case or variable")
            self._tooltip_management(False)
            return data, case_array
        else:
            self._tooltip_management(False)
            return data, case_array

    def _io_var_select_y_update(self, attr, old, new):
        """
        Update function for when the source Y Value dropdown is updated.
        """
        if new == self._case_iter_str or self.io_select_x.value == self._case_iter_str:
            self.case_iter_select.options = self.case_iter_options
        else:
            self.case_iter_select.options = ["N/A"]

        if self.io_select_x.value == self._num_points_str or new == self._case_iter_str:
            self.variables_plot.yaxis.axis_label = new
            self.variables_plot.xaxis.axis_label = self.io_select_x.value
        else:
            self.variables_plot.yaxis.axis_label = new
            self.variables_plot.xaxis.axis_label = self.io_select_x.value

        self._update()

    def _io_var_select_x_update(self, attr, old, new):
        """
        Update function for when the X Value dropdown is updated.
        """
        if new == self._case_iter_str or self.io_select_y.value == self._case_iter_str:
            self.io_select_y.options = self.io_select_x.options
            self.case_iter_select.options = self.case_iter_options
        else:
            self.case_iter_select.options = ["N/A"]

        if self.io_select_x.value == self._num_points_str or new == self._case_iter_str:
            self.variables_plot.xaxis.axis_label = new
            self.variables_plot.yaxis.axis_label = self.io_select_y.value
        else:
            self.io_select_y.options = self._var_compatability_check(self.io_options_x, new)

            self.variables_plot.xaxis.axis_label = new
            self.variables_plot.yaxis.axis_label = self.io_select_y.value

        self._update()

    def _update(self):
        """
        Change plot based on source, case, or variable changes.
        """
        new_data = dict(
            x_vals=[],
            y_vals=[],
            color=[],
            cases=[]
        )
        self._tooltip_management(False)
        self.warning_box.text = ""
        num_points_x = num_points_y = False
        self._case_iter_x = self._case_iter_y = False

        if self.io_select_y.value == self._num_points_str:
            num_points_y = True
        if self.io_select_x.value == self._num_points_str:
            num_points_x = True

        if self.io_select_y.value == self._case_iter_str:
            self._case_iter_y = True
        if self.io_select_x.value == self._case_iter_str:
            self._case_iter_x = True

        if len(self.case_options) != 1 and (self._case_iter_x or self._case_iter_y):
            if len(self.case_select.value) == 1:
                current_val = int(self.case_select.value[0])
                self.case_select.value.append(str(current_val + 1))
        elif self._case_iter_x or self._case_iter_y:
            self.warning_box.text = ("NOTE: Only one case found. Unable to view case iterations vs "
                                     "variable")

        for i in self.case_select.value:
            self.case = self.cr.get_case(self.case_options[int(i)][1])

            for key, val in self.io_options_x.items():
                if self.io_select_y.value in val:
                    y_io = getattr(self.case, key)

                if self.io_select_x.value in val:
                    x_io = getattr(self.case, key)

            if (num_points_y and num_points_x) or (self._case_iter_x and self._case_iter_y):
                x_variable = np.zeros(1)
                y_variable = np.zeros(1)
            elif (num_points_x and self._case_iter_y) or (num_points_y and self._case_iter_x):
                x_variable = np.zeros(1)
                y_variable = np.zeros(1)
                self.warning_box.text = "NOTE: Cannot compare Number of Points to Case Iterations"
            elif num_points_y or self._case_iter_y:
                x_variable = x_io[self.io_select_x.value].flatten()
                y_variable = np.arange(len(x_variable))
            elif num_points_x or self._case_iter_x:
                y_variable = y_io[self.io_select_y.value].flatten()
                x_variable = np.arange(len(y_variable))
            else:
                x_variable = x_io[self.io_select_x.value].flatten()
                y_variable = y_io[self.io_select_y.value].flatten()

            if not isinstance(new_data['x_vals'], np.ndarray):
                new_data['x_vals'] = np.empty((0, len(x_variable)), float)
                new_data['y_vals'] = np.empty((0, len(y_variable)), float)

            new_data['x_vals'] = np.vstack((new_data['x_vals'], x_variable))
            new_data['y_vals'] = np.vstack((new_data['y_vals'], y_variable))

        x_len = new_data['x_vals'].shape[1]
        y_len = new_data['y_vals'].shape[1]
        new_data['color'] = self._line_color_list(new_data['x_vals'])
        new_data['cases'] = [self.case_options[int(case)][1] for case in self.case_select.value]
        case_len = len(new_data['cases'])

        if new_data['x_vals'].shape[1] > 1:
            if (new_data['x_vals'].shape[0], new_data['y_vals'].shape[0]) == (1, 1) and \
               (set(new_data['x_vals'][0]), set(new_data['y_vals'][0])) == ({0.}, {0.}):
                self.warning_box.text = ("NOTE: Both X and Y values contain zeros for values, "
                                         "unable to plot")

            if self._case_iter_x:
                new_data['x_vals'] = np.full((x_len, case_len), [list(range(0, case_len))]).T
                new_data['y_vals'], new_data['x_vals'] = self._case_plot_calc(new_data['y_vals'],
                                                                              new_data['x_vals'])
            elif self._case_iter_y:
                new_data['y_vals'] = np.full((y_len, case_len), [list(range(0, case_len))]).T
                new_data['x_vals'], new_data['y_vals'] = self._case_plot_calc(new_data['x_vals'],
                                                                              new_data['y_vals'])

            new_data['x_vals'] = new_data['x_vals'].tolist()
            new_data['y_vals'] = new_data['y_vals'].tolist()

            self.multi_line_data.data = new_data
            self.circle_data.data = {"x_vals": [], "y_vals": [], "color": [], "cases": []}
        else:

            if self._case_iter_x:
                new_data['x_vals'] = np.full((x_len, case_len), [list(range(0, case_len))]).T
                new_data['y_vals'], new_data['x_vals'] = self._case_plot_calc(new_data['y_vals'],
                                                                              new_data['x_vals'])
            elif self._case_iter_y:
                new_data['y_vals'] = np.full((y_len, case_len), [list(range(0, case_len))]).T
                new_data['x_vals'], new_data['y_vals'] = self._case_plot_calc(new_data['x_vals'],
                                                                              new_data['y_vals'])

            new_data['x_vals'] = new_data['x_vals'].flatten().tolist()
            new_data['y_vals'] = new_data['y_vals'].flatten().tolist()

            self.circle_data.data = new_data
            self.multi_line_data.data = {"x_vals": [], "y_vals": [], "color": [], "cases": []}

    def _line_color_list(self, x_var_vals):
        """
        Create list of colors for multi line plots.

        Parameters
        ----------
        x_var_vals : np.array
            Array of x_vals to find number of colors.

        Returns
        -------
        list
            List of colors for multi_line or circle data.
        """
        length = len(x_var_vals)
        if length <= 3:
            colors = list(Category20[3])
            while len(colors) > length:
                colors.pop()
        else:
            if length > 20:
                colors = list(Turbo256[:length])
            elif length < 20:
                colors = list(Category20[length])
            else:
                self.warning_box.text = "NOTE: Cannot compare more than 256 cases"

        return colors
