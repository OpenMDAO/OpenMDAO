"""Define the CaseViewer class."""

import warnings

try:
    from bokeh.io import show, output_notebook, curdoc
    from bokeh.models import Select, HoverTool, MultiSelect, Paragraph, CheckboxGroup
    from bokeh.layouts import row, column
    from bokeh.plotting import figure, ColumnDataSource
    from bokeh.palettes import Category20, Turbo256
    from bokeh.util.warnings import BokehUserWarning
except ImportError:
    bokeh = None

from openmdao.utils.notebook_utils import notebook
from openmdao.recorders.case_reader import CaseReader
from openmdao.recorders.sqlite_reader import SqliteCaseReader

import numpy as np
import copy


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
    doc : Document or None
        The Bokeh document to build.

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

    def __init__(self, data, port=8888, doc=None):
        """
        Initialize attributes.
        """
        self.circle_data = ColumnDataSource(dict(x_vals=[], y_vals=[], color=[], cases=[]))
        self.multi_line_data = ColumnDataSource(dict(x_vals=[], y_vals=[], color=[], cases=[]))

        if isinstance(data, str):
            self.cr = CaseReader(data)
        elif isinstance(data, SqliteCaseReader):
            self.cr = data

        self._case_iter_str = "Case Iterations"
        self._num_points_str = "Variable Array Index"

        if notebook:
            output_notebook()
            warnings.simplefilter(action='ignore', category=BokehUserWarning)
            show(self._make_plot, notebook_handle=True, notebook_url=f"http://localhost:{port}")
        else:
            warnings.simplefilter(action='ignore', category=BokehUserWarning)
            if not doc:
                doc = curdoc()
            self._make_plot(doc)

    def _case_reader_to_dict(self):
        self.case_dict = {}
        if self.case.outputs is not None:
            self.case_dict.update(self.case.outputs)
        if self.case.inputs is not None:
            self.case_dict.update(self.case.inputs)
        if self.case.residuals is not None:
            self.case_dict.update(self.case.residuals)

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
        special_case_vals = [self._num_points_str, self._case_iter_str]

        self._case_reader_to_dict()

        for var_dict in [self.case.outputs, self.case.inputs, self.case.residuals]:
            if var_dict is not None:
                case_vars += list(var_dict.keys())

        if var_to_compare in special_case_vals:
            return self.io_options_x

        for variable in variables:
            if variable in case_vars and var_to_compare in case_vars and \
                    (isinstance(self.case_dict[variable], np.ndarray) and
                     isinstance(self.case_dict[var_to_compare], np.ndarray)):
                if self.case_dict[variable].size == self.case_dict[var_to_compare].size:
                    var_list.append(variable)

        if var_list:
            return sorted(var_list) + special_case_vals
        elif variables != special_case_vals:
            return special_case_vals

    def _case_options(self, source):
        cases = self.cr.list_cases(source, out_stream=None)
        case_select = []
        if source == 'driver':
            case_num = 0
            for case in cases:
                pipe_count = 0
                for letter in case:
                    if letter == "|":
                        pipe_count += 1
                if pipe_count <= 1:
                    case_select.append((str(case_num), case))
                    case_num += 1

            return case_select
        else:
            return [(str(i), case) for i, case in
                    enumerate(self.cr.list_cases(source, out_stream=None))]

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
        self.case_options = self._case_options(source_options[0])
        self.io_options_x = self.cr.list_source_vars(source_options[0], out_stream=None)

        for key in self.io_options_x:
            if self.io_options_x[key]:
                io_starting_option = self.io_options_x[key][0]
            self.io_options_x[key] = sorted(self.io_options_x[key])
        self.io_options_x['Other'] = [self._num_points_str, self._case_iter_str]

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

        self.doc.add_root(self.layout)

    def _source_update(self, attr, old, new):
        """
        Update function for when the source dropdown is updated.
        """
        self.case_select.options = self._case_options(new)

        self.io_options_x = self.cr.list_source_vars(new, out_stream=None)
        self.io_options_x['Other'] = [self._num_points_str, self._case_iter_str]
        self.io_select_x.options = self.io_options_x

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
        _num_case_check_x = (self.io_select_x.value in [self._case_iter_str, self._num_points_str])
        _num_case_check_y = (self.io_select_y.value in [self._case_iter_str, self._num_points_str])
        if self.case_iter_select.value == "Norm":
            data = np.linalg.norm(data, axis=1).reshape(1, num_of_cases)
            case_array = np.arange(num_of_cases).reshape(1, num_of_cases)
            self._tooltip_management()
            return data, case_array

        elif self.case_iter_select.value == "Vector Lines":
            data = data.T
            case_array = case_array.T
            self._tooltip_management(False)
            return data, case_array

        elif self.case_iter_select.value == "Min/Max":
            self._tooltip_management()
            return data, case_array

        elif (set(case_array.flatten()) == {0.} or set(data.flatten()) == {0.}) and \
                not (_num_case_check_x and _num_case_check_y):
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
            self.case_iter_select.value = self.case_iter_options[0]
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

        min_case_iter = True
        if len(self.case_options) != 1 and (self._case_iter_x or self._case_iter_y):
            if len(self.case_select.options) > 1 and len(self.case_select.value) == 1:
                current_val = int(self.case_select.value[0])
                if current_val == len(self.case_options) - 1:
                    self.case_select.value.insert(current_val - 1, str(current_val - 1))
                else:
                    self.case_select.value.append(str(current_val + 1))
            elif len(self.case_select.options) == 1:
                min_case_iter = False
                self.warning_box.text = "Case Iterations needs 2 or more cases to function"

        if min_case_iter:
            for i in self.case_select.value:
                self.case = self.cr.get_case(self.case_select.options[int(i)][1])

                self._case_reader_to_dict()

                x_io = y_io = self.case_dict

                if (num_points_y and num_points_x) or (self._case_iter_x and self._case_iter_y):
                    x_variable = np.zeros(1)
                    y_variable = np.zeros(1)
                elif (num_points_x and self._case_iter_y) or (num_points_y and self._case_iter_x):
                    x_variable = np.zeros(1)
                    y_variable = np.zeros(1)
                    self.warning_box.text = ("NOTE: Cannot compare Variable Array Index to Case "
                                             "Iterations")
                elif num_points_y or self._case_iter_y:
                    if isinstance(self.case[self.io_select_x.value], (np.ndarray, list, float)):
                        x_variable = self.case[self.io_select_x.value].flatten()
                    else:
                        x_variable = np.zeros(1)
                        print(f"X is a non compatible type")
                    y_variable = np.arange(len(x_variable))
                elif num_points_x or self._case_iter_x:
                    if isinstance(self.case[self.io_select_y.value], (np.ndarray, list, float)):
                        y_variable = self.case[self.io_select_y.value].flatten()
                    else:
                        y_variable = np.zeros(1)
                        print(f"Y is a non compatible type")
                    x_variable = np.arange(len(y_variable))
                else:
                    x_variable = self.case[self.io_select_x.value].flatten()
                    y_variable = self.case[self.io_select_y.value].flatten()

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
                    new_data['y_vals'], new_data['x_vals'] = \
                        self._case_plot_calc(new_data['y_vals'], new_data['x_vals'])
                elif self._case_iter_y:
                    new_data['y_vals'] = np.full((y_len, case_len), [list(range(0, case_len))]).T
                    new_data['x_vals'], new_data['y_vals'] = \
                        self._case_plot_calc(new_data['x_vals'], new_data['y_vals'])

                if self.case_iter_select.value == "Vector Lines":
                    new_data['cases'] = new_data['cases'][0:len(new_data['x_vals'])]
                    new_data['color'] = self._line_color_list(new_data['x_vals'])

                new_data['x_vals'] = new_data['x_vals'].tolist()
                new_data['y_vals'] = new_data['y_vals'].tolist()

                self.multi_line_data.data = new_data
                self.circle_data.data = {"x_vals": [], "y_vals": [], "color": [], "cases": []}
            else:
                if self._case_iter_x:
                    new_data['x_vals'] = np.full((x_len, case_len), [list(range(0, case_len))]).T
                    new_data['y_vals'], new_data['x_vals'] = \
                        self._case_plot_calc(new_data['y_vals'], new_data['x_vals'])
                elif self._case_iter_y:
                    new_data['y_vals'] = np.full((y_len, case_len), [list(range(0, case_len))]).T
                    new_data['x_vals'], new_data['y_vals'] = \
                        self._case_plot_calc(new_data['x_vals'], new_data['y_vals'])

                if self.case_iter_select.value == "Vector Lines":
                    new_data['cases'] = new_data['cases'][0:len(new_data['x_vals'])]
                    new_data['color'] = self._line_color_list(new_data['x_vals'])

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
        var_length = len(x_var_vals)

        if var_length <= 3:
            colors = list(Category20[3])
            while len(colors) > var_length:
                colors.pop()
        else:
            if var_length > 20 and var_length < 256:
                colors = list(Turbo256[:var_length])
            elif var_length < 20:
                colors = list(Category20[var_length])
            else:
                self.warning_box.text = "NOTE: Cannot compare more than 256 cases"
                colors = list(Turbo256)

        print(colors)
        return colors
