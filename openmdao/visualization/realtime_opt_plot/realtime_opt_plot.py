"""A real-time plot monitoring the optimization process as an OpenMDAO script runs."""

import ctypes
import errno
import importlib
import os
import pathlib
import sys
from collections import defaultdict
import sqlite3

try:
    from bokeh.models import (
        ColumnDataSource,
        LinearAxis,
        Range1d,
        Toggle,
        Button,
        Column,
        Row,
        CustomJS,
        Div,
        ScrollBox,
        BasicTicker
    )

    from bokeh.models.tools import (
        BoxZoomTool,
        ResetTool,
        HoverTool,
        PanTool,
        WheelZoomTool,
        SaveTool,
        ZoomInTool,
        ZoomOutTool,
    )
    from bokeh.plotting import figure
    from bokeh.server.server import Server
    from bokeh.application.application import Application
    from bokeh.application.handlers import FunctionHandler
    from bokeh.palettes import Category20, Colorblind
    bokeh_available = True
except ImportError:
    bokeh_available = False

import numpy as np

from openmdao.recorders.sqlite_reader import SqliteCaseReader
from openmdao.recorders.case import Case

try:
    from openmdao.utils.gui_testing_utils import _get_free_port
except ImportError:
    # If _get_free_port is unavailable, the default port will be used
    def _get_free_port():
        return 5000


# _images_dir = pathlib.Path(importlib.util.find_spec("openmdao").origin).parent.joinpath(
#     "visualization/realtime_opt_plot/images/"
# )

_images_dir = pathlib.Path("images")


# Constants
# the time between calls to the udpate method
_time_between_callbacks_in_ms = 1000
# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10
# color of the plot line for the objective function
_obj_color = "black"
# color of the buttons for variables not being shown
_non_active_plot_color = "black"
_plot_line_width = 3
# how transparent is the area part of the plot for desvars that are vectors
_varea_alpha = 0.3
# the CSS for the toggle buttons to let user choose what variables to plot
_variable_list_header_font_size = "14"
_toggle_styles = """
    font-size: 22px;
    box-shadow:
        0 4px 6px rgba(0, 0, 0, 0.1),    /* Distant shadow */
        0 1px 3px rgba(0, 0, 0, 0.08),   /* Close shadow */
        inset 0 2px 2px rgba(255, 255, 255, 0.2);  /* Top inner highlight */
"""

# colors used for the plot lines and associated buttons and axes labels
# start with color-blind friendly colors and then use others if needed
if bokeh_available:
    _colorPalette = Colorblind[8] + Category20[20]

# This is the JavaScript code that gets run when a user clicks on
#   one of the toggle buttons that change what variables are plotted
callback_code = f"""
// The ColorManager provides color from a palette. When the
//   user turns off the plotting of a variable, the color is returned to the ColorManager
//   for later use with a different variable plot
if (typeof window.ColorManager === 'undefined') {{
    window.ColorManager = class {{
        constructor() {{
            if (ColorManager.instance) {{
                return ColorManager.instance;
            }}

            this.palette = colorPalette;
            this.usedColors = new Set();
            this.variableColorMap = new Map();
            ColorManager.instance = this;
        }} //  end of constructor

        getColor(variableName) {{
            if (this.variableColorMap.has(variableName)) {{
                return this.variableColorMap.get(variableName);
            }}

            const availableColor = this.palette.find(color => !this.usedColors.has(color));
            const newColor = availableColor ||
                                this.palette[this.usedColors.size % this.palette.length];

            this.usedColors.add(newColor);
            this.variableColorMap.set(variableName, newColor);
            return newColor;
        }} // end of getColor

        releaseColor(variableName) {{
            const color = this.variableColorMap.get(variableName);
            if (color) {{
                this.usedColors.delete(color);
                this.variableColorMap.delete(variableName);
            }}
        }} // end of releaseColor

    }}; // end of class definition

    window.colorManager = new window.ColorManager();
}}

// Get the toggle that triggered the callback
const toggle = cb_obj;
const index = toggles.indexOf(toggle);
// index value of 0 is for the objective variable whose axis
// is on the left. The index variable really refers to the list of toggle buttons.
// The axes list variable only is for desvars and cons, whose axes are on the right.
// The lines list variables includes all vars

// Set line visibility
lines[index].visible = toggle.active;

// Set axis visibility if it exists (all except first line)
if (index > 0 && index-1 < axes.length) {{
    axes[index-1].visible = toggle.active;
}}

let variable_name = cb_obj.label;
// if turning on, get a color and set the line, axis label, and toggle button to that color
if (toggle.active) {{
    let color = window.colorManager.getColor(variable_name);

    if (index > 0) {{
        axes[index-1].axis_label_text_color = color
    }}

    // using set_value is a workaround because of a bug in Bokeh.
    // see https://github.com/bokeh/bokeh/issues/14364 for more info
    if (lines[index].glyph.type == "VArea"){{
        lines[index].glyph.properties.fill_color.set_value(color);
    }}
    if (lines[index].glyph.type == "Line"){{
        lines[index].glyph.properties.line_color.set_value(color);
    }}

    // make the button background color the same as the line, just slightly transparent
    toggle.stylesheets = [`
        .bk-btn.bk-active {{
            background-color: rgb(from ${{color}} R G B / 0.3);
            {_toggle_styles}
        }}
    `];
// if turning off a variable, return the color to the pool
}} else {{
    window.colorManager.releaseColor(variable_name);
    toggle.stylesheets = [`
        .bk-btn {{
            {_toggle_styles}
        }}
    `];

}}
"""


def _is_process_running(pid):
    if sys.platform == "win32":
        # PROCESS_QUERY_LIMITED_INFORMATION is available on Windows Vista and later.
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

        # Attempt to open the process.
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        else:
            # If OpenProcess fails, check if it's due to access being denied.
            ERROR_ACCESS_DENIED = 5
            if ctypes.windll.kernel32.GetLastError() == ERROR_ACCESS_DENIED:
                return True
            return False
    else:
        try:
            os.kill(pid, 0)
        except OSError as err:
            if err.errno == errno.ESRCH:  # No such process
                return False
            elif err.errno == errno.EPERM:  # Process exists, no permission to signal
                return True
            else:
                raise
        else:
            return True


def _realtime_opt_plot_setup_parser(parser):
    """
    Set up the realtime plot subparser for the 'openmdao opt_plot' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        "case_recorder_filename",
        type=str,
        help="Name of openmdao case recorder filename. It should contain driver cases",
    )

    parser.add_argument('--pid', type=int, default=None,
                        help='Process ID of calling optimization script, '
                        'defaults to None if called by the user directly')
    parser.add_argument('--no-display', action='store_false', dest='show',
                        help="Do not launch browser showing plot. Used for CI testing")


def _realtime_opt_plot_cmd(options, user_args):
    """
    Run the opt_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if bokeh_available:
        realtime_opt_plot(options.case_recorder_filename, _time_between_callbacks_in_ms,
                          options.pid, options.show)
    else:
        print("The bokeh library is not installed so the real-time optimizaton "
              "lot is not available. ")
        return


def _update_y_min_max(name, y, y_min, y_max):
    """
    Update the y_min and y_max dicts containing the min and max of the variables.

    Parameters
    ----------
    name : str
        Name of the variable.
    y : double
        Value of the variable.
    y_min : dict
        Dict of mins of each variable.
    y_max : dict
        Dict of maxs of each variable.

    Returns
    -------
    bool
        True if either min or max were updated.
    """
    min_max_changed = False
    if y < y_min[name]:
        y_min[name] = y
        min_max_changed = True
    if y > y_max[name]:
        y_max[name] = y
        min_max_changed = True
    return min_max_changed


def _get_value_for_plotting(value_from_recorder, var_type):
    """
    Return the double value to be used for plotting the variable.

    Handles variables that are vectors.

    Parameters
    ----------
    value_from_recorder : numpy array
        Value of the variable.
    var_type : str
        String indicating which type of variable it is: 'objs', 'desvars' or 'cons'.

    Returns
    -------
    double
        The value to be used for plotting the variable.
    """
    if value_from_recorder is None or value_from_recorder.size == 0:
        return (0.0)
    if var_type == 'cons':
        # plot the worst case value
        return np.linalg.norm(value_from_recorder, ord=np.inf)
    elif var_type == 'objs':
        return value_from_recorder.item()  # get as scalar
    else:  # for desvars, use L2 norm
        return np.linalg.norm(value_from_recorder)


def _make_header_text_for_variable_chooser(header_text):
    """
    Return a Div to be used for the label for the type of variables in the variable list.

    Parameters
    ----------
    header_text : str
        Label string.

    Returns
    -------
    Div
        The header Div for the section of variables of that type.
    """
    header_text_div = Div(
        text=f"<b>{header_text}</b>",
        styles={"font-size": _variable_list_header_font_size},
    )
    return header_text_div


class _CaseRecorderTracker:
    """
    A class that is used to get information from a case recorder.

    These methods are not provided by the SqliteCaseReader class.
    """

    def __init__(self, case_recorder_filename):
        self._case_recorder_filename = case_recorder_filename
        self._cr = None
        self._initial_case = None  # need the initial case to get info about the variables
        self._next_id_to_read = 1

    def _open_case_recorder(self):
        if self._cr is None:
            self._cr = SqliteCaseReader(self._case_recorder_filename)

    def _get_case_by_counter(self, counter):
        # use SQL to see if a case with this counter exists
        with sqlite3.connect(self._case_recorder_filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "counter=:counter",
                        {"counter": counter})
            row = cur.fetchone()
        con.close()

        # use SqliteCaseReader code to get the data from this case
        if row:
            # TODO would be better to not have to open up the file each time
            self._open_case_recorder()
            var_info = self._cr.problem_metadata['variables']
            case = Case('driver', row, self._cr._prom2abs, self._cr._abs2prom, self._cr._abs2meta,
                        self._cr._conns, var_info, self._cr._format_version)

            return case
        else:
            return None

    def _get_data_from_case(self, driver_case):
        objs = driver_case.get_objectives(scaled=False)
        design_vars = driver_case.get_design_vars(scaled=False)
        constraints = driver_case.get_constraints(scaled=False)

        new_data = {
            "counter": int(driver_case.counter),
        }

        # get objectives
        objectives = {}
        for name, value in objs.items():
            objectives[name] = value
        new_data["objs"] = objectives

        # get des vars
        desvars = {}
        for name, value in design_vars.items():
            desvars[name] = value
        new_data["desvars"] = desvars

        # get cons
        cons = {}
        for name, value in constraints.items():
            cons[name] = value
        new_data["cons"] = cons

        return new_data

    def _get_new_case(self):
        # get the next unread case from the recorder
        driver_case = self._get_case_by_counter(self._next_id_to_read)
        if driver_case is None:
            return None

        if self._initial_case is None:
            self._initial_case = driver_case

        self._next_id_to_read += 1

        return driver_case

    def _get_obj_names(self):
        obj_vars = self._initial_case.get_objectives()
        return obj_vars.keys()

    def _get_desvar_names(self):
        design_vars = self._initial_case.get_design_vars()
        return design_vars.keys()

    def _get_cons_names(self):
        cons = self._initial_case.get_constraints()
        
        # cons._var_info['const.g']['lower']
        return cons.keys()
    
    def _get_constraint_bounds(self, name):
        cons = self._initial_case.get_constraints()
        var_info = cons._var_info[name]
        return (var_info['lower'], var_info['upper'])

    def _get_units(self, name):
        try:
            units = self._initial_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
        except KeyError:
            return "Unavailable"

        if units is None:
            units = "Unitless"
        return units


class _RealTimeOptPlot(object):
    """
    A class that handles all of the real-time plotting.

    Parameters
    ----------
    case_recorder_filename : str
        The path to the case recorder file.
    callback_period : double
        The time between Bokeh callback calls (in seconds).
    doc : bokeh.document.Document
        The Bokeh document which is a collection of plots, layouts, and widgets.
    pid_of_calling_script : int or None
        The process ID of the process that called the command to start the realtime plot.

        None if the plot was called for directly by the user.
    """

    def __init__(self, case_recorder_filename, callback_period, doc, pid_of_calling_script):
        """
        Construct and initialize _RealTimeOptPlot instance.
        """
        self._case_recorder_filename = case_recorder_filename
        self._case_tracker = _CaseRecorderTracker(case_recorder_filename)
        self._pid_of_calling_script = pid_of_calling_script

        self._source = None
        self._lower_bounds_cons_source = None
        self._upper_bounds_cons_source = None
        
       
        # self._up_arrow_image_path = str(_images_dir / "up_arrow.png")
        # self._down_arrow_image_path = str(_images_dir / "down_arrow.png")
        
        
        # import os 
        # print(f"{os.getcwd()=}") 
        
        # print(os.path.abspath("./openmdao/visualization/realtime_opt_plot/up_arrow.png"))
        
        # image_path = "./openmdao/visualization/realtime_opt_plot/images/up_arrow.png"
        # image_path = "./static/up_arrow.png"
        # absolute_path = os.path.abspath(image_path)
        
        
        # print(f"{absolute_path=}")
        # file_url = f"file://{absolute_path}"

        
        self._up_arrow_image_path = "./static/up_arrow_small.png"
        self._down_arrow_image_path = "./static/down_arrow_small.png"
        
        self._constraint_bounds = {}
        self._lines = []
        self._toggles = []
        self._column_items = []
        self._axes = []
        # flag to prevent updating label with units each time we get new data
        self._labels_updated_with_units = False
        self._source_stream_dict = None

        self._setup_figure()

        # used to keep track of the y min and max of the data so that
        # the axes ranges can be adjusted as data comes in
        self._y_min = defaultdict(lambda: float("inf"))
        self._y_max = defaultdict(lambda: float("-inf"))

        def _update():
            # this is the main method of the class. It gets called periodically by Bokeh
            # It looks for new data and if found, updates the plot with the new data
            new_case = self._case_tracker._get_new_case()

            if new_case is None:
                if self._pid_of_calling_script is None or not _is_process_running(
                    self._pid_of_calling_script
                ):
                    # no more new data in the case recorder file and the
                    #   optimization script stopped running, so no possible way to
                    #   get new data.
                    # But just keep sending the last data point.
                    # This is a hack to force the plot to re-draw.
                    # Otherwise if the user clicks on the variable buttons, the
                    #   lines will not change color because of the set_value hack done to get
                    #   get around the bug in setting the line color from JavaScript
                    self._source.stream(self._source_stream_dict)
                    self._lower_bounds_cons_source.stream(self._lower_bounds_cons_source_stream_dict)
                    self._upper_bounds_cons_source.stream(self._upper_bounds_cons_source_stream_dict)
                return

            new_data = self._case_tracker._get_data_from_case(new_case)

            # See if Bokeh source object is defined yet. If not, set it up
            # since now we have data from the case recorder with info about the
            # variables to be plotted.
            if self._source is None:
                self._setup_data_source()

                # Check to make sure we have one and only one objective before going farther
                obj_names = self._case_tracker._get_obj_names()
                if len(obj_names) != 1:
                    raise ValueError(
                        f"Plot requires there to be one and only one objective \
                            but {len(obj_names)} objectives found"
                    )

                # Create CustomJS callback for toggle buttons.
                # Pass in the data from the Python side that the JavaScript side
                #   needs
                legend_item_callback = CustomJS(
                    args=dict(
                        lines=self._lines,
                        axes=self._axes,
                        toggles=self._toggles,
                        colorPalette=_colorPalette,
                        plot=self.plot_figure,
                    ),
                    code=callback_code,
                )

                # For the variables, make lines, axes, and the buttons to turn on and
                #   off the variable plot.
                # All the lines and axes for the desvars and cons are created in
                #   Python but initially are not visible. They are turned on and
                #   off on the JavaScript side.

                # objs
                obj_label = _make_header_text_for_variable_chooser("OBJECTIVE")
                self._column_items.append(obj_label)

                for i, obj_name in enumerate(obj_names):
                    units = self._case_tracker._get_units(obj_name)
                    self.plot_figure.yaxis.axis_label = f"{obj_name} ({units})"
                    self._make_variable_button(f"{obj_name} ({units})", _obj_color,
                                               True, legend_item_callback)
                    self._make_line_and_hover_tool("objs", obj_name, False, _obj_color,
                                                   "solid", True)
                    value = new_data["objs"][obj_name]
                    float_value = _get_value_for_plotting(value, "objs")
                    # just give it some non-zero initial range since we only have one point
                    self.plot_figure.y_range = Range1d(float_value - 1, float_value + 1)

                # desvars
                desvars_label = _make_header_text_for_variable_chooser("DESIGN VARS")
                self._column_items.append(desvars_label)
                desvar_names = self._case_tracker._get_desvar_names()
                for i, desvar_name in enumerate(desvar_names):
                    units = self._case_tracker._get_units(desvar_name)
                    self._make_variable_button(
                        f"{desvar_name} ({units})",
                        _non_active_plot_color,
                        False,
                        legend_item_callback,
                    )
                    value = new_data["desvars"][desvar_name]
                    # for desvars, if value is a vector, use Bokeh Varea glyph
                    use_varea = value.size > 1
                    self._make_line_and_hover_tool(
                        "desvars",
                        desvar_name,
                        use_varea,
                        _non_active_plot_color,
                        "solid",
                        False,
                    )
                    float_value = _get_value_for_plotting(value, "desvars")
                    self._make_axis("desvars", desvar_name, float_value, units)

                # cons
                cons_label = _make_header_text_for_variable_chooser("CONSTRAINTS")
                self._column_items.append(cons_label)
                cons_names = self._case_tracker._get_cons_names()
                for i, cons_name in enumerate(cons_names):
                    units = self._case_tracker._get_units(cons_name)
                    self._make_variable_button(
                        f"{cons_name} ({units})",
                        _non_active_plot_color,
                        False,
                        legend_item_callback,
                    )
                    self._make_line_and_hover_tool(
                        "cons",
                        cons_name,
                        False,
                        _non_active_plot_color,
                        "dashed",
                        False,
                    )
                    value = new_data["cons"][cons_name]
                    float_value = _get_value_for_plotting(value, "cons")
                    self._make_axis("cons", cons_name, float_value, units)
                    
                    
                    self._constraint_bounds[cons_name] = self._case_tracker._get_constraint_bounds(cons_name)

                # Create a Column of the variable buttons and headers inside a scrolling window
                toggle_column = Column(
                    children=self._column_items,
                    sizing_mode="stretch_both",
                    height_policy="fit",
                    styles={
                        "overflow-y": "auto",
                        "border": "1px solid #ddd",
                        "padding": "8px",
                        "background-color": "#dddddd",
                        'max-height': '100vh'  # Ensures it doesn't exceed viewport
                    },
                )

                quit_button = Button(label="Quit Application", button_type="danger")

                # Define callback function for the quit button
                def quit_app():
                    raise KeyboardInterrupt("Quit button pressed")

                # Attach the callback to the button
                quit_button.on_click(quit_app)

                # header for the variable list
                label = Div(
                    text="Variables",
                    width=200,
                    styles={"font-size": "20px", "font-weight": "bold"},
                )
                label_and_toggle_column = Column(
                    quit_button,
                    label,
                    toggle_column,
                    sizing_mode="stretch_height",
                    height_policy="fit",
                        styles={
                            'max-height': '100vh'  # Ensures it doesn't exceed viewport
                        },
                )

                scroll_box = ScrollBox(
                    child=label_and_toggle_column,
                    sizing_mode="stretch_height",
                    height_policy="max",
                )

                graph = Row(self.plot_figure, scroll_box, sizing_mode="stretch_both")
                doc.add_root(graph)
                # end of self._source is None - plotting is setup

            # Do the actual update of the plot including updating the plot range and adding the new
            # data to the Bokeh plot stream
            counter = new_data["counter"]

            self._source_stream_dict = {"iteration": [counter]}
            # self._lower_bounds_cons_source_stream_dict = {"iteration": [counter]}
            # self._upper_bounds_cons_source_stream_dict = {"iteration": [counter]}
            self._lower_bounds_cons_source_stream_dict = {
                "iteration": [counter],
                "urls": [self._up_arrow_image_path],
                }
            self._upper_bounds_cons_source_stream_dict = {
                "iteration": [counter],
                "urls": [self._down_arrow_image_path],
                }
            
            iline = 0
            for obj_name, obj_value in new_data["objs"].items():
                float_obj_value = _get_value_for_plotting(obj_value, "objs")
                self._source_stream_dict[obj_name] = [float_obj_value]
                min_max_changed = _update_y_min_max(obj_name, float_obj_value,
                                                    self._y_min, self._y_max)
                if min_max_changed:
                    self.plot_figure.y_range.start = self._y_min[obj_name]
                    self.plot_figure.y_range.end = self._y_max[obj_name]
                iline += 1

            for desvar_name, desvar_value in new_data["desvars"].items():
                if not self._labels_updated_with_units and desvar_value.size > 1:
                    units = self._case_tracker._get_units(desvar_name)
                    self._toggles[iline].label = f"{desvar_name} ({units}) {desvar_value.shape}"
                min_max_changed = False
                min_max_changed = min_max_changed or _update_y_min_max(
                    desvar_name, np.min(desvar_value), self._y_min, self._y_max
                )
                min_max_changed = min_max_changed or _update_y_min_max(
                    desvar_name, np.max(desvar_value), self._y_min, self._y_max
                )
                if min_max_changed:
                    range = Range1d(
                        self._y_min[desvar_name], self._y_max[desvar_name]
                    )
                    self.plot_figure.extra_y_ranges[f"extra_y_{desvar_name}_min"] = range
                self._source_stream_dict[f"{desvar_name}_min"] = [np.min(desvar_value)]
                self._source_stream_dict[f"{desvar_name}_max"] = [np.max(desvar_value)]
                iline += 1

            for cons_name, cons_value in new_data["cons"].items():
                float_cons_value = _get_value_for_plotting(cons_value, "cons")
                if not self._labels_updated_with_units and cons_value.size > 1:
                    units = self._case_tracker._get_units(cons_name)
                    self._toggles[iline].label = f"{cons_name} ({units}) {cons_value.shape}"
                self._source_stream_dict[cons_name] = [float_cons_value]
                
                lower_bound, upper_bound = self._constraint_bounds[cons_name]
                if float_cons_value < lower_bound :
                    self._lower_bounds_cons_source_stream_dict[cons_name] = [float_cons_value]
                else:
                    self._lower_bounds_cons_source_stream_dict[cons_name] = [np.nan]
                if float_cons_value > upper_bound :
                    self._upper_bounds_cons_source_stream_dict[cons_name] = [float_cons_value]
                else:
                    self._upper_bounds_cons_source_stream_dict[cons_name] = [np.nan]
                min_max_changed = _update_y_min_max(
                    cons_name, float_cons_value, self._y_min, self._y_max)
                if min_max_changed:
                    range = Range1d(
                        self._y_min[cons_name], self._y_max[cons_name]
                    )
                    self.plot_figure.extra_y_ranges[f"extra_y_{cons_name}"] = range
                iline += 1
            self._source.stream(self._source_stream_dict)
            
            
          
            
            self._lower_bounds_cons_source.stream(self._lower_bounds_cons_source_stream_dict)
            self._upper_bounds_cons_source.stream(self._upper_bounds_cons_source_stream_dict)
            
            self._labels_updated_with_units = True
            # end of _update method

        doc.add_periodic_callback(_update, callback_period)
        doc.title = "OpenMDAO Optimization Progress Plot"

    def _setup_data_source(self):
        self._source_dict = {"iteration": []}

        # Obj
        obj_names = self._case_tracker._get_obj_names()
        for obj_name in obj_names:
            self._source_dict[obj_name] = []

        # Desvars
        desvar_names = self._case_tracker._get_desvar_names()
        for desvar_name in desvar_names:
            self._source_dict[f"{desvar_name}_min"] = []
            self._source_dict[f"{desvar_name}_max"] = []

        # Cons
        con_names = self._case_tracker._get_cons_names()
        for con_name in con_names:
            self._source_dict[con_name] = []

        self._source = ColumnDataSource(self._source_dict)
        
        # self._lower_bounds_cons_source_dict = {"iteration": []}
        self._lower_bounds_cons_source_dict = {
            "iteration": [],
            "urls": [],
            }
        # Cons
        con_names = self._case_tracker._get_cons_names()
        for con_name in con_names:
            self._lower_bounds_cons_source_dict[con_name] = []
        self._lower_bounds_cons_source = ColumnDataSource(self._lower_bounds_cons_source_dict)
        
      
        # self._upper_bounds_cons_source_dict = {"iteration": []}
        self._upper_bounds_cons_source_dict = {
            "iteration": [],
            "urls": [],
            }
        # Cons
        for con_name in con_names:
            self._upper_bounds_cons_source_dict[con_name] = []
        self._upper_bounds_cons_source = ColumnDataSource(self._upper_bounds_cons_source_dict)



    def _make_variable_button(self, varname, color, active, callback):
        toggle = Toggle(
            label=varname,
            active=active,
            margin=(0, 0, 8, 0),
        )
        toggle.js_on_change("active", callback)
        self._toggles.append(toggle)
        self._column_items.append(toggle)

        # Add custom CSS styles for both active and inactive states
        toggle.stylesheets = [
            f"""
                .bk-btn {{
                    {_toggle_styles}
                }}
                .bk-btn.bk-active {{
                    background-color: rgb(from #000000 R G B / 0.3);
                    {_toggle_styles}
                }}
            """
        ]
        return toggle

    def _make_line_and_hover_tool(self, var_type, varname, use_varea, color, line_dash, visible):
        if use_varea:
            line = self.plot_figure.varea(
                x="iteration",
                y1=f"{varname}_min",
                y2=f"{varname}_max",
                source=self._source,
                color=color,
                alpha=_varea_alpha,
                visible=visible,
            )
        else:
            if var_type == "desvars":
                y_name = f"{varname}_min"
            else:
                y_name = varname
            line = self.plot_figure.line(
                x="iteration",
                y=y_name,
                line_width=_plot_line_width,
                line_dash=line_dash,
                source=self._source,
                color=color,
                visible=visible,
            )
            if var_type == "cons":
                # triangle_lower_bound = self.plot_figure.scatter(marker='triangle',
                #     x="iteration",
                #     y=y_name,
                #     source=self._lower_bounds_cons_source,
                #     size=15,
                #     color='red',
                #     # visible=True,
                # )
                # triangle_upper_bound = self.plot_figure.scatter(marker='inverted_triangle',
                #     x="iteration",
                #     y=y_name,
                #     source=self._upper_bounds_cons_source,
                #     size=15,
                #     color='red',
                #     # visible=True,
                # )
                
                
                arrow_lower_bound = self.plot_figure.image_url(url='urls', x='iteration', y=y_name, 
                                                               w=None, h=None, 
                                                               anchor="center", source=self._lower_bounds_cons_source)
                arrow_upper_bound = self.plot_figure.image_url(url='urls', x='iteration', y=y_name, 
                                                               w=None, h=None, 
                                                               anchor="center", source=self._upper_bounds_cons_source)

                # self._lines.append(triangle)


        if var_type == "desvars":
            line.y_range_name = f"extra_y_{varname}_min"
        elif var_type == "cons":
            line.y_range_name = f"extra_y_{varname}"
            # triangle_lower_bound.y_range_name = f"extra_y_{varname}"
            # triangle_upper_bound.y_range_name = f"extra_y_{varname}"
            arrow_lower_bound.y_range_name = f"extra_y_{varname}"
            arrow_upper_bound.y_range_name = f"extra_y_{varname}"
        self._lines.append(line)
        if not use_varea:  # hover tool does not work with Varea
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Iteration", "@iteration"),
                    (varname, "@{%s}" % varname + "{0.00}"),
                ],
                mode="vline",
                visible=visible,
            )
            self.plot_figure.add_tools(hover)

    def _make_axis(self, var_type, varname, plot_value, units):
        # Make axis for this variable on the right of the plot
        if var_type == "desvars":
            y_range_name = f"extra_y_{varname}_min"
        else:
            y_range_name = f"extra_y_{varname}"
        extra_y_axis = LinearAxis(
            y_range_name=y_range_name,
            axis_label=f"{varname} ({units})",
            axis_label_text_font_size="20px",
            visible=False,
        )
        self._axes.append(extra_y_axis)
        self.plot_figure.add_layout(extra_y_axis, "right")
        self.plot_figure.extra_y_ranges[y_range_name] = Range1d(
            plot_value - 1, plot_value + 1
        )

    def _setup_figure(self):
        # Make the figure and all the settings for it
        self.plot_figure = figure(
            tools=[
                PanTool(),
                WheelZoomTool(),
                ZoomInTool(),
                ZoomOutTool(),
                BoxZoomTool(),
                ResetTool(),
                SaveTool(),
            ],
            width_policy="max",
            height_policy="max",
            sizing_mode="stretch_both",
            title=f"Optimization Progress Plot for: {self._case_recorder_filename}",
            active_drag=None,
            active_scroll="auto",
            active_tap=None,
            output_backend="webgl",
        )
        self.plot_figure.x_range.start = 1
        self.plot_figure.x_range.follow = "start"
        self.plot_figure.title.text_font_size = "14px"
        self.plot_figure.title.text_color = "black"
        self.plot_figure.title.text_font = "arial"
        self.plot_figure.title.align = "left"
        self.plot_figure.title.standoff = 40  # Adds 40 pixels of space below the title

        self.plot_figure.xaxis.axis_label = "Driver iterations"
        self.plot_figure.xaxis.minor_tick_line_color = None
        self.plot_figure.xaxis.ticker = BasicTicker(desired_num_ticks = 10, min_interval=1)

        self.plot_figure.axis.axis_label_text_font_style = "bold"
        self.plot_figure.axis.axis_label_text_font_size = "20pt"


def realtime_opt_plot(case_recorder_filename, callback_period, pid_of_calling_script, show):
    """
    Visualize the objectives, desvars, and constraints during an optimization process.

    Parameters
    ----------
    case_recorder_filename : MetaModelStructuredComp or MetaModelUnStructuredComp
        The metamodel component.
    callback_period : float
        The time period between when the application calls the update method.
    pid_of_calling_script : int
        The process id of the calling optimization script, if called this way.
    show : bool
        If true, launch the browser display of the plot.
    """

    def _make_realtime_opt_plot_doc(doc):
        _RealTimeOptPlot(
            case_recorder_filename,
            callback_period,
            doc=doc,
            pid_of_calling_script=pid_of_calling_script,
        )

    _port_number = _get_free_port()

    try:
 
        from tornado.web import StaticFileHandler
        
 
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_opt_plot_doc))},
            port=_port_number,
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
            
            
                    extra_patterns=[
            ('/static/(.*)', StaticFileHandler, {'path': os.path.normpath(os.path.dirname(__file__) + '/static/')}),
        ],

            
            
        )
        
        # print(os.path.normpath(os.path.dirname(os.path.dirname(__file__)) + '/static/'))
        print(os.path.normpath(os.path.dirname(__file__) + '/static/'))
        
        server.start()
        
        
        import bokeh
        print(f"{os.path.dirname(__file__)=}")
        #######. os.path.dirname(__file__)='/Users/hschilli/Documents/OpenMDAO/dev/I3568-rtplot-ui-fixes/openmdao/visualization/realtime_opt_plot'
        
        
                #         handlers = [
                #     (
                #         self.prefix + r"/statics/(.*)",
                #         web.StaticFileHandler,
                #         {"path": os.path.join(os.path.dirname(__file__), "static")},
                #     )
                # ]

                # self.server._tornado.add_handlers(r".*", handlers)


        
        
        
        if show:
            server.io_loop.add_callback(server.show, "/")

        print(f"Real-time optimization plot server running on http://localhost:{_port_number}")
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(f"Real-time optimization plot server stopped due to keyboard interrupt: {e}")
    except Exception as e:
        print(f"Error starting real-time optimization plot server: {e}")
    finally:
        print("Stopping real-time optimization plot server")
        if "server" in globals():
            server.stop()
