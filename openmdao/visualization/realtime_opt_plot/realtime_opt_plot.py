""" A real-time plot monitoring the optimization process as an OpenMDAO script runs"""

import errno
import os
import sys
from collections import defaultdict
import sqlite3

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
    SingleIntervalTicker
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

import numpy as np

from openmdao.recorders.sqlite_reader import SqliteCaseReader

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000

# Constants
_time_between_callbacks_in_ms = 1000 # the time between calls to the udpate method
# Number of milliseconds for unused session lifetime
_unused_session_lifetime_milliseconds = 1000 * 60 * 10
_obj_color = "black"  # color of the plot line for the objective function
_non_active_plot_color = "black" # color of the buttons for variables not being shown
_varea_alpha = 0.3 # how transparent is the area part of the plot for desvars that are vectors
# the CSS for the toggle buttons to let user choose what variables to plot
toggle_styles = """
            font-size: 22px; 
            box-shadow: 
                0 4px 6px rgba(0, 0, 0, 0.1),    /* Distant shadow */
                0 1px 3px rgba(0, 0, 0, 0.08),   /* Close shadow */
                inset 0 2px 2px rgba(255, 255, 255, 0.2);  /* Top inner highlight */
"""

# colors used for the plot lines and associated buttons and axes labels
# start with color-blind friendly colors and then use others if needed
colorPalette = Colorblind[8] + Category20[20]

# This is the JavaScript code that gets run when a user clicks on 
#   one of the buttons that change what variables are plotted
callback_code=f"""
// The ColorManager provides color from a palette for plotting lines. When the 
//   user turns off the plotting of a line, the color is returned to the manager
//   for use with a different variable plot
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
            const newColor = availableColor || this.palette[this.usedColors.size % this.palette.length];
            
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
            {toggle_styles}
        }}
    `];
// if turning off a variable, return the color to the pool
}} else {{
    window.colorManager.releaseColor(variable_name);
    toggle.stylesheets = [`
        .bk-btn {{
            {toggle_styles}
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
    Set up the realtime plot subparser for the 'openmdao realtime_opt_plot' command.

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

    parser.add_argument('--pid', type=int, default=None, help='Process ID of calling optimization script')
    parser.add_argument('--no-display', action='store_false', dest='show',
                        help="do not launch browser showing plot. Primarily used for testing")


def _realtime_opt_plot_cmd(options, user_args):
    """
    Run the realtime_opt_plot command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    realtime_opt_plot(options.case_recorder_filename, _time_between_callbacks_in_ms, options.pid, options.show)

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

    Need to handle variables that are vectors.

    Parameters
    ----------
    value_from_recorder : str
        Name of the variable.
    var_type : str
        String indicating of 'objs', 'desvars' or 'cons'.

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
        return value_from_recorder.item() # get as scalar
    else:  # for desvars, just L2 norm
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
        styles={"font-size": "14"},
    ) 
    return header_text_div


class _CaseTracker:
    """
    A class that is used to get information from a case recorder 
    that is needed by the code in this file.
    """
    def __init__(self, case_recorder_filename):
        self._case_recorder_filename = case_recorder_filename
        self._cr = None
        self.source = None
        self._num_iterations_read = 0
        self._initial_cr_with_one_case = None
        self._next_id_to_read = 1

    def _get_case_by_counter(self, counter):
        with sqlite3.connect(self._case_recorder_filename) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("SELECT * FROM driver_iterations WHERE "
                        "counter=:counter",
                        {"counter": counter})
            row = cur.fetchone()
        con.close()

        if row:
            from openmdao.recorders.case import Case
            if self._cr is None:
                self._cr = SqliteCaseReader(self._case_recorder_filename)
            var_info = self._cr.problem_metadata['variables']
            case = Case('driver', row, self._cr._prom2abs, self._cr._abs2prom, self._cr._abs2meta,
                        self._cr._conns, var_info, self._cr._format_version)

            return case
        else:
            return None

    def get_new_case(self):
        driver_case = self._get_case_by_counter(self._next_id_to_read)
        if driver_case is None:
            return None
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

        self._next_id_to_read += 1
        return new_data

    def get_obj_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        obj_vars = driver_case.get_objectives()
        return obj_vars.keys()

    def get_desvar_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        design_vars = driver_case.get_design_vars()
        return design_vars.keys()

    def get_cons_names(self):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        case_ids = self._initial_cr_with_one_case.list_cases("driver", out_stream=None)
        driver_case = self._initial_cr_with_one_case.get_case(case_ids[0])
        cons = driver_case.get_constraints()
        return cons.keys()

    def get_units(self, name):
        if self._initial_cr_with_one_case is None:
            cr = SqliteCaseReader(self._case_recorder_filename)
            case_ids = cr.list_cases("driver", out_stream=None)
            if len(case_ids) > 0:
                self._initial_cr_with_one_case = cr
            else:
                return None
        driver_case = self._initial_cr_with_one_case.get_case(0)

        try:
            units = driver_case._get_units(name)
        except RuntimeError as err:
            if str(err).startswith("Can't get units for the promoted name"):
                return "Ambiguous"
            raise
        except KeyError as err:
            return "Unavailable"

        if units is None:
            units = "Unitless"
        return units

class RealTimeOptPlot(object):
    def __init__(self, case_recorder_filename, callback_period, doc, pid_of_calling_script):

        self._case_recorder_filename = case_recorder_filename
        self._pid_of_calling_script = pid_of_calling_script

        self._source = None
        self._lines = []
        self._toggles = []
        self._column_items = []
        self._axes = []
        self._labels_updated_with_units = False
        self._case_tracker = _CaseTracker(case_recorder_filename)
        self._doc = doc
        self._update_callback = None
        self._source_stream_dict = None

        self._setup_figure()

        # used to keep track of the y min and max of the data so that 
        #    the axes ranges can be adjusted as data comes in
        self.y_min = defaultdict(lambda: float("inf") )
        self.y_max = defaultdict(lambda: float("-inf"))

        def update():

            new_data = self._case_tracker.get_new_case()
            if new_data is None:
                if self._pid_of_calling_script is None or not _is_process_running(self._pid_of_calling_script):
                    # Just keep sending the last data point
                    # This is a hack to force the plot to re-draw
                    # Otherwise if the user clicks on the variable buttons, the
                    #   lines will not change color because of the hack done to get
                    #   get around the bug in setting the line color from JavaScript
                    self._source.stream(self._source_stream_dict)
                return

            # See if source is defined yet. If not, see if we have any data
            #   in the case file yet. If there is data, create the
            #   source object and add the lines to the figure
            # if source not setup yet, need to do that to setup streaming
            if self._source is None:
                self.setup_data_source()
                # index of lines across all variables: obj, desvars, cons
                i_line = 0
                
                # Objective
                obj_names = self._case_tracker.get_obj_names()
                if len(obj_names) != 1:
                    raise ValueError(
                        f"Plot assumes there is on objective but {len(obj_names)} found"
                    )
                
                # Create CustomJS callback for toggle buttons
                legend_item_callback = CustomJS(
                    args=dict(lines=self._lines, axes=self._axes, toggles=self._toggles, colorPalette=colorPalette, plot=self.p,                    
                    ),
                    code=callback_code,
                )

                # objs
                obj_label = _make_header_text_for_variable_chooser("OBJECTIVE")
                self._column_items.append(obj_label)

                for i, obj_name in enumerate(obj_names):
                    units = self._case_tracker.get_units(obj_name)
                    self.p.yaxis.axis_label = f"{obj_name} ({units})"
                    self._make_legend_item(f"{obj_name} ({units})", _obj_color, True, legend_item_callback)
                    self._make_line_and_hover_tool("objs", obj_name, False, _obj_color,"solid", True)
                    value = new_data["objs"][obj_name]
                    float_value = _get_value_for_plotting(value, "objs")
                    self.p.y_range = Range1d(float_value - 1, float_value + 1)

                # desvars
                desvars_label = _make_header_text_for_variable_chooser("DESIGN VARS")
                self._column_items.append(desvars_label)
                desvar_names = self._case_tracker.get_desvar_names()
                for i, desvar_name in enumerate(desvar_names):
                    units = self._case_tracker.get_units(desvar_name)
                    self._make_legend_item(f"{desvar_name} ({units})", "black", False, legend_item_callback)
                    value = new_data["desvars"][desvar_name]
                    use_varea = value.size > 1
                    self._make_line_and_hover_tool("desvars", desvar_name, use_varea, _non_active_plot_color, "solid", False)

                    float_value = _get_value_for_plotting(value, "desvars")
                    self._make_axis("desvars", desvar_name, float_value, units)
                    i_line += 1

                # cons
                cons_label = _make_header_text_for_variable_chooser("CONSTRAINTS")
                self._column_items.append(cons_label)
                cons_names = self._case_tracker.get_cons_names()
                for i, cons_name in enumerate(cons_names):
                    units = self._case_tracker.get_units(cons_name)
                    self._make_legend_item(f"{cons_name} ({units})", "black", False, legend_item_callback)
                    self._make_line_and_hover_tool("cons", cons_name, False, _non_active_plot_color, "dashed", False)
                    value = new_data["cons"][cons_name]
                    float_value = _get_value_for_plotting(value, "cons")
                    self._make_axis("cons", cons_name, float_value, units)
                    i_line += 1

                # Create a column of toggles with scrolling
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
                    # print("shutting down optimization plot server")
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

                graph = Row(self.p, scroll_box, sizing_mode="stretch_both")
                doc.add_root(graph)
                # end of self._source is None - plottng is setup

            counter = new_data["counter"]

            self._source_stream_dict = {"iteration": [counter]}

            iline = 0
            for obj_name, obj_value in new_data["objs"].items():
                float_obj_value = _get_value_for_plotting(obj_value, "objs")
                self._source_stream_dict[obj_name] = [float_obj_value]
                min_max_changed = _update_y_min_max(obj_name, float_obj_value, self.y_min, self.y_max)
                if min_max_changed:
                    self.p.y_range.start = self.y_min[obj_name]
                    self.p.y_range.end = self.y_max[obj_name]
                iline += 1

            for desvar_name, desvar_value in new_data["desvars"].items():
                if not self._labels_updated_with_units and desvar_value.size > 1:
                    units = self._case_tracker.get_units(desvar_name)
                    self._toggles[iline].label = f"{desvar_name} ({units}) {desvar_value.shape}"
                min_max_changed = False
                min_max_changed = min_max_changed or _update_y_min_max(desvar_name, np.min(desvar_value), self.y_min, self.y_max)
                min_max_changed = min_max_changed or _update_y_min_max(desvar_name, np.max(desvar_value), self.y_min, self.y_max)
                if min_max_changed:
                    range = Range1d(
                        self.y_min[desvar_name], self.y_max[desvar_name]
                    )
                    self.p.extra_y_ranges[f"extra_y_{desvar_name}_min"] = range

                self._source_stream_dict[f"{desvar_name}_min"] = [np.min(desvar_value)]
                self._source_stream_dict[f"{desvar_name}_max"] = [np.max(desvar_value)]
                iline += 1

            for cons_name, cons_value in new_data["cons"].items():
                float_cons_value = _get_value_for_plotting(cons_value, "cons")

                if not self._labels_updated_with_units and cons_value.size > 1:
                    units = self._case_tracker.get_units(cons_name)
                    self._toggles[iline].label = f"{cons_name} ({units}) {cons_value.shape}"

                self._source_stream_dict[cons_name] = [float_cons_value]
                min_max_changed = _update_y_min_max(cons_name, float_cons_value, self.y_min, self.y_max)
                if min_max_changed:
                    range = Range1d(
                        self.y_min[cons_name], self.y_max[cons_name]
                    )
                    self.p.extra_y_ranges[f"extra_y_{cons_name}"] = range
                iline += 1
            self._source.stream(self._source_stream_dict)
            self._labels_updated_with_units = True 
            # end of update method

        self._update_callback = doc.add_periodic_callback(update, callback_period)
        doc.title = "OpenMDAO Optimization"

    def setup_data_source(self):
        self._source_dict = {"iteration": []}

        # Obj
        obj_names = self._case_tracker.get_obj_names()
        for obj_name in obj_names:
            self._source_dict[obj_name] = []

        # Desvars
        desvar_names = self._case_tracker.get_desvar_names()
        for desvar_name in desvar_names:
            self._source_dict[f"{desvar_name}_min"] = []
            self._source_dict[f"{desvar_name}_max"] = []

        # Cons
        con_names = self._case_tracker.get_cons_names()
        for con_name in con_names:
            self._source_dict[con_name] = []

        self._source = ColumnDataSource(self._source_dict)

    def _make_legend_item(self, varname, color, active, callback):
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
                    {toggle_styles}
                }}
                .bk-btn.bk-active {{
                    background-color: rgb(from #000000 R G B / 0.3);
                    {toggle_styles}
                }}
            """
    ]
        return toggle

    def _make_line_and_hover_tool(self,var_type,varname, use_varea, color,line_dash,visible):
        if use_varea:
            line = self.p.varea(
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
            line = self.p.line(
                x="iteration",
                y=y_name,
                line_width=3,
                line_dash=line_dash,
                source=self._source,
                color=color,
                visible=visible,
            )

        if var_type == "desvars":
            line.y_range_name=f"extra_y_{varname}_min"
        elif var_type == "cons":
            line.y_range_name=f"extra_y_{varname}"
        self._lines.append(line)
        if not use_varea:
            hover = HoverTool(
                renderers=[line],
                tooltips=[
                    ("Iteration", "@iteration"),
                    (varname, "@{%s}" % varname + "{0.00}"),
                ],
                mode="vline",
                visible=visible,
            )
            self.p.add_tools(hover)


    def _make_axis(self, var_type, varname, plot_value, units):
        # Make axis for this variable on the right
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
        self.p.add_layout(extra_y_axis, "right")
        self.p.extra_y_ranges[y_range_name] = Range1d(
            plot_value - 1, plot_value + 1
        )


    def _setup_figure(self):
        # Make the figure and all the settings for it
        self.p = figure(
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
            title=f"Real-time Optimization Progress Plot for: {self._case_recorder_filename}",
            active_drag=None,
            active_scroll="auto",
            active_tap=None,
            output_backend="webgl",
        )
        self.p.x_range.follow = "start"
        # self.p.title.text_font_size = "25px"
        self.p.title.text_color = "black"
        self.p.title.text_font = "arial"
        self.p.title.align = "left"
        self.p.title.standoff = 40  # Adds 40 pixels of space below the title

        self.p.xaxis.axis_label = "Driver iterations"
        self.p.xaxis.minor_tick_line_color = None
        self.p.xaxis.ticker = SingleIntervalTicker(interval=1)

        self.p.axis.axis_label_text_font_style = "bold"
        self.p.axis.axis_label_text_font_size = "20pt"


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
    show : boolean
        If true, launch the browser display of the plot.
    """

    def _make_realtime_opt_plot_doc(doc):
        RealTimeOptPlot(case_recorder_filename, callback_period, doc=doc, pid_of_calling_script=pid_of_calling_script)

    _port_number = get_free_port()

    try:
        server = Server(
            {"/": Application(FunctionHandler(_make_realtime_opt_plot_doc))},
            port=_port_number,
            unused_session_lifetime_milliseconds=_unused_session_lifetime_milliseconds,
        )
        server.start()
        if show:
            server.io_loop.add_callback(server.show, "/")

        print(f"Real-time optimization plot server running on http://localhost:{_port_number}")
        server.io_loop.start()
    except KeyboardInterrupt as e:
        print(f"Real-time optimization plot server stopped due to keyboard interrupt")
    except Exception as e:
        print(f"Error starting real-time optimization plot server: {e}")
    finally:
        print("Stopping real-time optimization plot server")
        if "server" in globals():
            server.stop()
