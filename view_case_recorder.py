import argparse
import logging
import json
from io import StringIO
import pathlib
import importlib

import openmdao.api as om

import panel as pn
from panel.theme import DefaultTheme

import pandas as pd

from bokeh.models.widgets.tables import ScientificFormatter
from bokeh.models.formatters import BasicTickFormatter

scientific_formatter = ScientificFormatter(precision=2, power_limit_high = 3, power_limit_low = -3, nan_format="None")
# scientific_formatter = BasicTickFormatter(precision=2, power_limit_high = 3, power_limit_low = -3)
logger = logging.getLogger('panel.myapp')

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        return 5000

parser = argparse.ArgumentParser()
parser.add_argument(
    "case_recorder_name",
    type=str,
    help="Name of case recorder file.",
)
args = parser.parse_args()

case_recorder_name = args.case_recorder_name
cr = om.CaseReader(case_recorder_name)

tabs_list = []

# Summary tab
df = pd.DataFrame(columns=["Name", "Value"])
df.loc[0] = ["openmdao_version", cr.openmdao_version]
df.loc[1] = ["format_version", cr._format_version]
df.loc[2] = ["sources", cr.list_sources()]
df.loc[3] = ["driver name", cr.problem_metadata['driver']['name']]
if 'optimizer' in cr.problem_metadata['driver']['options']:
    df.loc[4] = ["driver optimizer", cr.problem_metadata['driver']['options']['optimizer']]
summary_table_pane = pn.widgets.Tabulator(df, show_index=False, selectable=False,
                                    sortable=False,
                                    disabled=True,  # disables editing of the table
                                    )
summary_text_pane = pn.pane.Str(f"This page is a very high level summary of what is in the case recorder file.")
summary_pane = pn.Column(summary_text_pane, summary_table_pane)
tabs_list.append(("Summary", summary_pane))

# Cases Summary 
data = []
cases = cr.get_cases()
df = pd.DataFrame(columns=["Name", "Source", "Num Inputs", "Num Outputs", "Num Residuals",
                           "Derivatives", "Abs Err", "Rel Err"])
for i, case in enumerate(cases):
    df.loc[i] = [case.name, case.source, len(case.inputs) if case.inputs else 0, len(case.outputs) if case.outputs else 0,
                 len(case.residuals) if case.residuals else 0, "Yes" if case.derivatives else "No", case.abs_err, case.rel_err]

cases_table_pane = pn.widgets.Tabulator(df, 
                                  show_index=False, selectable=False,
                                    sortable=True,
                                    disabled=True,
                                  )
cases_text_pane = pn.pane.Str(f"This page lists all the Cases found in the case recorder file.")
cases_pane = pn.Column(cases_text_pane, cases_table_pane)
tabs_list.append(("Cases", cases_pane))

# Case 0 inputs, outputs and residuals if available
case0 = cases[0]
case0_summary_text_pane = pn.pane.Str(f"There are {len(case0.inputs)if case0.inputs else 0} inputs, {len(case0.outputs) if case0.outputs else 0} outputs, and {len(case0.residuals) if case0.residuals else 0} residuals in Case 0")
case0_inputs_pane = pn.Column(pn.pane.Markdown("# Inputs"))
if case0.inputs:
    df_case0_inputs = pd.DataFrame(list(case0.inputs.keys()), columns=[f"{len(case0.inputs.keys())} Inputs"])
    case0_inputs_table_pane = pn.widgets.Tabulator(df_case0_inputs,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            # layout='fit_data_table',
                                            pagination=None,
                                            selectable=False,
                                            sortable=True,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
                                            )
else:
    case0_inputs_table_pane = pn.pane.Markdown(f"### There are no inputs in Case 0")
case0_inputs_pane.append(case0_inputs_table_pane)

case0_outputs_pane = pn.Column(pn.pane.Markdown("# Outputs"))
if case0.outputs:
    df_case0_outputs = pd.DataFrame(list(case0.outputs.keys()), columns=[f"{len(case0.outputs.keys())} Outputs"])
    case0_outputs_table_pane = pn.widgets.Tabulator(df_case0_outputs,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            # layout='fit_data_table',
                                            pagination=None,
                                            selectable=False,
                                            sortable=True,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
                                            )
else:
    case0_outputs_table_pane = pn.pane.Markdown(f"### There are no outputs in Case 0")
case0_outputs_pane.append(case0_outputs_table_pane)

case0_residuals_pane = pn.Column(pn.pane.Markdown("# Residuals"))
if case0.residuals:
    df_case0_residuals = pd.DataFrame(list(case0.residuals.keys()), columns=[f"{len(case0.residuals.keys())} Residuals"])
    case0_residuals_table_pane = pn.widgets.Tabulator(df_case0_residuals,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            layout='fit_data_fill',
                                            pagination=None,
                                            selectable=False,
                                            sortable=True,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
                                            )
else:
    case0_residuals_table_pane = pn.pane.Markdown(f"### There are no residuals in Case 0")
case0_residuals_pane.append(case0_residuals_table_pane)

case0_variables_pane = pn.Column(
    case0_summary_text_pane,
    pn.Row(
        case0_inputs_pane,
        case0_outputs_pane,
        case0_residuals_pane,
    sizing_mode='stretch_width',
    styles={
        'display': 'flex',
        'justify-content': 'space-between',
        'width': '100%'
    }
        )
)
tabs_list.append(("Case 0 variables", case0_variables_pane))


# variables pane
variables_text_pane = pn.pane.Str(f"Here is the metadata for the {len(cr.problem_metadata['variables'])} variables in the problem")

tabulator_editors = {
    'flat_indices': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'cache_linear_solution': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'distributed': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
}
metadata_names = set()
for name, properties in cr.problem_metadata['variables'].items():
    if isinstance(properties, dict):
        metadata_names.update(list(properties.keys()))
    else:
        logger.info(f"problem_metadata['variables'] has unexpected value of {name} : {properties}")
metadata_names = ['name', 'orig', 'alias', 'type', 'size', 'units','global_size', 'lower', 'upper',
                  'parallel_deriv_color', 'cache_linear_solution', 'distributed',
                  'adder', 'total_adder', 'scaler', 'total_scaler', 'ref', 'ref0',
                   'linear', 'indices', 
                   'source',  'flat_indices',    
                   'equals', 'parent',  
                    ]       

df_variables = pd.DataFrame(columns=metadata_names)
i = 0
for name, properties in cr.problem_metadata['variables'].items():
    if isinstance(properties, dict):
        row = []
        for metadata_name in metadata_names:
            if metadata_name in properties:
                if metadata_name in ["orig"]:  # need to handle this special way
                    value = ''
                    for item_i, item in enumerate(properties[metadata_name]):
                        value += item if item is not None else "None"
                        if item_i < len(properties[metadata_name]) - 1:
                            value += ", "
                else:
                    value = properties[metadata_name] if properties[metadata_name] is not None else "None"
            else:
                value = "None"
            row.append(value)
        df_variables.loc[i] = row
        i = i + 1

pd.set_option('display.max_rows', 1000)
# print(df_variables['total_adder'])

# print(df_variables['total_adder'].apply(type))

print("metadata variables")
print(df_variables['name'])

for var_name in ['lower', 'upper', 'equals', 'linear', 'scaler', 'total_scaler', 'adder', 'total_adder', 'ref', 'ref0']:
    df_variables[var_name] = df_variables[var_name].apply(lambda x: f"{x:.3g}" if isinstance(x,float) else x)


variables_names = df_variables['name'].tolist()

variables_table_pane = pn.widgets.Tabulator(df_variables,
                                            disabled=True,  # disables editing of the table
                                            sortable=False,
                                            header_filters=True,
                                            editors=tabulator_editors,
                                            layout='fit_data_fill',
                                            selectable=False,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            # formatters={
                                            #     # "lower": scientific_formatter,
                                            #     # "upper": scientific_formatter,
                                            #     # "equals": scientific_formatter,
                                            #     # "linear": scientific_formatter,
                                            #     # "scaler": scientific_formatter,
                                            #     # "total_scaler": scientific_formatter,
                                            #     # "adder": scientific_formatter,
                                            #     # "total_adder": scientific_formatter,
                                            #     "ref": scientific_formatter,
                                            #     "re0": scientific_formatter,
                                            #     },
                                            # theme='bootstrap5', 
                                            # stylesheets=[":host .tabulator {font-size: 10px;}"],
                                            )

variables_pane = pn.Column(
    variables_text_pane,
    variables_table_pane
)
tabs_list.append(("Metadata variables", variables_pane))

# design_vars pane
df_metadata_desvars = pd.DataFrame(
    columns=["name", "source", "units", "size", "global_size", "lower", "upper", "distributed",
                                            "parallel_deriv_color", "adder", "total_adder", "scaler", "total_scaler", 
                                            "ref", "ref0", "indices", "flat_indices", "cache_linear_solution"]
    )

def clean_nones(input_list):
    # if this is not done, then get warnings from Pandas about empty columns because of the None values
    return [item if item is not None else "None" for item in input_list]

for i, (var_name, meta) in enumerate(cr.problem_metadata['design_vars'].items()):
    df_metadata_desvars.loc[i] = clean_nones([meta["name"], meta["source"], meta["units"], meta["size"], meta["global_size"], 
                 meta["lower"], meta["upper"], meta["distributed"], meta["parallel_deriv_color"],
                 meta["adder"], meta["total_adder"], meta["scaler"], meta["total_scaler"], meta["ref"], meta["ref0"],
                 str(meta["indices"]), meta["flat_indices"], meta["cache_linear_solution"]])

tabulator_editors = {
    'flat_indices': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'cache_linear_solution': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'distributed': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
}


print("desvars variables")
print(df_metadata_desvars['name'])
desvars_names = df_metadata_desvars['name'].tolist()

for var_name in ['lower', 'upper', 'scaler', 'total_scaler', 'adder', 'total_adder', 'ref', 'ref0']:
    df_metadata_desvars[var_name] = df_metadata_desvars[var_name].apply(lambda x: f"{x:.3g}" if isinstance(x,float) else x)
# df_metadata_desvars['total_adder'] = df_metadata_desvars['total_adder'].apply(lambda x: f"{x:.3g}" if isinstance(x,float) else x)

metadata_desvars_table_pane = pn.widgets.Tabulator(df_metadata_desvars, 
                                            header_filters=True,
                                            editors=tabulator_editors,
                                  show_index=False, selectable=False,
                                    sortable=True,
                                    disabled=True,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            formatters={
                                                # "lower": scientific_formatter,
                                                # "upper": scientific_formatter,
                                                # "adder": scientific_formatter,
                                                # # "Total Adder": scientific_formatter,
                                                # "scaler": scientific_formatter,
                                                # "total_scalar": scientific_formatter,
                                                # "ref": scientific_formatter,
                                                # "ref0": scientific_formatter,
                                                },
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
                                  )
# metadata_desvars_table_pane.add_filter([True, False], 'flat_indices')


design_vars_pane = pn.Column(
    pn.pane.Str(f"Here is the metadata for the {len(cr.problem_metadata['design_vars'])} design variables in the problem")
    metadata_desvars_table_pane
)
tabs_list.append(("Metadata design_vars", design_vars_pane))

# responses pane
tabulator_editors = {
    'flat_indices': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'cache_linear_solution': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
    'distributed': {'type': 'tickCross', 'tristate': True, 'indeterminateValue': None},
}
json_string = json.dumps(cr.problem_metadata['responses'])
df_responses = pd.read_json(StringIO(json_string), orient="index") 

# df_responses['total_adder'] = df_responses['total_adder'].apply(lambda x: f"{x:.3g}" if isinstance(x,float) else x)
for var_name in ['lower', 'upper', 'equals', 'linear', 'scaler', 'total_scaler', 'adder', 'total_adder', 'ref', 'ref0']:
    df_responses[var_name] = df_responses[var_name].apply(lambda x: f"{x:.3g}" if isinstance(x,float) else x)


print("responses variables")
print(df_responses['name'])
print(df_responses['name'].to_string(index=False))
response_names = df_responses['name'].tolist()


desvars_plus_response_names = set(desvars_names) | set(response_names)

missing_vars = set(variables_names) - desvars_plus_response_names

print(f"{missing_vars=}")

responses_table_pane = pn.widgets.Tabulator(df_responses,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            editors=tabulator_editors,
                                            layout='fit_data_fill',
                                            sortable=True,
                                            selectable=False,
                                            sizing_mode='stretch_height', # this works great!
                                            titles={'index': 'abs_name'},
                                            # formatters={
                                            #     "lower": scientific_formatter,
                                            #     "upper": scientific_formatter,
                                            #     "equals": scientific_formatter,
                                            #     "linear": scientific_formatter,
                                            #     "scaler": scientific_formatter,
                                            #     "total_scaler": scientific_formatter,
                                            #     "adder": scientific_formatter,
                                            #     "total_adder": scientific_formatter,
                                            #     "ref": scientific_formatter,
                                            #     "re0": scientific_formatter,
                                            #     },
                                            theme='bootstrap5', 
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
                                            )

responses_pane = pn.Column(
    pn.pane.Str(f"Here is the metadata for the {len(cr.problem_metadata['responses'])} responses in the problem"),
    responses_table_pane,
)
tabs_list.append(("Metadata responses", responses_pane))

# Execution order pane
if 'execution_order' in cr.problem_metadata['variables'].keys():
    # A hack to get around the fact that Panel doesn't do column sizing
    # correctly. It only seems to look at the first 40 lines to see what the max size is.
    # But making the column heading the length of the longest string is a close 
    # approximation to what is needed
    execution_order = cr.problem_metadata['variables']['execution_order']
    max_string_length = len(max(execution_order, key=len))
    padded_column_name = "Systems".center(max_string_length, '_')
    df_execution_order = pd.DataFrame(execution_order, columns=[padded_column_name])
    execution_order_table_pane = pn.widgets.Tabulator(df_execution_order,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            layout='fit_data',
                                            pagination=None,
                                            selectable=False,
                                            show_index=False,
                                            sizing_mode='stretch_both', # this works great!
                                            theme='bootstrap5', 
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
                                            )
    execution_order_pane = pn.Column(
        pn.pane.Str("This page lists the execution order of the systems"),
        execution_order_table_pane
    )
    tabs_list.append(("Execution Order", execution_order_pane))

# abs2prom pane
abs2prom_text_pane = pn.pane.Str(f"There are {len(cr.problem_metadata['abs2prom']['input'])} inputs and {len(cr.problem_metadata['abs2prom']['output'])} outputs in the metadata abs2prom")
df_abs2prom_inputs = pd.DataFrame(list(cr.problem_metadata['abs2prom']['input'].items()), columns=["Absolute Name", "Promoted Name"])
abs2prom_inputs_table_pane = pn.widgets.Tabulator(df_abs2prom_inputs,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            layout='fit_columns',
                                            sortable=True,
                                            pagination=None,
                                            selectable=False,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            width=800,
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
)
df_abs2prom_outputs = pd.DataFrame(list(cr.problem_metadata['abs2prom']['output'].items()), columns=["Absolute Name", "Promoted Name"])
abs2prom_outputs_table_pane = pn.widgets.Tabulator(df_abs2prom_outputs,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            layout='fit_columns',
                                            sortable=True,
                                            pagination=None,
                                            selectable=False,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            width=800,
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
)

abs2prom_pane = pn.Column(
    abs2prom_text_pane,
    pn.pane.Markdown(f"# Inputs"),
    abs2prom_inputs_table_pane,
    pn.pane.Markdown(f"# Outputs"),
    abs2prom_outputs_table_pane
)
tabs_list.append(("Metadata abs2prom", abs2prom_pane))

# connections_list pane
connections_text_pane = pn.pane.Str(f"There are {len(cr.problem_metadata['connections_list'])} items in the metadata connections_list")
df_connections_list = pd.DataFrame(cr.problem_metadata['connections_list'], columns=["src", "tgt"])
df_connections_list.columns = ["Source", "Target"]
df_connections_table_pane = pn.widgets.Tabulator(df_connections_list,
                                            disabled=True,  # disables editing of the table
                                            header_filters=True,
                                            layout='fit_data_fill',
                                            pagination=None,
                                            selectable=False,
                                            sortable=True,
                                            show_index=False,
                                            sizing_mode='stretch_height', # this works great!
                                            theme='bootstrap5', 
                                            width=800,
                                            stylesheets=[":host .tabulator {font-size: 10px;}"],
)

connections_list_pane = pn.Column(
    connections_text_pane,
    df_connections_table_pane
)
tabs_list.append(("Metadata connections_list", connections_list_pane))


# declare_partials_list pane
declare_partials_list_text_pane = pn.pane.Str(f"There are {len(cr.problem_metadata['declare_partials_list'])} items in the metadata declare_partials_list")
df_declare_partials_list = pd.DataFrame([item.split(" > ") for item in cr.problem_metadata['declare_partials_list']], 
                                        columns=["Of the Function", "With Respect To"])
declare_partials_list_table_pane = pn.widgets.Tabulator(df_declare_partials_list,
                                        disabled=True,  # disables editing of the table
                                        header_filters=True,
                                        layout='fit_data_fill',
                                        pagination=None,
                                            sortable=True,
                                        selectable=False,
                                        show_index=False,
                                        sizing_mode='stretch_height', # this works great!
                                        theme='bootstrap5', 
                                        stylesheets=[":host .tabulator {font-size: 10px;}"],
                                        )

declare_partials_list_pane = pn.Column(
    declare_partials_list_text_pane,
    declare_partials_list_table_pane
)
tabs_list.append(("Metadata declare_partials_list", declare_partials_list_pane))

tabs = pn.Tabs(*tabs_list, stylesheets=["assets/view_case_recorder_styles.css"])
template = pn.template.FastListTemplate(
    title=f"Dashboard for case recorder file '{case_recorder_name}'",
    main=[tabs],
    accent_base_color="black",
    header_background="rgb(0, 212, 169)",
    # header=header,
    background_color="white",
    theme=DefaultTheme,
    theme_toggle=False,
    main_layout=None,
    # css_files=["assets/aviary_styles.css"],
)

# NOT WORKING YET
# save_dashboard_button = pn.widgets.Button(
#     name="Save Dashboard",
#     width_policy="min",
#     # css_classes=["save-button"],
#     button_type="success",
#     button_style="solid",
#     # stylesheets=["assets/aviary_styles.css"],
# )
# header = pn.Row(save_dashboard_button, pn.HSpacer(), pn.HSpacer(), pn.HSpacer())
# def save_dashboard(event):
#     case_recorder_dashboard_filename = 'case_recorder_dashboard.html'
#     print(f"Saving dashboard files to {case_recorder_dashboard_filename}")
#     template.save(case_recorder_dashboard_filename)
# save_dashboard_button.on_click(save_dashboard)

show = True
threaded = False

port = 0
home_dir = "."

assets_dir = pathlib.Path(
    importlib.util.find_spec("openmdao").origin
).parent.joinpath("recorders/assets/")

if port == 0:
    port = get_free_port()
server = pn.serve(
    template,
    port=port,
    address="localhost",
    websocket_origin=f"localhost:{port}",
    show=show,
    threaded=threaded,
    static_dirs={
        "home": home_dir,
        "assets": assets_dir,
    },
)
server.stop()
