"""
Code for generating a dashboard for case recorder files.
"""
import argparse
import pathlib
import importlib

import openmdao.api as om
from openmdao.utils.om_warnings import CaseRecorderWarning, issue_warning

try:
    import panel as pn
except ModuleNotFoundError:
    pn = None

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

try:
    from openmdao.utils.gui_testing_utils import get_free_port
except ImportError:
    # If get_free_port is unavailable, the default port will be used
    def get_free_port():
        """
        Return the default port number.

        Returns
        -------
        int
            Default port number.
        """
        return 5000


def _view_cases_setup_parser(parser):
    """
    Set up the subparser for the 'openmdao view_cases' command.

    Parameters
    ----------
    parser : argparse subparser
        The parser we're adding options to.
    """
    parser.add_argument(
        "case_recorder_file",
        type=str,
        help="Name of the case recorder file to view",
    )


def _view_cases_cmd(options, user_args):
    """
    Run the view_cases command.

    Parameters
    ----------
    options : argparse Namespace
        Command line options.
    user_args : list of str
        Args to be passed to the user script.
    """
    if not options.case_recorder_file:
        raise argparse.ArgumentError("case_recorder_file argument missing")

    view_cases(options.case_recorder_file)


# boolean value user interface filter elements. Lets user filter based on true, false, or either
tabulator_editors = {
    "flat_indices": {"type": "tickCross", "tristate": True, "indeterminateValue": None},
    "cache_linear_solution": {
        "type": "tickCross",
        "tristate": True,
        "indeterminateValue": None,
    },
    "distributed": {"type": "tickCross", "tristate": True, "indeterminateValue": None},
}

# default options used in all the Tabulators
tabulator_defaults = {
    'disabled': True,
    'header_filters': True,
    'pagination': None,
    'selectable': False,
    'editors': tabulator_editors,
    'sortable': True,
    'show_index': False,
    'sizing_mode': 'stretch_height',
    'theme': 'bootstrap5',
    'stylesheets': [":host .tabulator {font-size: 10px;}"]
}

# to have a standard order of the metadata across the table in the dashboard display
metadata_names = [
    "name",
    "orig",
    "alias",
    "type",
    "size",
    "units",
    "global_size",
    "lower",
    "upper",
    "parallel_deriv_color",
    "cache_linear_solution",
    "distributed",
    "adder",
    "total_adder",
    "scaler",
    "total_scaler",
    "ref",
    "ref0",
    "linear",
    "indices",
    "source",
    "flat_indices",
    "equals",
    "parent",
]


def variable_metadata_to_data_frame(variables_metadata):
    """
    Prep the metadata for DataFrame displayed by Panel's Tabulator widget.

    Parameters
    ----------
    variables_metadata : dict of dicts
        The metadata of variables.

    Returns
    -------
    DataFrame
        DataFrame containing the information about the variable metadata as a table.
    """
    df = pd.DataFrame(columns=metadata_names)
    i = 0
    for var_name, metadata in variables_metadata.items():
        if isinstance(metadata, dict):
            row = []
            for metadata_name in metadata_names:
                if metadata_name not in metadata:
                    value = "N/A"
                else:
                    value = metadata[metadata_name]
                    if value is None:  # if this is not done, then get warnings from
                        # Pandas about empty columns because of the None values
                        value = "None"
                    elif isinstance(value, float):  # use Pythons' formatting instead of Panel's
                        value = f"{value:.3g}"
                    elif isinstance(value, list):  # use Pythons' formatting instead of Panel's
                        # Check if all elements are floats
                        all_floats = all(isinstance(item, float) for item in value)
                        if all_floats:
                            value = f"min: {min(value)} max: {max(value)}"
                        else:
                            value = _elide_string(", ".join(str(x) for x in value), 25)
                row.append(value)
            df.loc[i] = row
            i = i + 1
    return df


def _elide_string(s, max_length):
    """
    Shorten a string by removing the middle part and replacing with ...

    Parameters
    ----------
    s : str
        The string to shorten.
    max_length : int
        Keep the string less than or equal to this length.

    Return
    ----------
    s_short : str
        A shortened version of the string.
    """
    if len(s) <= max_length:
        return s
    half_length = (max_length - 3) // 2
    return s[:half_length] + '...' + s[-half_length:]


def view_cases(case_recorder_file, show=True):
    """
    View the contents of a case recorder file as a dashboard.

    Parameters
    ----------
    case_recorder_file : str
        The path to the case recorder file to view.
    show : bool
        If True, show the dashboard. If False, do not show. Mostly for running tests.
    """
    if pn is None:
        raise RuntimeError(
            "The view_cases function requires the 'panel' package, "
            "which can be installed with one of the following commands:\n"
            "    pip install openmdao[visualization]\n"
            "    pip install panel"
        )

    if pd is None:
        raise RuntimeError(
            "The view_cases function requires the 'pandas' package, "
            "which can be installed with one of the following commands:\n"
            "    pip install openmdao[visualization]\n"
            "    pip install pandas"
        )

    cr = om.CaseReader(case_recorder_file)

    tabs_list = []

    # Summary tab
    data = [
        ["openmdao_version", cr.openmdao_version],
        ["format_version", cr._format_version],
        ["sources", cr.list_sources()],
        ["driver name", cr.problem_metadata['driver']['name']],
    ]
    if 'optimizer' in cr.problem_metadata['driver']['options']:
        data.append(["driver optimizer", cr.problem_metadata['driver']['options']['optimizer']])
    df = pd.DataFrame(data, columns=["Name", "Value"])
    summary_table_pane = pn.widgets.Tabulator(
        df,
        show_index=False,
        selectable=False,
        sortable=False,
        disabled=True,
    )
    summary_pane = pn.Column(
        "This page is a very high level summary of what is in the case recorder file.",
        summary_table_pane,
    )
    tabs_list.append(("Summary", summary_pane))

    # Cases Summary
    data = []
    cases = cr.get_cases()
    df = pd.DataFrame(
        columns=[
            "Counter",
            "Name",
            "Source",
            "Num Inputs",
            "Num Outputs",
            "Num Residuals",
            "Derivatives",
            "Abs Err",
            "Rel Err",
            "Message",
            "Success",
            "Timestamp",
        ]
    )
    for i, case in enumerate(cases):
        df.loc[i] = [
            case.counter,
            case.name,
            case.source,
            len(case.inputs) if case.inputs else 0,
            len(case.outputs) if case.outputs else 0,
            len(case.residuals) if case.residuals else 0,
            "Yes" if case.derivatives else "No",
            case.abs_err,
            case.rel_err,
            case.msg,
            case.success,
            case.timestamp,
        ]

    cases_table_pane = pn.widgets.Tabulator(
        df,
        show_index=False,
        selectable=False,
        sortable=True,
        disabled=True,
    )
    cases_pane = pn.Column(
        "This page lists all the Cases found in the case recorder file.", cases_table_pane
    )
    tabs_list.append(("Cases", cases_pane))

    # Give the user an idea of what is in a case by showing them
    # what is are the Case inputs, outputs and residuals if available

    sources = cr.list_sources()
    for source in sources:
        case0_id = cr.list_cases(source)[0]
        case0 = cr.get_case(case0_id)

        case0_inputs_pane = pn.Column(pn.pane.Markdown("# Inputs"))
        if case0.inputs:
            df_case0_inputs = pd.DataFrame(list(case0.inputs.keys()),
                                           columns=[f"{len(case0.inputs.keys())} Inputs"])
            case0_inputs_table_pane = pn.widgets.Tabulator(df_case0_inputs,
                                                           **tabulator_defaults)
        else:
            case0_inputs_table_pane = pn.pane.Markdown("### There are no inputs in Case 0")
        case0_inputs_pane.append(case0_inputs_table_pane)

        case0_outputs_pane = pn.Column(pn.pane.Markdown("# Outputs"))
        if case0.outputs:
            df_case0_outputs = pd.DataFrame(list(case0.outputs.keys()),
                                            columns=[f"{len(case0.outputs.keys())} Outputs"])
            case0_outputs_table_pane = pn.widgets.Tabulator(df_case0_outputs,
                                                            **tabulator_defaults)
        else:
            case0_outputs_table_pane = pn.pane.Markdown("### There are no outputs in Case 0")
        case0_outputs_pane.append(case0_outputs_table_pane)

        case0_residuals_pane = pn.Column(pn.pane.Markdown("# Residuals"))
        if case0.residuals:
            df_case0_residuals = pd.DataFrame(list(case0.residuals.keys()),
                                              columns=[f"{len(case0.residuals.keys())} Residuals"])
            case0_residuals_table_pane = pn.widgets.Tabulator(df_case0_residuals,
                                                              **tabulator_defaults)
        else:
            case0_residuals_table_pane = pn.pane.Markdown("### There are no residuals in Case 0")
        case0_residuals_pane.append(case0_residuals_table_pane)

        case0_variables_pane = pn.Column(
            f"There are {len(case0.inputs)if case0.inputs else 0} inputs,"
            f" {len(case0.outputs) if case0.outputs else 0} outputs,"
            f" and {len(case0.residuals) if case0.residuals else 0} residuals in Case 0.",
            pn.Row(
                case0_inputs_pane,
                case0_outputs_pane,
                case0_residuals_pane,
                sizing_mode="stretch_width",
                styles={
                    "display": "flex",
                    "justify-content": "space-between",
                    "width": "100%",
                },
            ),
        )
        tabs_list.append((f"Variables recorded for {source}", case0_variables_pane))

    # variables pane

    # See if we are missing any metadata names.
    metadata_names_in_file = set()
    for name, properties in cr.problem_metadata['variables'].items():
        if isinstance(properties, dict):
            metadata_names_in_file.update(list(properties.keys()))
        else:
            if name not in ['execution_order',]:
                issue_warning(
                    f"problem_metadata['variables'] has unexpected type"
                    f" of '{type(properties)}' for '{name}'",
                    category=CaseRecorderWarning,
                )

    df_variables = variable_metadata_to_data_frame(cr.problem_metadata['variables'])
    variables_table_pane = pn.widgets.Tabulator(df_variables, **tabulator_defaults)
    variables_pane = pn.Column(
        f"Here is the metadata for the {len(cr.problem_metadata['variables'])}"
        " variables in the problem",
        variables_table_pane
    )
    tabs_list.append(("Metadata variables", variables_pane))

    # design_vars pane
    df_metadata_desvars = variable_metadata_to_data_frame(cr.problem_metadata['design_vars'])
    metadata_desvars_table_pane = pn.widgets.Tabulator(
        df_metadata_desvars, **tabulator_defaults
    )
    design_vars_pane = pn.Column(
        f"Here is the metadata for the {len(cr.problem_metadata['design_vars'])}"
        " design variables in the problem",
        metadata_desvars_table_pane
    )
    tabs_list.append(("Metadata design_vars", design_vars_pane))

    # responses pane
    df_responses = variable_metadata_to_data_frame(cr.problem_metadata['responses'])
    responses_table_pane = pn.widgets.Tabulator(df_responses, **tabulator_defaults)
    responses_pane = pn.Column(
        f"Here is the metadata for the {len(cr.problem_metadata['responses'])}"
        " responses in the problem",
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
        execution_order_table_pane = pn.widgets.Tabulator(df_execution_order, **tabulator_defaults)
        execution_order_pane = pn.Column(
            "This page lists the execution order of the systems",
            execution_order_table_pane
        )
        tabs_list.append(("Execution Order", execution_order_pane))

    # abs2prom pane
    df_abs2prom_inputs = pd.DataFrame(
        list(cr.problem_metadata["abs2prom"]["input"].items()),
        columns=["Absolute Name", "Promoted Name"],
    )
    abs2prom_inputs_table_pane = pn.widgets.Tabulator(
        df_abs2prom_inputs, **tabulator_defaults
    )
    df_abs2prom_outputs = pd.DataFrame(
        list(cr.problem_metadata["abs2prom"]["output"].items()),
        columns=["Absolute Name", "Promoted Name"],
    )
    abs2prom_outputs_table_pane = pn.widgets.Tabulator(
        df_abs2prom_outputs, **tabulator_defaults
    )
    abs2prom_pane = pn.Column(
        f"There are {len(cr.problem_metadata['abs2prom']['input'])} inputs"
        f" and {len(cr.problem_metadata['abs2prom']['output'])} outputs in the metadata abs2prom",
        pn.pane.Markdown("# Inputs"),
        abs2prom_inputs_table_pane,
        pn.pane.Markdown("# Outputs"),
        abs2prom_outputs_table_pane,
    )
    tabs_list.append(("Metadata abs2prom", abs2prom_pane))

    # connections_list pane
    df_connections_list = pd.DataFrame(
        cr.problem_metadata["connections_list"], columns=["src", "tgt"]
    )
    df_connections_list.columns = ["Source", "Target"]
    df_connections_table_pane = pn.widgets.Tabulator(df_connections_list, **tabulator_defaults)
    connections_list_pane = pn.Column(
        f"There are {len(cr.problem_metadata['connections_list'])}"
        " items in the metadata connections_list",
        df_connections_table_pane,
    )
    tabs_list.append(("Metadata connections_list", connections_list_pane))

    # declare_partials_list pane
    df_declare_partials_list = pd.DataFrame(
        [item.split(" > ") for item in cr.problem_metadata["declare_partials_list"]],
        columns=["Of the Function", "With Respect To"],
    )
    declare_partials_list_table_pane = pn.widgets.Tabulator(
        df_declare_partials_list, **tabulator_defaults
    )

    declare_partials_list_pane = pn.Column(
        f"There are {len(cr.problem_metadata['declare_partials_list'])}"
        " items in the metadata declare_partials_list",
        declare_partials_list_table_pane,
    )
    tabs_list.append(("Metadata declare_partials_list", declare_partials_list_pane))

    tabs = pn.Tabs(*tabs_list, stylesheets=["assets/view_case_recorder_styles.css"])
    template = pn.template.FastListTemplate(
        title=f"Dashboard for case recorder file '{case_recorder_file}'",
        main=[tabs],
        accent_base_color="black",
        header_background="rgb(0, 212, 169)",
        background_color="white",
        theme=pn.theme.DefaultTheme,
        theme_toggle=False,
        main_layout=None,
    )

    port = 0
    home_dir = "."
    assets_dir = pathlib.Path(
        importlib.util.find_spec("openmdao").origin
    ).parent.joinpath("recorders/assets/")

    if show:
        if port == 0:
            port = get_free_port()
        server = pn.serve(
            template,
            port=port,
            address="localhost",
            websocket_origin=f"localhost:{port}",
            show=True,
            threaded=False,
            static_dirs={
                "home": home_dir,
                "assets": assets_dir,
            },
        )
        server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    _view_cases_setup_parser(parser)
    args = parser.parse_args()
    _view_cases_cmd(args, None)
