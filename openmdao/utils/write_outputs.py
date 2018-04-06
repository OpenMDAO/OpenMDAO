"""
Utility functions and constants related to writing outputs.
"""
import numpy as np
from six import iteritems

column_widths = {
    'value': 20,
    'resids': 20,
    'units': 10,
    'shape': 10,
    'lower': 20,
    'upper': 20,
    'ref': 20,
    'ref0': 20,
    'res_ref': 20,
}
align = ''
column_spacing = 2
indent_inc = 2


def write_outputs(in_or_out, comp_type, dict_of_outputs, hierarchical, print_arrays,
                  out_stream, pathname, var_allprocs_abs_names):
    """
    Write table of variable names, values, residuals, and metadata to out_stream.

    The output values could actually represent input variables.
    In this context, outputs refers to the data that is being logged to an output stream.

    Parameters
    ----------
    in_or_out : str, 'input' or 'output'
        indicates whether the values passed in are from inputs or output variables.
    comp_type : str, 'Explicit' or 'Implicit'
        the type of component with the output values.
    dict_of_outputs : dict
        dict storing vals and metadata for each var name
    hierarchical : bool
        When True, human readable output shows variables in hierarchical format.
    print_arrays : bool
        When False, in the columnar display, just display norm of any ndarrays with size > 1.
        The norm is surrounded by vertical bars to indicate that it is a norm.
        When True, also display full values of the ndarray below the row. Format  is affected
        by the values set with numpy.set_printoptions
        Default is False.
    out_stream : file-like object
        Where to send human readable output.
        Set to None to suppress.
    pathname : str
        pathname to be printed. If None, defaults to 'model'
    var_allprocs_abs_names : {'input': [], 'output': []}
        set of variable names across all processes
    """
    count = len(dict_of_outputs)

    # Write header
    pathname = pathname if pathname else 'model'
    header_name = 'Input' if in_or_out == 'input' else 'Output'
    if in_or_out == 'input':
        header = "%d %s(s) in '%s'" % (count, header_name, pathname)
    else:
        header = "%d %s %s(s) in '%s'" % (
            count, comp_type, header_name, pathname)
    out_stream.write(header + '\n')
    out_stream.write('-' * len(header) + '\n' + '\n')

    if not count:
        return

    # Need an ordered list of possible output values for the two cases: inputs and outputs
    #  so that we do the column output in the correct order
    if in_or_out == 'input':
        out_types = ('value', 'units',)
    else:
        out_types = ('value', 'resids', 'units', 'shape', 'lower', 'upper', 'ref',
                     'ref0', 'res_ref')
    # Figure out which columns will be displayed
    # Look at any one of the outputs, they should all be the same
    outputs = dict_of_outputs[list(dict_of_outputs)[0]]
    column_names = []
    for out_type in out_types:
        if out_type in outputs:
            column_names.append(out_type)

    top_level_system_name = 'top'

    # Find with width of the first column in the table
    #    Need to look through all the possible varnames to find the max width
    max_varname_len = max(len(top_level_system_name), len('varname'))
    if hierarchical:
        for name, outs in iteritems(dict_of_outputs):
            for i, name_part in enumerate(name.split('.')):
                total_len = (i + 1) * indent_inc + len(name_part)
                max_varname_len = max(max_varname_len, total_len)
    else:
        for name, outs in iteritems(dict_of_outputs):
            max_varname_len = max(max_varname_len, len(name))

    # Determine the column widths of the data fields by finding the max width for all rows
    for column_name in column_names:
        column_widths[column_name] = len(
            column_name)  # has to be able to display name!
    for name in var_allprocs_abs_names[in_or_out]:
        if name in dict_of_outputs:
            for column_name in column_names:
                if isinstance(dict_of_outputs[name][column_name], np.ndarray) and \
                        dict_of_outputs[name][column_name].size > 1:
                    out = '|{}|'.format(
                        str(np.linalg.norm(dict_of_outputs[name][column_name])))
                else:
                    out = str(dict_of_outputs[name][column_name])
                column_widths[column_name] = max(column_widths[column_name],
                                                 len(str(out)))

    # Write out the column headers
    column_header = '{:{align}{width}}'.format('varname', align=align,
                                               width=max_varname_len)
    column_dashes = max_varname_len * '-'
    for column_name in column_names:
        column_header += column_spacing * ' '
        column_header += '{:{align}{width}}'.format(column_name, align=align,
                                                    width=column_widths[column_name])
        column_dashes += column_spacing * ' ' + \
            column_widths[column_name] * '-'
    out_stream.write(column_header + '\n')
    out_stream.write(column_dashes + '\n')

    # Write out the variable names and optional values and metadata
    if hierarchical:
        out_stream.write(top_level_system_name + '\n')

        cur_sys_names = []
        # _var_allprocs_abs_names has all the vars across all procs in execution order
        #   But not all the values need to be written since, at least for output vars,
        #      the output var lists are divided into explicit and implicit
        for varname in var_allprocs_abs_names[in_or_out]:
            if varname not in dict_of_outputs:
                continue

            # For hierarchical, need to display system levels in the rows above the
            #   actual row containing the var name and values. Want to make use
            #   of the hierarchies that have been written about this.
            existing_sys_names = []
            varname_sys_names = varname.split('.')[:-1]
            for i, sys_name in enumerate(varname_sys_names):
                if varname_sys_names[:i + 1] != cur_sys_names[:i + 1]:
                    break
                else:
                    existing_sys_names = cur_sys_names[:i + 1]

            # What parts of the hierarchy for this varname need to be written that
            #   were not already written above this
            remaining_sys_path_parts = varname_sys_names[len(
                existing_sys_names):]

            # Write the Systems in the var name path
            indent = len(existing_sys_names) * indent_inc
            for i, sys_name in enumerate(remaining_sys_path_parts):
                indent += indent_inc
                out_stream.write(indent * ' ' + sys_name + '\n')
            cur_sys_names = varname_sys_names

            indent += indent_inc
            row = '{:{align}{width}}'.format(indent * ' ' + varname.split('.')[-1],
                                             align=align, width=max_varname_len)
            _write_outputs_rows(out_stream, row, column_names, dict_of_outputs[varname],
                                print_arrays)
    else:
        for name in var_allprocs_abs_names[in_or_out]:
            if name in dict_of_outputs:
                row = '{:{align}{width}}'.format(
                    name, align=align, width=max_varname_len)
                _write_outputs_rows(out_stream, row, column_names, dict_of_outputs[name],
                                    print_arrays)
    out_stream.write(2 * '\n')


def _write_outputs_rows(out_stream, row, column_names, dict_of_outputs, print_arrays):
    """
    For one variable, write name, values, residuals, and metadata to out_stream.

    Parameters
    ----------
    out_stream : file-like object
        Where to send human readable output.
        Set to None to suppress.
    row : str
        The string containing the contents of the beginning of this row output.
        Contains the name of the System or varname, possibley indented to show hierarchy.

    column_names : list of str
        Indicates which columns will be written in this row.

    dict_of_outputs : dict
        Contains the values to be written in this row. Keys are columns names.

    print_arrays : bool
        When False, in the columnar display, just display norm of any ndarrays with size > 1.
        The norm is surrounded by vertical bars to indicate that it is a norm.
        When True, also display full values of the ndarray below the row. Format  is affected
        by the values set with numpy.set_printoptions
        Default is False.

    """
    if out_stream is None:
        return
    left_column_width = len(row)
    have_array_values = []  # keep track of which values are arrays
    for column_name in column_names:
        row += column_spacing * ' '
        if isinstance(dict_of_outputs[column_name], np.ndarray) and \
                dict_of_outputs[column_name].size > 1:
            have_array_values.append(column_name)
            out = '|{}|'.format(
                str(np.linalg.norm(dict_of_outputs[column_name])))
        else:
            out = str(dict_of_outputs[column_name])
        row += '{:{align}{width}}'.format(out, align=align,
                                          width=column_widths[column_name])
    out_stream.write(row + '\n')
    if print_arrays:
        for column_name in have_array_values:
            out_stream.write("{}  {}:\n".format(
                left_column_width * ' ', column_name))
            out_str = str(dict_of_outputs[column_name])
            indented_lines = [(left_column_width + indent_inc) * ' ' +
                              s for s in out_str.splitlines()]
            out_stream.write('\n'.join(indented_lines) + '\n')
