"""
Utility functions and constants related to writing a table of variable metadata.
"""
import sys
import pprint

from io import TextIOBase

import numpy as np

from openmdao.core.constants import _DEFAULT_OUT_STREAM
from openmdao.utils.notebook_utils import notebook, display, HTML
from openmdao.visualization.tables.table_builder import generate_table

# string to display when an attribute is not available
NA = 'n/a'

column_widths = {
    'val': 20,
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


def write_var_table(pathname, var_list, var_type, var_dict,
                    hierarchical=True, print_arrays=False,
                    out_stream=_DEFAULT_OUT_STREAM):
    """
    Write table of variable names, values, residuals, and metadata to out_stream.

    Parameters
    ----------
    pathname : str
        Pathname to be printed. If None, defaults to 'model'.
    var_list : list of str
        List of variable names in the order they are to be written.
    var_type : 'input', 'explicit', 'implicit', or 'all'
        Indicates type of variables, input or explicit/implicit output, or all for all vars.
    var_dict : dict
        Dict storing vals and metadata for each var name.
    hierarchical : bool
        When True, human readable output shows variables in hierarchical format.
    print_arrays : bool
        When False, in the columnar display, just display norm of any ndarrays with size > 1.
        The norm is surrounded by vertical bars to indicate that it is a norm.
        When True, also display full values of the ndarray below the row. Format  is affected
        by the values set with numpy.set_printoptions.
    out_stream : file-like object
        Where to send human readable output.
    """
    if out_stream is None:
        return

    if notebook and not hierarchical and out_stream is _DEFAULT_OUT_STREAM:
        use_html = True
    else:
        use_html = False

    if out_stream is _DEFAULT_OUT_STREAM:
        out_stream = sys.stdout
    elif not isinstance(out_stream, TextIOBase):
        raise TypeError("Invalid output stream specified for 'out_stream'.")

    count = len(var_dict)

    # Write header
    rel_idx = len(pathname) + 1 if pathname else 0
    pathname = pathname if pathname else 'model'

    if var_type == 'input':
        header = "%d Input(s) in '%s'" % (count, pathname)
    elif var_type == 'all':
        header = "%d Variables(s) in '%s'" % (count, pathname)
    else:
        header = "%d %s Output(s) in '%s'" % (count, var_type.capitalize(), pathname)

    out_stream.write(header + '\n')

    if not count:
        out_stream.write('\n\n')
        return

    # Need an ordered list of possible output values for the two cases: inputs and outputs
    #  so that we do the column output in the correct order
    if var_type == 'input':
        out_types = ('val', 'units', 'shape', 'global_shape', 'prom_name', 'desc', 'min', 'max',
                     'tags')
    elif var_type == 'all':
        out_types = ('val', 'io', 'resids', 'units', 'shape', 'global_shape', 'lower', 'upper',
                     'ref', 'ref0', 'res_ref', 'prom_name', 'desc', 'min', 'max', 'tags')
    else:
        out_types = ('val', 'resids', 'units', 'shape', 'global_shape', 'lower', 'upper',
                     'ref', 'ref0', 'res_ref', 'prom_name', 'desc', 'min', 'max', 'tags')

    # Figure out which columns will be displayed.
    for var_meta in var_dict.values():
        # if 'all', look for an output as some fields (bounds, scaling) are only found in outputs
        # otherwise just take the first meta dict since they should all be the same
        if var_type == 'all' and var_meta['io'] != 'output':
            continue
        break

    column_names = [out_type for out_type in out_types if out_type in var_meta]

    if 'tags' in column_names:
        # if printing tags, print as a list (value may be a list or a set)
        for meta in var_dict.values():
            meta['tags'] = list(meta['tags'])

    if use_html and var_list:
        rows = []
        for name in var_list:
            rows.append([name] + [var_dict[name][field] for field in column_names])

        hdrs = ['varname'] + column_names
        display(HTML(str(generate_table(rows, headers=hdrs, tablefmt='html'))))
        return

    # Find with width of the first column in the table
    #    Need to look through all the possible varnames to find the max width
    max_varname_len = len('varname')
    if hierarchical:
        for name in var_dict:
            for i, name_part in enumerate(name[rel_idx:].split('.')):
                total_len = i * indent_inc + len(name_part)
                max_varname_len = max(max_varname_len, total_len)
    else:
        for name in var_dict:
            max_varname_len = max(max_varname_len, len(name[rel_idx:]))

    # Determine the column widths of the data fields by finding the max width for all rows
    for column_name in column_names:
        column_widths[column_name] = len(column_name)  # has to be able to display name!

    for name in var_list:
        for column_name in column_names:
            try:
                column_value = var_dict[name][column_name]
            except KeyError:
                column_value = NA
            if isinstance(column_value, np.ndarray) and column_value.size > 1:
                out = '|{}|'.format(str(np.linalg.norm(column_value)))
            else:
                out = str(column_value)
            column_widths[column_name] = max(column_widths[column_name], len(str(out)))

    # Write out the column headers
    column_header = '{:{align}{width}}'.format('varname', align=align,
                                               width=max_varname_len)
    column_dashes = max_varname_len * '-'
    for column_name in column_names:
        column_header += column_spacing * ' '
        column_header += '{:{align}{width}}'.format(column_name, align=align,
                                                    width=column_widths[column_name])
        column_dashes += column_spacing * ' ' + column_widths[column_name] * '-'

    out_stream.write('\n')
    out_stream.write(column_header + '\n')
    out_stream.write(column_dashes + '\n')

    # Write out the variable names and optional values and metadata
    if hierarchical:

        cur_sys_names = []

        for abs_name in var_list:
            rel_name = abs_name[rel_idx:]

            # For hierarchical, need to display system levels in the rows above the
            #   actual row containing the var name and values. Want to make use
            #   of the hierarchies that have been written about this.
            existing_sys_names = []
            sys_names = rel_name.split('.')[:-1]
            for i, sys_name in enumerate(sys_names):
                if sys_names[:i + 1] != cur_sys_names[:i + 1]:
                    break
                else:
                    existing_sys_names = cur_sys_names[:i + 1]

            # What parts of the hierarchy for this varname need to be written that
            #   were not already written above this
            remaining_sys_path_parts = sys_names[len(existing_sys_names):]

            # Write the Systems in the var name path
            indent = len(existing_sys_names) * indent_inc
            for i, sys_name in enumerate(remaining_sys_path_parts):
                out_stream.write(indent * ' ' + sys_name + '\n')
                indent += indent_inc
            cur_sys_names = sys_names

            row = '{:{align}{width}}'.format(indent * ' ' + abs_name.split('.')[-1],
                                             align=align, width=max_varname_len)
            _write_variable(out_stream, row, column_names, var_dict[abs_name], print_arrays)
    else:
        for name in var_list:
            row = '{:{align}{width}}'.format(name[rel_idx:], align=align, width=max_varname_len)
            _write_variable(out_stream, row, column_names, var_dict[name], print_arrays)

    out_stream.write('\n\n')


def write_source_table(source_dicts, out_stream):
    """
    Write tables of cases and their respective sources.

    Parameters
    ----------
    source_dicts : dict or list of dicts
        Dict of source and cases.
    out_stream : file-like object
        Where to send human readable output.
    """
    if out_stream is None:
        return

    # use table_builder if we are in a notebook and are using the default out_stream
    use_html = notebook and out_stream is _DEFAULT_OUT_STREAM

    if out_stream is _DEFAULT_OUT_STREAM:
        out_stream = sys.stdout
    elif not isinstance(out_stream, TextIOBase):
        raise TypeError("Invalid output stream specified for 'out_stream'.")

    if not source_dicts:
        out_stream.write('No data found.\n')
        return

    if not isinstance(source_dicts, list):
        source_dicts = [source_dicts]

    for source_dict in source_dicts:
        if use_html:
            display(HTML(str(generate_table(source_dict, headers='keys', tablefmt='html'))))
        else:
            for key, value in source_dict.items():
                if value:
                    out_stream.write(f'{key}\n')
                    for val in value:
                        out_stream.write(f'    {val}\n')


def _write_variable(out_stream, row, column_names, var_dict, print_arrays):
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

    var_dict : dict
        Contains the values to be written in this row. Keys are columns names.

    print_arrays : bool
        When False, in the columnar display, just display norm of any ndarrays with size > 1.
        The norm is surrounded by vertical bars to indicate that it is a norm.
        When True, also display full values of the ndarray below the row. Format is affected
        by the values set with numpy.set_printoptions
        Default is False.

    """
    if out_stream is None:
        return
    elif out_stream is _DEFAULT_OUT_STREAM:
        out_stream = sys.stdout

    left_column_width = len(row)
    have_array_values = []  # keep track of which values are arrays
    print_options = np.get_printoptions()
    np_precision = print_options['precision']
    for column_name in column_names:
        row += column_spacing * ' '

        try:
            column_val = var_dict[column_name]
        except KeyError:
            column_val = NA
        if isinstance(column_val, np.ndarray) and column_val.size > 1:
            have_array_values.append(column_name)
            norm = np.linalg.norm(var_dict[column_name])
            out = '|{}|'.format(str(np.round(norm, np_precision)))
        else:
            out = str(column_val)
        row += '{:{align}{width}}'.format(out, align=align,
                                          width=column_widths[column_name])
    out_stream.write(row + '\n')
    if print_arrays:
        for column_name in have_array_values:
            out_stream.write("{}  {}:\n".format(
                left_column_width * ' ', column_name))
            out_str = pprint.pformat(var_dict[column_name])
            indented_lines = [(left_column_width + indent_inc) * ' ' +
                              s for s in out_str.splitlines()]
            out_stream.write('\n'.join(indented_lines) + '\n')
