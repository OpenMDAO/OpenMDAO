"""
A collection of utilities for file wrapping.

Note: This is a work in progress.
"""


import re

from pyparsing import CaselessLiteral, Combine, OneOrMore, Optional, \
    TokenConverter, Word, nums, oneOf, printables, ParserElement, alphanums

import numpy as np


def _getformat(val):
    """
    Get the output format for a floating point number.

    The general format is used with 16 places of accuracy, except for when
    the floating point value is an integer, in which case a decimal point
    followed by a single zero is used.

    Parameters
    ----------
    val : float or int
        the number which needs formatted.

    Returns
    -------
    string
        the format string.
    """
    if int(val) == val:
        return "%.1f"
    else:
        return "%.16g"


class _SubHelper(object):
    """
    Replaces file text at the correct word location in a line.

    This class contains the Helper Function that is passed to re.sub.

    Attributes
    ----------
    _newtext : string
        text to insert.
    _replace_location : int
        location in the file where replacement is to occur.
    _current_location : int
        current location in the file.
    _counter : int
        counter
    _start_location : int
        initial location where replacement is to occur.
    _end_location : int
        final location where replacement is to occur.
    """

    def __init__(self):
        """
        Initialize attributes.
        """
        self._newtext = ""
        self._replace_location = 0
        self._current_location = 0
        self._counter = 0
        self._start_location = 0
        self._end_location = 0

    def set(self, newtext, location):
        """
        Set a new word location and value for replacement.

        Parameters
        ----------
        newtext : string
            text to insert.
        location : int
            location in the file where replacement is to occur.
        """
        self._newtext = newtext
        self._replace_location = location
        self._current_location = 0

    def set_array(self, newtext, start_location, end_location):
        """
        Set a new starting location, ending location, and value for replacement.

        Parameters
        ----------
        newtext : string
            text to insert.
        start_location : int
            location
        end_location : int
            location
        """
        self._newtext = newtext
        self._start_location = start_location
        self._end_location = end_location
        self._current_location = 0

    def replace(self, text):
        """
        Replace text in file.

        This function should be passed to re.sub.

        Parameters
        ----------
        text : string
            text to insert.

        Returns
        -------
        string
            newtext if current location is replace location else the input text.
        """
        self._current_location += 1

        if self._current_location == self._replace_location:
            if isinstance(self._newtext, float):
                return _getformat(self._newtext) % self._newtext
            else:
                return str(self._newtext)
        else:
            return text.group()

    def replace_array(self, text):
        """
        Replace array of text values in file.

        This function should be passed to re.sub.

        Parameters
        ----------
        text : string
            text to insert.

        Returns
        -------
        string
            newtext if current location is replace location else the input text.
        """
        self._current_location += 1
        end = len(self._newtext)

        if self._current_location >= self._start_location and \
           self._current_location <= self._end_location and \
           self._counter < end:
            if isinstance(self._newtext[self._counter], float):
                val = self._newtext[self._counter]
                newval = _getformat(val) % val
            else:
                newval = str(self._newtext[self._counter])
            self._counter += 1
            return newval
        else:
            return text.group()


class ToInteger(TokenConverter):
    """
    Converter for PyParsing that is used to turn a token into an int.
    """

    def postParse(self, instring, loc, tokenlist):
        """
        Convert token into an integer.

        Parameters
        ----------
        instring : string
            the input string
        loc : int
            the location of the matching string
        tokenlist : list
            list of matched tokens

        Returns
        -------
        int
            integer value for token.
        """
        return int(tokenlist[0])


class ToFloat(TokenConverter):
    """
    Converter for PyParsing that is used to turn a token into a float.
    """

    def postParse(self, instring, loc, tokenlist):
        """
        Convert token into a float.

        Parameters
        ----------
        instring : string
            the input string
        loc : int
            the location of the matching string
        tokenlist : list
            list of matched tokens

        Returns
        -------
        float
            float value for token.
        """
        return float(tokenlist[0].replace('D', 'E'))


class ToNan(TokenConverter):
    """
    Converter for PyParsing that is used to turn a token into Python nan.
    """

    def postParse(self, instring, loc, tokenlist):
        """
        Convert token into Python nan.

        Parameters
        ----------
        instring : string
            the input string
        loc : int
            the location of the matching string
        tokenlist : list
            list of matched tokens

        Returns
        -------
        float
            the float value for NaN.
        """
        return float('nan')


class ToInf(TokenConverter):
    """
    Converter for PyParsing that is used to turn a token into Python inf.
    """

    def postParse(self, instring, loc, tokenlist):
        """
        Convert token into Python inf.

        Parameters
        ----------
        instring : string
            the input string
        loc : int
            the location of the matching string
        tokenlist : list
            list of matched tokens

        Returns
        -------
        float
            the float value for infinity.
        """
        return float('inf')


class InputFileGenerator(object):
    """
    Utility to generate an input file from a template.

    Substitution of values is supported. Data is located with a simple API.

    Attributes
    ----------
    _template_filename : string or None
        the name of the template file.
    _output_filename : string or None
        the name of the output file.
    _delimiter : int
        delimiter.
    _reg : int
        regular expression.
    _data : list of string
        the contents of the file, by line
    _current_row : int
        the current row of the file
    _anchored : bool
        indicator that position is relative to a landmark location.
    """

    def __init__(self):
        """
        Initialize attributes.
        """
        self._template_filename = None
        self._output_filename = None

        self._delimiter = " "
        self._reg = re.compile('[^ \n]+')

        self._data = []
        self._current_row = 0
        self._anchored = False

    def set_template_file(self, filename):
        """
        Set the name of the template file to be used.

        The template file is also read into memory when this method is called.

        Parameters
        ----------
        filename : string
            Name of the template file to be used.
        """
        self._template_filename = filename

        templatefile = open(filename, 'r')
        self._data = templatefile.readlines()
        templatefile.close()

    def set_generated_file(self, filename):
        """
        Set the name of the file that will be generated.

        Parameters
        ----------
        filename : string
            Name of the input file to be generated.
        """
        self._output_filename = filename

    def set_delimiters(self, delimiter):
        """
        Set the delimiters that are used to identify field boundaries.

        Parameters
        ----------
        delimiter : str
            A string containing characters to be used as delimiters.
        """
        self._delimiter = delimiter
        self._reg = re.compile('[^' + delimiter + '\n]+')

    def mark_anchor(self, anchor, occurrence=1):
        """
        Mark the location of a landmark.

        This lets you describe data by relative position. Note that a forward
        search begins at the old anchor location. If you want to restart the
        search for the anchor at the file beginning, then call ``reset_anchor()``
        before ``mark_anchor``.

        Parameters
        ----------
        anchor : string
            The text you want to search for.

        occurrence : integer, optional
            Find nth instance of text; default is 1 (first). Use -1 to
            find last occurrence. Reverse searches always start at the end
            of the file no matter the state of any previous anchor.
        """
        if not isinstance(occurrence, int):
            raise ValueError("The value for occurrence must be an integer")

        instance = 0
        if occurrence > 0:
            count = 0
            max_lines = len(self._data)
            for index in range(self._current_row, max_lines):
                line = self._data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only after the anchor.
                if count == 0 and self._anchored:
                    line = line.split(anchor)[-1]

                if line.find(anchor) > -1:

                    instance += 1
                    if instance == occurrence:
                        self._current_row += count
                        self._anchored = True
                        return

                count += 1

        elif occurrence < 0:
            max_lines = len(self._data) - 1
            count = max_lines
            for index in range(max_lines, -1, -1):
                line = self._data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only before the anchor.
                if count == max_lines and self._anchored:
                    line = line.split(anchor)[0]

                if line.find(anchor) > -1:
                    instance += -1
                    if instance == occurrence:
                        self._current_row = count
                        self._anchored = True
                        return

                count -= 1
        else:
            raise ValueError("0 is not valid for an anchor occurrence.")

        raise RuntimeError("Could not find pattern %s in template file %s" %
                           (anchor, self._template_filename))

    def reset_anchor(self):
        """
        Reset anchor to the beginning of the file.
        """
        self._current_row = 0
        self._anchored = False

    def transfer_var(self, value, row, field):
        """
        Change a single variable in the template relative to the current anchor.

        Parameters
        ----------
        value : float, integer, bool, string
            New value to set at the location.

        row : integer
            Number of lines offset from anchor line (0 is anchor line).
            This can be negative.

        field : integer
            Which word in line to replace, as denoted by delimiter(s)
        """
        j = self._current_row + row
        line = self._data[j]

        sub = _SubHelper()
        sub.set(value, field)
        newline = re.sub(self._reg, sub.replace, line)

        self._data[j] = newline

    def transfer_array(self, value, row_start, field_start, field_end,
                       row_end=None, sep=", "):
        """
        Change the values of an array in the template relative to the current anchor.

        This should generally be used for one-dimensional or free form arrays.

        Parameters
        ----------
        value : float, integer, bool, str
            Array of values to insert.
        row_start : integer
            Starting row for inserting the array. This is relative
            to the anchor, and can be negative.
        field_start : integer
            Starting field in the given row_start as denoted by
            delimiter(s).
        field_end : integer
            The final field the array uses in row_end.
            We need this to figure out if the template is too small or large
        row_end : integer, optional
            Use if the array wraps to cover additional lines.
        sep : integer, optional
            Separator to use if we go beyond the template.
        """
        # Simplified input for single-line arrays
        if row_end is None:
            row_end = row_start

        sub = _SubHelper()

        for row in range(row_start, row_end + 1):
            j = self._current_row + row
            line = self._data[j]

            if row == row_end:
                f_end = field_end
            else:
                f_end = 99999

            sub.set_array(value, field_start, f_end)
            field_start = 0

            newline = re.sub(self._reg, sub.replace_array, line)
            self._data[j] = newline

        # Sometimes an array is too large for the example in the template
        # This is resolved by adding more fields at the end
        if sub._counter < len(value):
            for val in value[sub._counter:]:
                newline = newline.rstrip() + sep + str(val)
            self._data[j] = newline

        # Sometimes an array is too small for the template
        # This is resolved by removing fields
        elif sub._counter > len(value):
            # TODO - Figure out how to handle this.
            # Ideally, we'd remove the extra field placeholders
            raise ValueError("Array is too small for the template.")

    def transfer_2Darray(self, value, row_start, row_end, field_start, field_end):
        """
        Change the values of a 2D array in the template relative to the current anchor.

        This method is specialized for 2D arrays, where each row of the array is
        on its own line.

        Parameters
        ----------
        value : ndarray
            Array of values to insert.
        row_start : integer
            Starting row for inserting the array. This is relative
            to the anchor, and can be negative.
        row_end : integer
            Final row for the array, relative to the anchor.
        field_start : integer
            starting field in the given row_start as denoted by
            delimiter(s).
        field_end : integer
            The final field the array uses in row_end.
            We need this to figure out if the template is too small or large.
        """
        sub = _SubHelper()

        i = 0

        for row in range(row_start, row_end + 1):
            j = self._current_row + row
            line = self._data[j]

            sub.set_array(value[i, :], field_start, field_end)

            newline = re.sub(self._reg, sub.replace_array, line)
            self._data[j] = newline

            sub._current_location = 0
            sub._counter = 0
            i += 1

        # TODO - Note, we currently can't handle going beyond the end of
        #        the template line

    def clearline(self, row):
        """
        Replace the contents of a row with the newline character.

        Parameters
        ----------
        row : integer
            Row number to clear, relative to current anchor.
        """
        self._data[self._current_row + row] = "\n"

    def generate(self, return_data=False):
        """
        Use the template file to generate the input file.

        Parameters
        ----------
        return_data : bool
            if True, generated file data will be returned as a string

        Returns
        -------
        string
            the generated file data if return_data is True or output filename
            has not been provided, else None
        """
        if self._output_filename:
            with open(self._output_filename, 'w') as f:
                f.writelines(self._data)
        else:
            return_data = True

        if return_data:
            return '\n'.join(self._data)
        else:
            return None


class FileParser(object):
    """
    Utility to locate and read data from a file.

    Attributes
    ----------
    _filename : string
        the name of the file.
    _data : list of string
        the contents of the file, by line
    _delimiter : string
        the name of the file.
    _end_of_line_comment_char : string
        end-of-line comment character to be ignored.
    _full_line_comment_char : string
        comment character that signifies a line should be skipped.
    _current_row : int
        the current row of the file.
    _anchored : bool
        indicator that position is relative to a landmark location.
    """

    def __init__(self, end_of_line_comment_char=None, full_line_comment_char=None):
        """
        Initialize attributes.

        Parameters
        ----------
        end_of_line_comment_char : string, optional
            end-of-line comment character to be ignored.
            (e.g., Python supports in-line comments with "#".)

        full_line_comment_char : string, optional
            comment character that signifies a line should be skipped.
        """
        self._filename = None
        self._data = []

        self._delimiter = " \t"
        self._end_of_line_comment_char = end_of_line_comment_char
        self._full_line_comment_char = full_line_comment_char

        self._current_row = 0
        self._anchored = False

        self.set_delimiters(self._delimiter)

    def set_file(self, filename):
        """
        Set the name of the file that will be generated.

        Parameters
        ----------
        filename : string
            Name of the input file to be generated.
        """
        self._filename = filename

        inputfile = open(filename, 'r')

        if not self._end_of_line_comment_char and not self._full_line_comment_char:
            self._data = inputfile.readlines()
        else:
            self._data = []
            for line in inputfile:
                if line[0] == self._full_line_comment_char:
                    continue
                self._data.append(line.split(self._end_of_line_comment_char)[0])

        inputfile.close()

    def set_delimiters(self, delimiter):
        r"""
        Set the delimiters that are used to identify field boundaries.

        Parameters
        ----------
        delimiter : string
            A string containing characters to be used as delimiters. The
            default value is ' \t', which means that spaces and tabs are not
            taken as data but instead mark the boundaries. Note that the
            parser is smart enough to recognize characters within quotes as
            non-delimiters.
        """
        self._delimiter = delimiter

        if delimiter != "columns":
            ParserElement.setDefaultWhitespaceChars(str(delimiter))

        self._reset_tokens()

    def mark_anchor(self, anchor, occurrence=1):
        """
        Mark the location of a landmark, which lets you describe data by relative position.

        Note that a forward search begins at the old anchor location. If you want to restart
        the search for the anchor at the file beginning, then call ``reset_anchor()`` before
        ``mark_anchor``.

        Parameters
        ----------
        anchor : str
            The text you want to search for.
        occurrence : integer
            Find nth instance of text; default is 1 (first). Use -1 to
            find last occurrence. Reverse searches always start at the end
            of the file no matter the state of any previous anchor.
        """
        if not isinstance(occurrence, int):
            raise ValueError("The value for occurrence must be an integer")

        instance = 0

        if occurrence > 0:
            count = 0
            max_lines = len(self._data)
            for index in range(self._current_row, max_lines):
                line = self._data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only after the anchor.
                if count == 0 and self._anchored:
                    line = line.split(anchor)[-1]

                if anchor in line:

                    instance += 1
                    if instance == occurrence:
                        self._current_row += count
                        self._anchored = True
                        return

                count += 1

        elif occurrence < 0:
            max_lines = len(self._data) - 1
            count = max_lines
            for index in range(max_lines, -1, -1):
                line = self._data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only before the anchor.
                if count == max_lines and self._anchored:
                    line = line.split(anchor)[0]

                if anchor in line:
                    instance += -1
                    if instance == occurrence:
                        self._current_row = count
                        self._anchored = True
                        return

                count -= 1
        else:
            raise ValueError("0 is not valid for an anchor occurrence.")

        raise RuntimeError("Could not find pattern %s in output file %s" %
                           (anchor, self._filename))

    def reset_anchor(self):
        """
        Reset anchor to the beginning of the file.
        """
        self._current_row = 0
        self._anchored = False

    def transfer_line(self, row):
        """
        Return an entire line, relative to current anchor.

        Parameters
        ----------
        row : integer
            Number of lines offset from anchor line (0 is anchor line).
            This can be negative.

        Returns
        -------
        string
            line at the location requested
        """
        return self._data[self._current_row + row].rstrip()

    def transfer_var(self, row, field, fieldend=None):
        """
        Get a single variable relative to the current anchor.

        Parameters
        ----------
        row : integer
            Number of lines offset from anchor line (0 is anchor line).
            This can be negative.
        field : integer
            If the delimiter is a set of chars: which word in line to retrieve.
            If the delimiter is 'columns': character position to start.
        fieldend : integer (optional)
            If the delimiter is a set of chars: IGNORED.
            If the delimiter is 'columns': position of last character to return, or if
            omitted, the end of the line is used.

        Returns
        -------
        string
            data from the requested location in the file
        """
        j = self._current_row + row

        line = self._data[j]

        if self._delimiter == "columns":
            if not fieldend:
                line = line[(field - 1):]
            else:
                line = line[(field - 1):(fieldend)]

            # Let pyparsing figure out if this is a number, and return it
            # as a float or int as appropriate
            data = self._parse_line().parseString(line)

            # data might have been split if it contains whitespace. If so,
            # just return the whole string
            if len(data) > 1:
                return line
            else:
                return data[0]
        else:
            data = self._parse_line().parseString(line)
            return data[field - 1]

    def transfer_keyvar(self, key, field, occurrence=1, rowoffset=0):
        """
        Search for a key relative to the current anchor and get a field from that line.

        You can do the same thing with a call to ``mark_anchor`` and ``transfer_var``.
        This function just combines them for convenience.

        Parameters
        ----------
        key : string
            the key to search for.
        field : integer
            Which field to transfer. Field 0 is the key.
        occurrence : integer
            Find nth instance of text; default is 1 (first value
            field). Use -1 to find last occurance. Position 0 is the key
            field, so it should not be used as a value for occurrence.
        rowoffset : integer (optional)
            Optional row offset from the occurrence of key. This can
            also be negative.

        Returns
        -------
        string
            data from the requested location in the file
        """
        if not isinstance(occurrence, int) or occurrence == 0:
            msg = "The value for occurrence must be a nonzero integer"
            raise ValueError(msg)

        instance = 0
        if occurrence > 0:
            row = 0
            for line in self._data[self._current_row:]:
                if line.find(key) > -1:
                    instance += 1
                    if instance == occurrence:
                        break
                row += 1

        elif occurrence < 0:
            row = -1
            for line in reversed(self._data[self._current_row:]):
                if line.find(key) > -1:
                    instance += -1
                    if instance == occurrence:
                        break
                row -= 1

        j = self._current_row + row + rowoffset
        line = self._data[j]

        fields = self._parse_line().parseString(line.replace(key, "KeyField"))

        return fields[field]

    def transfer_array(self, rowstart, fieldstart, rowend=None, fieldend=None):
        """
        Get an array of variables relative to the current anchor.

        Setting the delimiter to 'columns' elicits some special behavior
        from this method. Normally, the extraction process wraps around
        at the end of a line and continues grabbing each field at the start of
        a newline. When the delimiter is set to columns, the parameters
        (rowstart, fieldstart, rowend, fieldend) demark a box, and all
        values in that box are retrieved. Note that standard whitespace
        is the secondary delimiter in this case.

        Parameters
        ----------
        rowstart : integer
            Row number to start, relative to the current anchor.
        fieldstart : integer
            Field number to start.
        rowend : integer, optional
            Row number to end. If not set, then only one row is grabbed.
        fieldend : integer
            Field number to end.

        Returns
        -------
        string
            data from the requested location in the file
        """
        j1 = self._current_row + rowstart

        if rowend is None:
            j2 = j1 + 1
        else:
            j2 = self._current_row + rowend + 1

        if not fieldend:
            raise ValueError("fieldend is missing, currently required")

        lines = self._data[j1:j2]

        data = np.zeros(shape=(0, 0))

        for i, line in enumerate(lines):
            if self._delimiter == "columns":
                line = line[(fieldstart - 1):fieldend]

                # Stripping whitespace may be controversial.
                line = line.strip()

                # Let pyparsing figure out if this is a number, and return it
                # as a float or int as appropriate
                parsed = self._parse_line().parseString(line)

                newdata = np.array(parsed[:])
                # data might have been split if it contains whitespace. If the
                # data is string, we probably didn't want this.
                if newdata.dtype.type is np.str_:
                    newdata = np.array(line)

                data = np.append(data, newdata)
            else:
                parsed = self._parse_line().parseString(line)

                if i == j2 - j1 - 1:
                    data = np.append(data, np.array(parsed[(fieldstart - 1):fieldend]))
                else:
                    data = np.append(data, np.array(parsed[(fieldstart - 1):]))

                fieldstart = 1

        return data

    def transfer_2Darray(self, rowstart, fieldstart, rowend, fieldend=None):
        """
        Get a 2D array of variables relative to the current anchor.

        Each line of data is placed in a separate row.

        If the delimiter is set to 'columns', then the values contained in
        fieldstart and fieldend should be the column number instead of the
        field number.

        Parameters
        ----------
        rowstart : integer
            Row number to start, relative to the current anchor.
        fieldstart : integer
            Field number to start.
        rowend : integer
            Row number to end relative to current anchor.
        fieldend : integer (optional)
            Field number to end. If not specified, grabs all fields up to the
            end of the line.

        Returns
        -------
        string
            data from the requested location in the file
        """
        if fieldend and (fieldstart > fieldend):
            msg = "fieldend must be greater than fieldstart"
            raise ValueError(msg)

        if rowstart > rowend:
            msg = "rowend must be greater than rowstart"
            raise ValueError(msg)

        j1 = self._current_row + rowstart
        j2 = self._current_row + rowend + 1
        lines = list(self._data[j1:j2])

        if self._delimiter == "columns":
            if fieldend:
                line = lines[0][(fieldstart - 1):fieldend]
            else:
                line = lines[0][(fieldstart - 1):]

            parsed = self._parse_line().parseString(line)
            row = np.array(parsed[:])
            data = np.zeros(shape=(abs(j2 - j1), len(row)))
            data[0, :] = row

            for i, line in enumerate(list(lines[1:])):
                if fieldend:
                    line = line[(fieldstart - 1):fieldend]
                else:
                    line = line[(fieldstart - 1):]

                parsed = self._parse_line().parseString(line)
                data[i + 1, :] = np.array(parsed[:])
        else:
            parsed = self._parse_line().parseString(lines[0])
            if fieldend:
                row = np.array(parsed[(fieldstart - 1):fieldend])
            else:
                row = np.array(parsed[(fieldstart - 1):])

            data = np.zeros(shape=(abs(j2 - j1), len(row)))
            data[0, :] = row

            for i, line in enumerate(list(lines[1:])):
                parsed = self._parse_line().parseString(line)

                if fieldend:
                    try:
                        data[i + 1, :] = np.array(parsed[(fieldstart - 1):fieldend])
                    except Exception:
                        print(data)
                else:
                    data[i + 1, :] = np.array(parsed[(fieldstart - 1):])

        return data

    def _parse_line(self):
        """
        Parse a single data line that may contain string or numerical data.

        Float and Int 'words' are converted to their appropriate type.
        Exponentiation is supported, as are NaN and Inf.

        Returns
        -------
        <ParserElement>
            the parsed line.
        """
        return self.line_parse_token

    def _reset_tokens(self):
        """
        Set up the tokens for pyparsing.
        """
        # Somewhat of a hack, but we can only use printables if the delimiter is
        # just whitespace. Otherwise, some seprators (like ',' or '=') potentially
        # get parsed into the general string text. So, if we have non whitespace
        # delimiters, we need to fall back to just alphanums, and then add in any
        # missing but important symbols to parse.
        if self._delimiter.isspace():
            textchars = printables
        else:
            textchars = alphanums

            symbols = ['.', '/', '+', '*', '^', '(', ')', '[', ']', '=',
                       ':', ';', '?', '%', '&', '!', '#', '|', '<', '>',
                       '{', '}', '-', '_', '@', '$', '~']

            for symbol in symbols:
                if symbol not in self._delimiter:
                    textchars = textchars + symbol

        digits = Word(nums)
        dot = "."
        sign = oneOf("+ -")
        ee = CaselessLiteral('E') | CaselessLiteral('D')

        num_int = ToInteger(Combine(Optional(sign) + digits))

        num_float = ToFloat(Combine(
            Optional(sign) +
            ((digits + dot + Optional(digits)) | (dot + digits)) +
            Optional(ee + Optional(sign) + digits)
        ))

        # special case for a float written like "3e5"
        mixed_exp = ToFloat(Combine(digits + ee + Optional(sign) + digits))

        nan = (ToInf(oneOf("Inf -Inf")) |
               ToNan(oneOf("NaN nan NaN%  NaNQ NaNS qNaN sNaN 1.#SNAN 1.#QNAN -1.#IND")))

        string_text = Word(textchars)

        self.line_parse_token = (OneOrMore((nan | num_float | mixed_exp | num_int | string_text)))
