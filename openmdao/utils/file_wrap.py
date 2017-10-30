"""
A collection of utilities for file wrapping.

Note: This is a work in progress.
"""

from __future__ import print_function

import re
from six.moves import range

from pyparsing import CaselessLiteral, Combine, OneOrMore, Optional, \
    TokenConverter, Word, nums, oneOf, printables, ParserElement, alphanums

import numpy as np

# public symbols
__all__ = ['InputFileGenerator', 'FileParser']


def _getformat(val):
    # Returns the output format for a floating point number.
    # The general format is used with 16 places of accuracy, except for when
    # the floating point value is an integer, in which case a decimal point
    # followed by a single zero is used.

    if int(val) == val:
        return "%.1f"
    else:
        return "%.16g"


class _SubHelper(object):
    """Replaces file text at the correct word location in a line. This
    class contains the Helper Function that is passed to re.sub, etc."""

    def __init__(self):

        self.newtext = ""
        self.replace_location = 0
        self.current_location = 0
        self.counter = 0
        self.start_location = 0
        self.end_location = 0

    def set(self, newtext, location):
        """Sets a new word location and value for replacement."""

        self.newtext = newtext
        self.replace_location = location
        self.current_location = 0

    def set_array(self, newtext, start_location, end_location):
        """For an array, sets a new starting location, ending location, and
        value for replacement."""

        self.newtext = newtext
        self.start_location = start_location
        self.end_location = end_location
        self.current_location = 0

    def replace(self, text):
        """This function should be passed to re.sub.
        Outputs newtext if current_location = replace_location
        Otherwise, outputs the input text."""

        self.current_location += 1

        if self.current_location == self.replace_location:
            if isinstance(self.newtext, float):
                return _getformat(self.newtext) % self.newtext
            else:
                return str(self.newtext)
        else:
            return text.group()

    def replace_array(self, text):
        """This function should be passed to re.sub.
        Outputs newtext if current_location = replace_location
        Otherwise, outputs the input text."""

        self.current_location += 1
        end = len(self.newtext)

        if self.current_location >= self.start_location and \
           self.current_location <= self.end_location and \
           self.counter < end:
            if isinstance(self.newtext[self.counter], float):
                val = self.newtext[self.counter]
                newval = _getformat(val) % val
            else:
                newval = str(self.newtext[self.counter])
            self.counter += 1
            return newval
        else:
            return text.group()


class ToInteger(TokenConverter):
    """Converter for PyParsing that is used to turn a token into an int."""
    def postParse(self, instring, loc, tokenlist):
        """Converter to make token into an integer."""
        return int(tokenlist[0])


class ToFloat(TokenConverter):
    """Converter for PyParsing that is used to turn a token into a float."""
    def postParse(self, instring, loc, tokenlist):
        """Converter to make token into a float."""
        return float(tokenlist[0].replace('D', 'E'))


class ToNan(TokenConverter):
    """Converter for PyParsing that is used to turn a token into Python nan."""
    def postParse(self, instring, loc, tokenlist):
        """Converter to make token into Python nan."""
        return float('nan')


class ToInf(TokenConverter):
    """Converter for PyParsing that is used to turn a token into Python inf."""
    def postParse(self, instring, loc, tokenlist):
        """Converter to make token into Python inf."""
        return float('inf')


class InputFileGenerator(object):
    """Utility to generate an input file from a template.
    Substitution of values is supported. Data is located with
    a simple API."""

    def __init__(self):

        self.template_filename = []
        self.output_filename = []

        self.delimiter = " "
        self.reg = re.compile('[^ \n]+')

        self.data = []
        self.current_row = 0
        self.anchored = False

    def set_template_file(self, filename):
        """Set the name of the template file to be used The template
        file is also read into memory when this method is called.

        Args
        ----
        filename : string
            Name of the template file to be used."""

        self.template_filename = filename

        templatefile = open(filename, 'r')
        self.data = templatefile.readlines()
        templatefile.close()

    def set_generated_file(self, filename):
        """Set the name of the file that will be generated.

        Args
        ----
        filename : string
            Name of the input file to be generated."""

        self.output_filename = filename

    def set_delimiters(self, delimiter):
        """Lets you change the delimiter that is used to identify field
        boundaries.

        Args
        ----
        delimiter : str
            A string containing characters to be used as delimiters."""

        self.delimiter = delimiter
        self.reg = re.compile('[^' + delimiter + '\n]+')

    def mark_anchor(self, anchor, occurrence=1):
        """Marks the location of a landmark, which lets you describe data by
        relative position. Note that a forward search begins at the old anchor
        location. If you want to restart the search for the anchor at the file
        beginning, then call ``reset_anchor()`` before ``mark_anchor``.

        Args
        ----
        anchor : string
            The text you want to search for.

        occurrence : integer, optional
            Find nth instance of text; default is 1 (first). Use -1 to
            find last occurrence. Reverse searches always start at the end
            of the file no matter the state of any previous anchor."""

        if not isinstance(occurrence, int):
            raise ValueError("The value for occurrence must be an integer")

        instance = 0
        if occurrence > 0:
            count = 0
            max_lines = len(self.data)
            for index in range(self.current_row, max_lines):
                line = self.data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only after the anchor.
                if count == 0 and self.anchored:
                    line = line.split(anchor)[-1]

                if line.find(anchor) > -1:

                    instance += 1
                    if instance == occurrence:
                        self.current_row += count
                        self.anchored = True
                        return

                count += 1

        elif occurrence < 0:
            max_lines = len(self.data) - 1
            count = max_lines
            for index in range(max_lines, -1, -1):
                line = self.data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only before the anchor.
                if count == max_lines and self.anchored:
                    line = line.split(anchor)[0]

                if line.find(anchor) > -1:
                    instance += -1
                    if instance == occurrence:
                        self.current_row = count
                        self.anchored = True
                        return

                count -= 1
        else:
            raise ValueError("0 is not valid for an anchor occurrence.")

        raise RuntimeError("Could not find pattern %s in template file %s" %
                           (anchor, self.template_filename))

    def reset_anchor(self):
        """Resets anchor to the beginning of the file."""

        self.current_row = 0
        self.anchored = False

    def transfer_var(self, value, row, field):
        """Changes a single variable in the template relative to the
        current anchor.

        Args
        ----
        value : float, integer, bool, string
            New value to set at the location.

        row : integer
            Number of lines offset from anchor line (0 is anchor line).
            This can be negative.

        field : integer
            Which word in line to replace, as denoted by delimiter(s)
        """

        j = self.current_row + row
        line = self.data[j]

        sub = _SubHelper()
        sub.set(value, field)
        newline = re.sub(self.reg, sub.replace, line)

        self.data[j] = newline

    def transfer_array(self, value, row_start, field_start, field_end,
                       row_end=None, sep=", "):
        """Changes the values of an array in the template relative to the
        current anchor. This should generally be used for one-dimensional
        or free form arrays.

        Args
        ----
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
            Separator to use if we go beyond the template."""

        # Simplified input for single-line arrays
        if row_end is None:
            row_end = row_start

        sub = _SubHelper()
        for row in range(row_start, row_end + 1):

            j = self.current_row + row
            line = self.data[j]

            if row == row_end:
                f_end = field_end
            else:
                f_end = 99999
            sub.set_array(value, field_start, f_end)
            field_start = 0

            newline = re.sub(self.reg, sub.replace_array, line)
            self.data[j] = newline

        # Sometimes an array is too large for the example in the template
        # This is resolved by adding more fields at the end
        if sub.counter < len(value):
            for val in value[sub.counter:]:
                newline = newline.rstrip() + sep + str(val)

            self.data[j] = newline

        # Sometimes an array is too small for the template
        # This is resolved by removing fields
        elif sub.counter > len(value):

            # TODO - Figure out how to handle this.
            # Ideally, we'd remove the extra field placeholders
            raise ValueError("Array is too small for the template.")

        self.data[j] += "\n"

    def transfer_2Darray(self, value, row_start, row_end, field_start, field_end):
        """Changes the values of a 2D array in the template relative to the
        current anchor. This method is specialized for 2D arrays, where each
        row of the array is on its own line.

        Args
        ----
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

            j = self.current_row + row
            line = self.data[j]

            sub.set_array(value[i, :], field_start, field_end)

            newline = re.sub(self.reg, sub.replace_array, line)
            self.data[j] = newline

            sub.current_location = 0
            sub.counter = 0
            i += 1

        # TODO - Note, we currently can't handle going beyond the end of
        #        the template line

    def clearline(self, row):
        """Replace the contents of a row with the newline character.

        Args
        ----
        row : integer
            Row number to clear, relative to current anchor."""

        self.data[self.current_row + row] = "\n"

    def generate(self):
        """Use the template file to generate the input file."""

        infile = open(self.output_filename, 'w')
        infile.writelines(self.data)
        infile.close()


class FileParser(object):
    """Utility to locate and read data from a file.

    Args
    ----
    end_of_line_comment_char : string, optional
        Specify an end-of-line comment character to be ignored (e.g., Python
        supports in-line comments with "#".)

    full_line_comment_char : string, optional
        Sepcify a comment character that signifies a line should be skipped.
    """

    def __init__(self, end_of_line_comment_char=None, full_line_comment_char=None):

        self.filename = []
        self.data = []

        self.delimiter = " \t"
        self.end_of_line_comment_char = end_of_line_comment_char
        self.full_line_comment_char = full_line_comment_char

        self.current_row = 0
        self.anchored = False
        self.set_delimiters(self.delimiter)

    def set_file(self, filename):
        """Set the name of the file that will be generated.

        Args
        ----
        filename : string
            Name of the input file to be generated."""

        self.filename = filename

        inputfile = open(filename, 'r')
        if not self.end_of_line_comment_char and not self.full_line_comment_char:
            self.data = inputfile.readlines()
        else:
            self.data = []
            for line in inputfile:
                if line[0] == self.full_line_comment_char:
                    continue
                self.data.append(line.split(self.end_of_line_comment_char)[0])
        inputfile.close()

    def set_delimiters(self, delimiter):
        """Lets you change the delimiter that is used to identify field
        boundaries.

        Args
        ----
        delimiter : string
            A string containing characters to be used as delimiters. The
            default value is ' \t', which means that spaces and tabs are not
            taken as data but instead mark the boundaries. Note that the
            parser is smart enough to recognize characters within quotes as
            non-delimiters."""

        self.delimiter = delimiter
        if delimiter != "columns":
            ParserElement.setDefaultWhitespaceChars(str(delimiter))
        self._reset_tokens()

    def mark_anchor(self, anchor, occurrence=1):
        """Marks the location of a landmark, which lets you describe data by
        relative position. Note that a forward search begins at the old anchor
        location. If you want to restart the search for the anchor at the file
        beginning, then call ``reset_anchor()`` before ``mark_anchor``.

        Args
        ----
        anchor : str
            The text you want to search for.

        occurrence : integer
            Find nth instance of text; default is 1 (first). Use -1 to
            find last occurrence. Reverse searches always start at the end
            of the file no matter the state of any previous anchor."""

        if not isinstance(occurrence, int):
            raise ValueError("The value for occurrence must be an integer")

        instance = 0
        if occurrence > 0:
            count = 0
            max_lines = len(self.data)
            for index in range(self.current_row, max_lines):
                line = self.data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only after the anchor.
                if count == 0 and self.anchored:
                    line = line.split(anchor)[-1]

                if anchor in line:

                    instance += 1
                    if instance == occurrence:
                        self.current_row += count
                        self.anchored = True
                        return

                count += 1

        elif occurrence < 0:
            max_lines = len(self.data) - 1
            count = max_lines
            for index in range(max_lines, -1, -1):
                line = self.data[index]

                # If we are marking a new anchor from an existing anchor, and
                # the anchor is mid-line, then we still search the line, but
                # only before the anchor.
                if count == max_lines and self.anchored:
                    line = line.split(anchor)[0]

                if anchor in line:
                    instance += -1
                    if instance == occurrence:
                        self.current_row = count
                        self.anchored = True
                        return

                count -= 1
        else:
            raise ValueError("0 is not valid for an anchor occurrence.")

        raise RuntimeError("Could not find pattern %s in output file %s" %
                           (anchor, self.filename))

    def reset_anchor(self):
        """Resets anchor to the beginning of the file."""

        self.current_row = 0
        self.anchored = False

    def transfer_line(self, row):
        """Returns a whole line, relative to current anchor.

        Args
        ----
        row : integer
            Number of lines offset from anchor line (0 is anchor line).
            This can be negative.

        Returns
        -------
            string : line at the location requested"""

        return self.data[self.current_row + row].rstrip()

    def transfer_var(self, row, field, fieldend=None):
        """Grabs a single variable relative to the current anchor.

        Args
        ----
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
            string : data from the requested location in the file
        """

        j = self.current_row + row
        line = self.data[j]

        if self.delimiter == "columns":

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
        """Searches for a key relative to the current anchor and then grabs
        a field from that line.

        You can do the same thing with a call to ``mark_anchor`` and ``transfer_var``.
        This function just combines them for convenience.

        Args
        ----
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
            string : data from the requested location in the file
        """

        if not isinstance(occurrence, int) or occurrence == 0:
            msg = "The value for occurrence must be a nonzero integer"
            raise ValueError(msg)

        instance = 0
        if occurrence > 0:
            row = 0
            for line in self.data[self.current_row:]:
                if line.find(key) > -1:
                    instance += 1
                    if instance == occurrence:
                        break

                row += 1

        elif occurrence < 0:
            row = -1
            for line in reversed(self.data[self.current_row:]):
                if line.find(key) > -1:
                    instance += -1
                    if instance == occurrence:
                        break

                row -= 1

        j = self.current_row + row + rowoffset
        line = self.data[j]

        fields = self._parse_line().parseString(line.replace(key, "KeyField"))

        return fields[field]

    def transfer_array(self, rowstart, fieldstart, rowend=None, fieldend=None):
        """Grabs an array of variables relative to the current anchor.

        Setting the delimiter to 'columns' elicits some special behavior
        from this method. Normally, the extraction process wraps around
        at the end of a line and continues grabbing each field at the start of
        a newline. When the delimiter is set to columns, the parameters
        (rowstart, fieldstart, rowend, fieldend) demark a box, and all
        values in that box are retrieved. Note that standard whitespace
        is the secondary delimiter in this case.

        Args
        ----
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
            string : data from the requested location in the file
        """

        j1 = self.current_row + rowstart

        if rowend is None:
            j2 = j1 + 1
        else:
            j2 = self.current_row + rowend + 1

        if not fieldend:
            raise ValueError("fieldend is missing, currently required")

        lines = self.data[j1:j2]

        data = np.zeros(shape=(0, 0))

        for i, line in enumerate(lines):
            if self.delimiter == "columns":
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
        """Grabs a 2D array of variables relative to the current anchor. Each
        line of data is placed in a separate row.

        If the delimiter is set to 'columns', then the values contained in
        fieldstart and fieldend should be the column number instead of the
        field number.

        Args
        ----
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
            string : data from the requested location in the file
        """

        if fieldend and (fieldstart > fieldend):
            msg = "fieldend must be greater than fieldstart"
            raise ValueError(msg)

        if rowstart > rowend:
            msg = "rowend must be greater than rowstart"
            raise ValueError(msg)

        j1 = self.current_row + rowstart
        j2 = self.current_row + rowend + 1
        lines = list(self.data[j1:j2])

        if self.delimiter == "columns":

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
        """Parse a single data line that may contain string or numerical data.
        Float and Int 'words' are converted to their appropriate type.
        Exponentiation is supported, as are NaN and Inf."""

        return self.line_parse_token

    def _reset_tokens(self):
        """Sets up the tokens for pyparsing."""

        # Somewhat of a hack, but we can only use printables if the delimiter is
        # just whitespace. Otherwise, some seprators (like ',' or '=') potentially
        # get parsed into the general string text. So, if we have non whitespace
        # delimiters, we need to fall back to just alphanums, and then add in any
        # missing but important symbols to parse.
        if self.delimiter.isspace():
            textchars = printables
        else:
            textchars = alphanums

            symbols = ['.', '/', '+', '*', '^', '(', ')', '[', ']', '=',
                       ':', ';', '?', '%', '&', '!', '#', '|', '<', '>',
                       '{', '}', '-', '_', '@', '$', '~']

            for symbol in symbols:
                if symbol not in self.delimiter:
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
