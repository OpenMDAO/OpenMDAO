"""
Define the HtmlPreprocessor class to generate a single HTML file from many source files.
"""
import base64
import re
import zlib
import json
from pathlib import Path


class HtmlPreprocessor():
    """
    Recursively substitute and insert source files to produce a single HTML file.

    Source files contain text with directives in the form: <<hpp_directive arg [flag1] [flag2]>>

    Parameters
    ----------
    start_filename : str
        The file to begin processing from.
    output_filename : str
        The path to the newly merged HTML file.
    search_path : str[]
        List of directory names to search for files.
    allow_overwrite : bool
        If true, overwrite the output file if it exists.
    var_dict : dict
        Dictionary of variable names and values referenced by hpp_pyvar directives.
    json_dumps_default : function
        Passed to json.dumps() as the "default" parameter that gets
        called for objects that can't be serialized.
    verbose : bool
        If True, print some status messages to stdout.

    Attributes
    ----------
    _start_path : Path
        A Path object to help with testing functions for the provided start_filename.
    _start_filename : str
        The path to the file to begin processing from, normally an HTML file.
    _search_path : Path
        List of Path objects to search for included files.
    _output_filename : str
        The path to the newly merged HTML file.
    _var_dict : dict
        Dictionary of variable names and values referenced by hpp_pyvar directives.
    _allow_overwrite : bool
        If true, overwrite the output file if it exists.
    _verbose : bool
        If True, print some status messages to stdout.
    _loaded_filenames : list
        List of filenames loaded so far, to determine if a file has been already loaded.
    _json_dumps_default : function
        Passed to json.dumps() as the "default" parameter that gets
        called for objects that can't be serialized.

    Notes
    -----
    Recognized Directives:

    :code:`<<hpp_insert path/to/file [compress] [dup]>>`
        Paste :code:`path/to/file` verbatim into the text.

    :code:`<<hpp_script path/to/script [dup]>>`
        Insert :code:`path/to/script` between script tags.

    :code:`<<hpp_style path/to/css [dup]>>`
        Insert :code:`path/to/css` between style tags.

    :code:`<<hpp_bin2b64 path/to/file [dup]>>`
        Convert a binary file to a b64 string and insert it.

    :code:`<<hpp_pyvar variable_name [compress]>>`
        Insert the string value of the named Python variable. If the referenced
        variable is non-primitive, it's converted to JSON.

    Flags:

    :code:`compress` :
        The replacement content will be compressed and converted to
        a base64 string. It's up to the JavaScript code to decode and uncompress it.
    :code:`dup` :
        If a file has already been included once, it will be ignored on subsequent inclusions
        unless the dup flag is used.

    Commented directives (:code:`//`, :code:`/* */`, or :code:`<!-- -->`) will replace the
    entire comment. When a directive is commented, it can only be on a single line or the
    comment-ending characters will not be replaced.

    All paths in the directives are relative to the directory that the start file
    is located in (or if start_path was specified) unless it is absolute.

    Nothing is written until every directive has been successfully processed.
    """

    def __init__(self, start_filename: str, output_filename: str, search_path=[],
                 allow_overwrite=False, var_dict: dict = None, json_dumps_default=None,
                 verbose=False):
        """
        Configure the preprocessor and validate file paths.
        """
        self._start_path = Path(start_filename)

        if self._start_path.is_file() is False:
            raise FileNotFoundError(f"Error: {self._start_path} not found")

        self._search_path = []
        for path_name in search_path:
            self._search_path.append(Path(path_name))

        # Put location of start file after manually-specified paths:
        self._search_path.append(self._start_path.resolve().parent)

        # Current folder:
        self._search_path.append(Path('.'))

        output_path = Path(output_filename)
        if output_path.is_file() and not allow_overwrite:
            raise FileExistsError(f"Error: {output_filename} already exists")

        self._start_filename = start_filename
        self._output_filename = output_filename
        self._allow_overwrite = allow_overwrite
        self._var_dict = var_dict
        self._json_dumps_default = json_dumps_default
        self._verbose = verbose

        # Keep track of filenames already loaded, to make sure
        # we don't unintentionally include the exact same file twice.
        self._loaded_filenames = []

        self.msg("HtmlProcessor object created.")

    def find_file(self, filename: str) -> str:
        """
        Check specified locations for the provided filename.

        Parameters
        ----------
        filename : str
            The path to the text file to locate.

        Returns
        -------
        str
            The full path to the existing file if found.
        """
        file_path = Path(filename)
        if file_path.is_absolute():
            if file_path.is_file():
                return filename
        else:
            for path in self._search_path:
                test_path = Path(path / filename)
                if test_path.is_file():
                    return test_path

        raise FileNotFoundError(f"Error: {filename} not found")

    def load_file(self, filename: str, rlvl=0, binary=False, allow_dup=False) -> str:
        """
        Open and read the specified text file.

        Parameters
        ----------
        filename : str
            The path to the text file to read.
        rlvl : int
            Recursion level to help with indentation when verbose is enabled.
        binary : bool
            True if the file is to be opened in binary mode and converted to a base64 str.
        allow_dup : bool
            If False, return an empty string for a filename that's been previously loaded.

        Returns
        -------
        str
            The complete contents of the file.
        """
        pathname = self.find_file(filename)

        if pathname in self._loaded_filenames and not allow_dup:
            self.msg(f"Ignoring previously-loaded file {filename}.", rlvl)
            return ""

        self._loaded_filenames.append(pathname)
        self.msg(f"Loading file {pathname}.", rlvl)

        if binary:
            with open(pathname, 'rb') as f:
                file_contents = str(base64.b64encode(f.read()).decode('UTF-8'))
        else:
            with open(pathname, 'r', encoding='UTF-8') as f:
                file_contents = str(f.read())

        return file_contents

    def msg(self, msg: str, rlvl=0):
        """
        Print a message to stdout if self.verbose is True.

        Parameters
        ----------
        msg : str
            The message to print.
        rlvl : int
            Recursion level to help with indentation when verbose is enabled.
        """
        if self._verbose:
            print(rlvl * '--' + msg)

    def parse_contents(self, contents: str, rlvl=0) -> str:
        """
        Find the preprocessor directives in the file and replace them with the desired content.

        Will recurse if directives are also found in the new content.

        Parameters
        ----------
        contents : str
            The contents of a preloaded text file.
        rlvl : int
            Recursion level to help with indentation when verbose is enabled.

        Returns
        -------
        str
            The complete contents represented as a base64 string.
        """
        # Find all possible keywords:
        keyword_regex = \
            r'(//|/\*|<\!--)?\s*<<\s*hpp_(\S+)\s+(\S+)(\s+compress|\s+dup)*\s*>>(\*/|-->)?'
        matches = re.finditer(keyword_regex, contents)
        rlvl += 1
        new_content = None

        for found_directive in matches:

            full_match = found_directive.group(0)
            comment_start = found_directive.group(1)
            keyword = found_directive.group(2)
            arg = found_directive.group(3)

            flags = {'compress': False, 'dup': False}
            if found_directive.group(4) is not None:
                flags['compress'] = True if 'compress' in found_directive.group(4) else False
                flags['dup'] = True if 'dup' in found_directive.group(4) else False

            do_compress = False  # Change below with directives where it's allowed

            self.msg(f"Handling {keyword} directive.", rlvl)

            if keyword == 'insert':
                # Recursively insert a plain text file which may also have hpp directives
                new_content = self.parse_contents(self.load_file(arg, rlvl=rlvl,
                                                  allow_dup=flags['dup']), rlvl)

                if new_content != "":
                    do_compress = True if flags['compress'] else False

            elif keyword == 'script':
                # Recursively insert a JavaScript file which may also have hpp directives
                new_content = self.parse_contents(self.load_file(arg, rlvl=rlvl,
                                                  allow_dup=flags['dup']), rlvl)

                if new_content != "":
                    new_content = f'<script type="text/javascript">\n{new_content}\n</script>'

            elif keyword == 'style':
                # Recursively insert a CSS file which may also have hpp directives
                new_content = '<style type="text/css">\n' + \
                              self.parse_contents(self.load_file(arg, rlvl=rlvl,
                                                  allow_dup=flags['dup']), rlvl) + f'\n</style>'

            elif keyword == 'bin2b64':
                new_content = self.load_file(arg, binary=True, allow_dup=flags['dup'], rlvl=rlvl)

            elif keyword == 'pyvar':
                if arg in self._var_dict:
                    val = self._var_dict[arg]
                    if type(val) in (str, bool, int, float):
                        # Use string representations of primitive types
                        new_content = str(self._var_dict[arg])
                        do_compress = True if flags['compress'] else False
                    else:
                        raw_data = json.dumps(val, default=self._json_dumps_default)
                        if flags['compress']:
                            self.msg("Compressing content.", rlvl)
                            compressed_content = zlib.compress(raw_data.encode('UTF-8'))
                            new_content = str(base64.b64encode(compressed_content).decode("UTF-8"))
                        else:
                            new_content = raw_data

                else:
                    raise ValueError(f"Can't substitute for undefined variable {arg}")

            else:
                # Bad keyword
                raise ValueError(f"Unknown HTML preprocessor directive hpp_{keyword}")

            if do_compress:
                self.msg("Compressing new content.", rlvl)
                new_content = str(base64.b64encode(zlib.compress(new_content)).decode("UTF-8"))

            if new_content is not None:
                self.msg(f"Replacing directive '{full_match}' with new content.", rlvl)
                # contents = re.sub(full_match, new_content, contents)
                contents = contents.replace(full_match, new_content)

        return contents

    def run(self) -> None:
        """
        Initialize the preprocessor, then save the result as a new file.
        """
        new_html_content = self.parse_contents(self.load_file(self._start_filename))

        path = Path(self._output_filename)
        if path.is_file() and not self._allow_overwrite:
            raise FileExistsError(f"Error: {self._output_filename} already exists")

        output_file = open(self._output_filename, "w", encoding='UTF-8')
        output_file.write(new_html_content)
        output_file.close()
