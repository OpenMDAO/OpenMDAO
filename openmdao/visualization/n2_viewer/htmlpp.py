import base64
import re
import zlib
import json
from openmdao.utils.general_utils import default_noraise
from pathlib import Path

class HtmlPreprocessor():
    """
    Recursively substitute and insert source files to produce a single HTML file.

    Source files contain text with directives in the form: <<directive value_arg>>

    Recognized directives are:
    <<hpp_insert path/to/file [compress]>>: Paste path/to/file verbatim into the surrounding text
    <<hpp_script path/to/script>>: Paste path/to/script into the text inside a <script> tag
    <<hpp_style path/to/css>>: Paste path/to/css into the text inside a <style> tag
    <<hpp_bin2b64 path/to/file>>: Convert a binary file to a b64 string and insert it
    <<hpp_pyvar variable_name [compress]>>: Insert the string value of the named Python variable.
        If the referenced variable is non-primitive, it's converted to JSON.

    If the compress option is used, the replacement content will be compressed and converted to
    a base64 string. It's up to the JavaScript code to decode and uncompress it.

    Commented directives (//, /* */, or <!-- -->) will replace the entire comment.
    When a directive is commented, it can only be on a single line or the comment-ending
    chars will not be replaced.

    Nothing is written until every directive has been successfully processed.
    """

    def __init__(self, start_filename, output_filename, allow_overwrite = False, var_dict = None):
        """
        Configure the preprocessor and validate file paths.

        Parameters
        ----------
        start_filename: str
            The file to begin processing from.
        output_filename: str
            The path to the new merged HTML file.
        allow_overwrite: bool
            If true, overwrite the output file if it exists.
        var_dict: dict
            Dictionary of variable names and values that hpp_pyvar will reference.
        """
        path = Path(start_filename)
        if path.is_file() is False:
            raise FileNotFoundError(f"Error: {start_filename} not found")

        path = Path(output_filename)
        if path.is_file() and not allow_overwrite:
            raise FileExistsError(f"Error: {output_filename} already exists")

        self.start_filename = start_filename
        self.output_filename = output_filename
        self.allow_overwrite = allow_overwrite
        self.var_dict = var_dict

    def load_text_file(self, filename) -> str:
        """
        Open and read the specified text file.

        Parameters
        ----------
        filename: str
            The path to the text file to read.

        Returns
        -------
        str
            The complete contents of the text file.
        """
        with open(filename, 'r') as f:
            file_contents = str(f.read())

        return file_contents

    def load_bin_file(self, filename) -> str:
        """
        Open and read the specified binary file, converting it to a base64 string.
        Parameters
        ----------
        filename: str
            The path to the binary file to read.

        Returns
        -------
        str
            The complete contents represented as a base64 string.
        """
        with open(filename, 'rb') as f:
            file_contents = str(base64.b64encode(f.read()).decode("ascii"))

        return file_contents

    def parse_contents(self, contents) -> str:
        """
        Find the preprocessor directives in the file and replace them with the desired content.

        Will recurse if directives are also found in the new content.

        Parameters
        ----------
        contents: str
            The contents of a preloaded text file.

        Returns
        -------
        str
            The complete contents represented as a base64 string.
        """
        # Find all possible keywords:
        keyword_regex = '(//|/\*|<\!--)?\s*<<\s*hpp_(insert|script|style|bin2b64|pyvar)\s+(\S+)(\s+compress)?\s*>>(\*/|-->)?'
        matches = re.finditer(keyword_regex, contents)

        for found_directive in matches:
            full_match = found_directive.group(0)
            # Group 1 is the possible comment chars
            keyword = found_directive.group(2)
            arg = found_directive.group(3)
            compress_selected = not (found_directive.group(4) is None)
            do_compress = False # Change below with directives where it's allowed

            if keyword == 'insert':
                # Recursively insert a plain text file which may also have hpp directives
                new_content = self.parse_contents(self.load_text_file(arg))

            elif keyword == 'script':
                # Recursively insert a JavaScript file which may also have hpp directives
                new_content = f'<script type="text/javascript">\n' + \
                    self.parse_contents(self.load_text_file(arg)) + f'\n</script>'
                do_compress = True if compress_selected else False

            elif keyword == 'style':
                # Recursively insert a CSS file which may also have hpp directives
                new_content = f'<style type="text/css">\n' + \
                    self.parse_contents(self.load_text_file(arg)) + f'\n</style>'
                
            elif keyword == 'bin2b64':
                new_content = self.load_bin_file(arg)

            elif keyword == 'pyvar':
                if arg in self.var_dict:
                    val = self.var_dict[arg]
                    if type(val) in (str, bool, int, float): # Use string representations of primitive types
                        new_content = str(self.var_dict[arg])
                        do_compress = True if compress_selected else False
                    else:
                        raw_data = json.dumps(val, default=default_noraise) # .encode('utf8')
                        if compress_selected:
                            new_content = str(base64.b64encode(zlib.compress(raw_data.encode('utf8'))).decode("ascii"))
                        else:
                            new_content = raw_data

                else:
                    raise ValueError(f"Variable substitution requested for undefined variable {arg}")

            else:
                # Bad keyword
                raise ValueError(f"Unrecognized HTML preprocessor directive hpp_{keyword} encountered")
            
            if do_compress:
                new_content = str(base64.b64encode(zlib.compress(new_content)).decode("ascii"))

            # Replace the directive with the content:
            contents = re.sub(full_match, new_content, contents)

        return contents

    def run(self) -> None:
        """
        Initiate the preprocessor, then save the result as a new file.
        """
        new_html_content = self.parse_contents(self.load_text_file(self.start_filename))

        path = Path(self.output_filename)
        if path.is_file() and not self.allow_overwrite:
            raise FileExistsError(f"Error: {self.output_filename} already exists")

        output_file = open(self.output_filename, "w")
        output_file.write(new_html_content)
        output_file.close()
