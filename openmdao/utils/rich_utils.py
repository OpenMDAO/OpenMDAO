"""Utilities for using rich with OpenMDAO."""

import itertools
import re

try:
    import rich
except ImportError:
    rich = None


def rich_wrap(s, tags=None):
    """
    If rich is available, escape square brackets and wrap the given string in the provided tags.

    If rich is not available, just return the string.

    Parameters
    ----------
    s : str
        The string to be wrapped in rich tags.
    tags : Sequence[str] or set[str], optional
        The rich tags to be wrapped around s. These can either be
        strings.

    Returns
    -------
    str
        The given string wrapped in the provided rich tags.
    """
    if rich is None:
        return s

    # Escape any open square brackets in the string.
    s = re.sub(r'(\[)(?=[^]]*\])', '\\[', str(s))

    if not tags:
        return s

    if isinstance(tags, str):
        tags = {tags}

    def flatten(lst):
        seq = list(itertools.chain.from_iterable(x if isinstance(x, (list, set, tuple))
                                                 else [x] for x in lst))
        return seq

    cmds = sorted(flatten([t if isinstance(t, str) else t.value for t in tags]))
    on = ' '.join(cmds)
    off = '/' + ' '.join(reversed(cmds))
    return f'[{on}]{s}[{off}]'


def strip_formatting(s):
    """
    Remove formatting applied to a string by rich.

    This method is useful when testing the content of rich-formatted output.

    Parameters
    ----------
    s : str
        The string to have its rich formatting removed.

    Returns
    -------
    str
        String s with any ascii formatting sequences removed.
    """
    escape_pattern = r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])'
    return re.sub(escape_pattern, '', s)


def strip_tags(s):
    """
    Remove non-escaped tags from a string.

    This method is useful for determining lengths of rich-formatted text.

    Parameters
    ----------
    s : str
        The string to have its rich formatting removed.

    Returns
    -------
    str
        String s with any rich tags removed.
    """
    escape_pattern = r'(?<!\\)\[/?\w+[^\]]*\]'
    return re.sub(escape_pattern, '', s)
