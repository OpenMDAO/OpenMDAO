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
    s = re.sub(r'(\[)(?=[^]]*\])', '\\[', s)

    if tags is None:
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
