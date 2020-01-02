# a short sphinx extension to take care of hyperlinking in docstrings
# where a syntax of <Class> is employed.
import openmdao
import pkgutil
import inspect
import re
from openmdao.docs.config_params import IGNORE_LIST

# first, we will need a dict that contains full pathnames to every class.
# we construct that here, once, then use it for lookups in om_process_docstring
package = openmdao

om_classes = {}

def build_dict():
    global om_classes
    for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,
                                                        prefix=package.__name__ + '.',
                                                        onerror=lambda x: None):
        if not ispkg:
            if 'docs' not in modname:
                if any(ignore in modname for ignore in IGNORE_LIST):
                    continue
                module = importer.find_module(modname).load_module(modname)
                for classname, class_object in inspect.getmembers(module, inspect.isclass):
                    if class_object.__module__.startswith("openmdao"):
                        om_classes[classname] = class_object.__module__ + "." + classname


def om_process_docstring(app, what, name, obj, options, lines):
    """
    our process_docstring
    """
    global om_classes
    if not om_classes:
        build_dict()

    for i in range(len(lines)):
        # create a regex pattern to match <linktext>
        pat = r'(<.*?>)'
        # find all matches of the pattern in a line
        match = re.findall(pat, lines[i])
        if match:
            for ma in match:
                # strip off the angle brackets `<>`
                m = ma[1:-1]
                # to get rid of bad matches in OrderedDict.set_item
                if m == "==":
                    continue
                # if there's a dot in the pattern, it's a method
                # e.g. <classname.method_name>
                if '.' in m:
                    # need to grab the class name and method name separately
                    split_match = m.split('.')
                    justclass = split_match[0]  # class
                    justmeth = split_match[1]   # method
                    if justclass in om_classes:
                        classfullpath = om_classes[justclass]
                        # construct a link  :meth:`class.method <openmdao.core.class.method>`
                        link = ":meth:`" + m + " <" + classfullpath + "." + justmeth + ">`"
                        # replace the <link> text with the constructed line.
                        lines[i] = lines[i].replace(ma, link)
                    else:
                        # the class isn't in the class table!
                        print("WARNING: {} not found in dictionary of OpenMDAO methods".format
                              (justclass))
                        # replace instances of <class> with just class in docstring
                        # (strip angle brackets)
                        lines[i] = lines[i].replace(ma, m)
                # otherwise, it's a class
                else:
                    if m in om_classes:
                        classfullpath = om_classes[m]
                        lines[i] = lines[i].replace(ma, ":class:`~" + classfullpath + "`")
                    else:
                        # the class isn't in the class table!
                        print("WARNING: {} not found in dictionary of OpenMDAO classes"
                              .format(m))
                        # replace instances of <class> with class in docstring
                        # (strip angle brackets)
                        lines[i] = lines[i].replace(ma, m)


# This is the crux of the extension--connecting an internal
# Sphinx event, "autodoc-process-docstring" with our own custom function.
def setup(app):
    """
    """
    app.connect('autodoc-process-docstring', om_process_docstring)
