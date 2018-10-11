import ast

class ImportScanner(ast.NodeVisitor):
    """
    This node visitor collects all import information from a file.
    """
    def __init__(self):
        self.imports = []

    def visit_Import(self, node):
        """
        This executes every time an "import foo" style import statement
        is parsed.
        """
        for n in node.names:
            self.imports.append((n.name, n.asname, None))

    def visit_ImportFrom(self, node):
        """
        This executes every time a "from foo import bar" style import
        statement is parsed.
        """
        self.imports.append((node.module, None, [(n.name, n.asname) for n in node.names]))

    def get_import_lines(self):
        """
        Return source lines containing all of the imports.
        """
        lines = []
        for mod, alias, lst in self.imports:
            if lst is None:  # regular import
                if alias is None:
                    lines.append("import %s" % mod)
                else:
                    lines.append("import %s as %s" % (mod, alias))
            else:  # a 'from' import
                froms = []
                for imp, imp_alias in lst:
                    if imp_alias is None:
                        froms.append(imp)
                    else:
                        froms.append("%s as %s" % (imp, imp_allias))
                lines.append("from %s import %s" % (mod, ', '.join(froms)))

        return lines


if __name__ == '__main__':
    import sys
    fname = sys.argv[1]
    with open(fname, "r") as f:
        contents = f.read()
    node = ast.parse(contents, fname)
    imp_scanner = ImportScanner()
    imp_scanner.visit(node)

    for l in imp_scanner.get_import_lines():
        print(l)
