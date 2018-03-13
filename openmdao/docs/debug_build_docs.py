# using this file to build the docs allows you run the doc build in a debugger.
if __name__ == '__main__':
    import sys
    import os
    import sphinx
    #os.system('make clean')
    testargs = '-b html -d _build/doctrees . _build/html -t "usr"'.split(' ')
    sys.argv.extend(testargs)

    sys.exit(sphinx.main())
