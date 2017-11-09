if __name__ == '__main__':
    import sys
    import sphinx
    testargs = '-b html -d _build/doctrees . _build/html -t "usr"'.split(' ')
    sys.argv.extend(testargs)

    sys.exit(sphinx.main())
