import os
import sys
import time
import argparse


def main():
    """
    A standalone program for testing ExternalCodeComp.

    Writes "test data" to the specified output file after an optional delay.
    Optionally writes the value of the environment variable "TEST_ENV_VAR"
    to the file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("output_filename")
    parser.add_argument("-e", "--write_test_env_var",
                        help="Write the value of TEST_ENV_VAR to the file",
                        action="store_true", default=False)
    parser.add_argument("-d", "--delay", type=float,
                        help="time in seconds to delay")
    parser.add_argument("-r", "--return_code", type=int,
                        help="value to return as the return code", default=0)

    args = parser.parse_args()

    if args.delay:
        if args.delay < 0:
            raise ValueError('delay must be >= 0')
        time.sleep(args.delay)

    with open(args.output_filename, 'w') as out:
        out.write("test data\n")
        if args.write_test_env_var:
            out.write("%s\n" % os.environ['TEST_ENV_VAR'])

    return args.return_code


if __name__ == '__main__':
    sys.exit(main())
