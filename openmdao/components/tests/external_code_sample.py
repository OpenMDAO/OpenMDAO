import os
import time
import argparse

def main():
    """ Just an external program for testing ExternalCode. """

    parser = argparse.ArgumentParser()
    parser.add_argument("output_filename")
    parser.add_argument("-e", "--write_test_env_var", help="Write the value of TEST_ENV_VAR to the file",
                    action="store_true", default=False)
    parser.add_argument("-d", "--delay", type=float,
                    help="time in seconds to delay")

    args = parser.parse_args()

    if args.delay:
        if args.delay < 0:
            raise ValueError('delay must be >= 0')
        time.sleep(args.delay)

    with open(args.output_filename, 'w') as out:
        out.write("test data\n")
        if args.write_test_env_var:
            out.write("%s\n" % os.environ['TEST_ENV_VAR'])

    return 0

if __name__ == '__main__': # pragma no cover
    main()

