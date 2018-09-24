"""
A script for postprocessing an 'openmdao trace -m' output file and plotting the memory usage
vs. time, with highligting of a specified function.
"""

if __name__ == '__main__':
    import sys
    import argparse
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--func', action='store', dest='func',
                        help='Plot occurrences of func along timeline.')
    parser.add_argument('file', metavar='file', nargs='+',
                        help='Python file to profile.')

    options = parser.parse_args()

    fig, ax = plt.subplots()

    changes = []

    for i, arg in enumerate(options.file):
        delta = []
        total = []
        elapsed = []
        calls = []
        call_times = []

        with open(arg, 'r') as f:
            for line in f:
                line = line.lstrip()
                if line.startswith('<--'):
                    parts = line.split()
                    timestamp = float(parts[3].split(')')[0])

                    fullname = parts[1]
                    fname = parts[1].rsplit('.', 1)[-1]

                    if len(parts) > 7:
                        delta.append(float(parts[8]))
                    else:
                        delta.append(0.0)
                    total.append(float(parts[5]))
                    elapsed.append(timestamp)
                    changes.append((delta[-1], elapsed[-1], total[-1], fullname, arg))

                    if fname == options.func:
                        calls.append(float(parts[5]))
                        call_times.append(timestamp)

        ax.plot(elapsed, total, label=arg.rsplit('.', 1)[0])

        # ax.plot(elapsed, total[0] + np.cumsum(delta) / 1024.,
        #         label="%s sum of deltas" % arg.rsplit('.', 1)[0],
        #         linestyle='-.')

    sorted_changes = sorted(changes, key=lambda t: t[0], reverse=True)
    nresults = 50

    for i, (change, ctime, tot, func, fname) in enumerate(sorted_changes):
        print("Change of %+9.2f KB at %8.5f sec in file %s in %s." %
              (change, ctime, fname, func))
        if i == nresults:
            break

    delta_totals = []
    for i, (change, ctime, tot, func, fname) in enumerate(changes):
        if i > 0:
            delta_totals.append((tot - changes[i-1][2], changes[i-1], changes[i]))
        else:
            delta_totals.append((0.0, changes[i], changes[i]))

    for i, (delta, changes1, changes2) in enumerate(sorted(delta_totals, key=lambda t: t[0],
                                                           reverse=True)):
        print("Largest total delta of %+9.2f between %s and %s at %8.5f" %
              (delta, changes1[3], changes2[3], changes2[1]))
        if i == nresults:
            break

    if options.func:
        # apparently if a label starts with '_' matplotlib ignores it
        ax.plot(call_times, calls, 'kd', label=options.func.lstrip('_'))

    ax.legend(loc='lower right')
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('memory (MB)')
    plt.grid(True)
    plt.show()
