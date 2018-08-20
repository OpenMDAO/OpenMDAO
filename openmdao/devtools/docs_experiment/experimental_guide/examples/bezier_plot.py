# Adding a comment and a future import to make sure it works.
from __future__ import print_function
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

if __name__ == '__main__':

    Path = mpath.Path

    fig, ax = plt.subplots()
    pp1 = mpatches.PathPatch(
        Path([(0, 0), (1, 0), (1, 1), (0, 0)],
             [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]),
        fc="none", transform=ax.transData)

    print("Here's some output.")

    ax.add_patch(pp1)
    ax.plot([0.75], [0.25], "ro")

    print("Here's more output.")
    ax.set_title('The red point should be on the path')

    plt.show()

    print("Some output\nat the end.")
