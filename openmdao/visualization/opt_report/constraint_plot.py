"""
Function to show a plot where a value is relative to constraints
"""
from functools import partial

from matplotlib import pyplot as plt, patches
import matplotlib as mpl

import numpy as np

from openmdao.core.constants import INF_BOUND

# https://jfly.uni-koeln.de/color/
# _in_bound_color = 'green'
_in_bound_color = (0., 0.61960, 0.45098,0.2)
# _out_of_bound_color = 'red'
_out_of_bound_color = (0.83529,0.36862,0.)
# _out_of_bound_color = 'white'
_plot_x_max = 1.0
_plot_y_max = 1.0
_plot_y_margin = 0.2
_lower_plot = _plot_x_max / 3.
_upper_plot = 2 * _plot_x_max / 3.
_pointer_half_width = 0.04
_pointer_height = 0.4
_text_height = 0.3
_font_size = 5
_ellipse_width = 0.01
_ellipse_height = 0.15
_pointer_line_width = 0.05
_relative_diff_from_bound = 1e-4
_near_bound_highlight_half_width  = 0.05
_near_bound_highlight_half_width_y_min = 0.0
_near_bound_highlight_half_width_y_max = _plot_y_max
_near_bound_highlight_alpha = 0.7
_equality_bound_width = 0.01

_out_of_bound_hatch_pattern = 'xxxxx'
_out_of_bound_hatch_color = (0.,0.,0.)
_out_of_bound_hatch_width = 0.3


def _val_to_plot_coord(value, lower, upper):
    # need to get function that maps actual values to 0.0 to 1.0
    # where lower maps to 1./3 and upper to 2/3
    plot_coord = 1./3. + (value - lower) / (upper - lower) * 1./3.
    return plot_coord


# def _draw_in_or_out_bound_section(ax, x_left, width, color):
#     rectangle = patches.Rectangle((x_left, 0), width, _plot_y_max, facecolor=color, hatch='xxxxx')
#     ax.add_patch(rectangle)

def _draw_in_or_out_bound_section(ax, x_left, width, is_in_bound):
    if is_in_bound:
        color = _in_bound_color
        hatch=None
    else:
        color = _out_of_bound_color
        hatch = _out_of_bound_hatch_pattern
    rectangle = patches.Rectangle((x_left, 0), width, _plot_y_max, facecolor=color, hatch=hatch)
    ax.add_patch(rectangle)

def _draw_bound_highlight(ax, x):
    rectangle = patches.Rectangle((
        x - _near_bound_highlight_half_width, _near_bound_highlight_half_width_y_min),
        2 * _near_bound_highlight_half_width, _near_bound_highlight_half_width_y_max,
        edgecolor='black', facecolor='yellow', alpha=_near_bound_highlight_alpha)
    ax.add_patch(rectangle)


def _draw_ellipsis(ax, x_left):
    # Draw three dots as an ellipsis to show that the value is beyond
    #   either the left or right edge of the plot
    for i in [5, 6, 7]:
        circle = patches.Ellipse((x_left + i * _lower_plot / 12., 0),
                                 _ellipse_width, _ellipse_height,
                                 facecolor=_out_of_bound_color)
        ax.add_patch(circle)

def _draw_boundary_label(ax, pointer_plot_coord, s):
    ax.text(pointer_plot_coord, _plot_y_max + _text_height,
             s,
             horizontalalignment='center',
             verticalalignment='bottom')

def _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value):
    pts = np.array([
        [pointer_plot_coord - _pointer_half_width, -_pointer_height],
        [pointer_plot_coord + _pointer_half_width, -_pointer_height],
        [pointer_plot_coord, 0.0]
    ])
    p = patches.Polygon(pts, closed=True, facecolor=pointer_color, edgecolor='black',
    # p = patches.Polygon(pts, closed=True, facecolor='yellow', edgecolor='yellow',
                        linewidth=_pointer_line_width)
    ax.add_patch(p)
    plt.text(pointer_plot_coord, -_pointer_height - _text_height, f"{value:5.2f}",
             horizontalalignment='center', verticalalignment='top')


# def in_or_out_of_bounds_plot(value, lower, upper):
def var_bounds_plot(ax, value, lower, upper, equals):
    """
    Make a plot to show where a design variable is relative to constraints.

    Parameters
    ----------
    value : float
        The design var value.
    lower : float or None
        Lower constraint.
    upper : float or None
        Upper constraint.
    """

    # must handle 5 cases if both upper and lower are given:
    #  - value much less than lower
    #  - value a little less than lower
    #  - value between lower and upper
    #  - value a little greater than upper
    #  - value much greater than upper

    # also need to handle one-sided constraints where only one of lower and upper is given

    # the plotting coordinates will always be from 0.0 to 1.0
    #   below bounds, in bounds, above bounds sections are 1/3 of the width.
    #   this sets the scale of the plot


    # _backend = mpl.get_backend()
    # plt.style.use('default')
    # plt.autoscale(False)
    # mpl.use('Agg')



    if upper == INF_BOUND and lower == -INF_BOUND and equals is None:
        raise ValueError("Upper, lower, and equals bounds cannot all be None")

    # Basic plot setup
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.size'] = _font_size
    plt.rcParams['hatch.color'] = _out_of_bound_hatch_color
    plt.rcParams['hatch.linewidth'] = _out_of_bound_hatch_width
    plt.axis('off')
    ax.set_xlim([-_pointer_half_width, _plot_x_max + _pointer_half_width])
    ax.set_ylim(-_pointer_height - _text_height - _plot_y_margin,
                _plot_y_max + _text_height + _plot_y_margin)

    func_val_to_plot_coord = partial(_val_to_plot_coord, lower=lower, upper=upper)
    value_in_plot_coord = func_val_to_plot_coord(value)

    if equals is not None:
        # make a very narrow in bound range on the plot
        _draw_in_or_out_bound_section(ax, 0,
                                      _plot_x_max / 2 - _equality_bound_width * _plot_x_max,
                                      False)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2 - _equality_bound_width * _plot_x_max,
                                      2 *  _equality_bound_width * _plot_x_max,
                                      True)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2 + _equality_bound_width * _plot_x_max,
                                      _plot_x_max / 2 - _plot_x_max * _equality_bound_width,
                                      False)

        _draw_boundary_label(ax, _plot_x_max / 2, f"equals={equals:5.2f}")

        if abs(value - equals) < _relative_diff_from_bound:
            pointer_plot_coord = _plot_x_max / 2
            pointer_color = _in_bound_color
        else:
            pointer_color = _out_of_bound_color
            if value < equals:
                pointer_plot_coord = _plot_x_max / 4
            else:
                pointer_plot_coord = 3 * _plot_x_max / 4
        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        return

    if upper == INF_BOUND:  # so there is a lower bound
        _draw_in_or_out_bound_section(ax, 0, _plot_x_max / 2, False)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2, _plot_x_max / 2, True)
        _draw_boundary_label(ax, _plot_x_max / 2, f"lower={lower:5.2f}")

        if abs(value - lower) < _relative_diff_from_bound:
            pointer_plot_coord = _plot_x_max / 2
            _draw_bound_highlight(ax, _plot_x_max / 2)
            pointer_color = _in_bound_color
        elif value >= lower:
            pointer_plot_coord = 3. * _plot_x_max / 4
            pointer_color = _in_bound_color
        else:
            pointer_color = _out_of_bound_color
            pointer_plot_coord = 1. * _plot_x_max / 4
        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        # plt.show()
        return

    if lower == -INF_BOUND:  # so there is an upper bound
        _draw_in_or_out_bound_section(ax, 0, _plot_x_max / 2, True)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2, _plot_x_max / 2, False)
        _draw_boundary_label(ax, _plot_x_max / 2, f"upper={upper:5.2f}")

        if abs(value - upper) < _relative_diff_from_bound:
            pointer_plot_coord = _plot_x_max / 2
            _draw_bound_highlight(ax, _plot_x_max / 2)
            pointer_color = _in_bound_color
        elif value <= upper:
            pointer_plot_coord = 1. * _plot_x_max / 4
            pointer_color = _in_bound_color
        else:
            pointer_plot_coord = 3. * _plot_x_max / 4
            pointer_color = _out_of_bound_color

        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        return

    # There are both lower and upper bounds

    # Draw the rectangles for below bounds, in bounds, and above bounds

    # in bounds is always the same
    _draw_in_or_out_bound_section(ax, _lower_plot, _upper_plot - _lower_plot, True)

    # below bound
    if value_in_plot_coord >= 0.0:
        _draw_in_or_out_bound_section(ax, 0, _lower_plot, False)
    else:
        _draw_in_or_out_bound_section(ax, 0, _lower_plot / 3., False)
        _draw_in_or_out_bound_section(ax, 2 * _lower_plot / 3., _lower_plot / 3.,
                                      False)
        _draw_ellipsis(ax, 0.0)

    # upper bound
    if value_in_plot_coord <= _plot_x_max:
        _draw_in_or_out_bound_section(ax, _upper_plot, _lower_plot, False)
    else:
        _draw_in_or_out_bound_section(ax, _upper_plot, _lower_plot / 3., False)
        _draw_in_or_out_bound_section(ax, _upper_plot + 2 * _lower_plot / 3.,
                                      _lower_plot / 3., False)
        _draw_ellipsis(ax, _upper_plot)

    # upper and lower labels
    _draw_boundary_label(ax, func_val_to_plot_coord(lower), str(lower))
    _draw_boundary_label(ax, func_val_to_plot_coord(upper), str(upper))

    # add highlight if value near a bound
    if upper != INF_BOUND and lower != -INF_BOUND:
        hightlight_near_bound = abs(value - lower) / abs(upper - lower) < _relative_diff_from_bound
    elif upper != INF_BOUND:
        hightlight_near_bound = abs(value - upper) < _relative_diff_from_bound
    elif lower != -INF_BOUND:
        hightlight_near_bound = abs(value - lower) < _relative_diff_from_bound
    else:
        hightlight_near_bound = False

    if hightlight_near_bound:
        _draw_bound_highlight(ax, func_val_to_plot_coord(lower))

    # pointer and pointer label
    if value_in_plot_coord < 0.0:
        pointer_plot_coord = _plot_x_max / 18.0
    elif value_in_plot_coord > _plot_x_max:
        pointer_plot_coord = _plot_x_max - _plot_x_max / 18.0
    else:
        pointer_plot_coord = value_in_plot_coord
    pointer_color = _in_bound_color if (lower <= value <= upper) else _out_of_bound_color
    _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)


    # mpl.use(_backend)
    # plt.close()

# in_or_out_of_bounds_plot(0.0000000001, 2000000001, 4)
# in_or_out_of_bounds_plot(-2.5, 2, None)
# in_or_out_of_bounds_plot(3, None, 20 )
