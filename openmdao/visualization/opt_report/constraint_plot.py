"""
Function to show a plot where a value is relative to constraints
"""
from functools import partial

from matplotlib import pyplot as plt, patches

import numpy as np

from openmdao.core.constants import INF_BOUND

_in_bound_color = 'green'
_out_of_bound_color = 'red'
_plot_x_max = 1.0
_plot_y_max = 1.0
_plot_y_margin = 0.2
_lower_plot = _plot_x_max / 3.
_upper_plot = 2. * _plot_x_max / 3.
_pointer_half_width = 0.04
_pointer_height = 0.4
_text_height = 0.3
_font_size = 5
_ellipse_width = 0.01
_ellipse_height = 0.15
_pointer_line_width = 0.02
_relative_diff_from_bound = 1e-4
_near_bound_highlight_half_width  = 0.05
_near_bound_highlight_half_width_y_min = -0.5
_near_bound_highlight_half_width_y_max = _plot_y_max + 0.6
_near_bound_highlight_alpha = 0.7

def _val_to_plot_coord(value, lower, upper):
    # need to get function that maps actual values to 0.0 to 1.0
    # where lower maps to 1./3 and upper to 2./3
    plot_coord = 1./3. + (value - lower) / (upper - lower) * 1./3.
    return plot_coord


def _draw_in_or_out_bound_section(ax, x_left, width, color):
    rectangle = patches.Rectangle((x_left, 0), width, _plot_y_max, facecolor=color)
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
    plt.text(pointer_plot_coord, -_pointer_height - _text_height, f"{value}",
             horizontalalignment='center', verticalalignment='top')


# def in_or_out_of_bounds_plot(value, lower, upper):
def in_or_out_of_bounds_plot(ax, value, lower, upper):
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

    if upper == INF_BOUND and lower == -INF_BOUND:
        raise ValueError("Upper and lower bounds cannot both be None")

    # Basic plot setup
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.size'] = _font_size
    plt.axis('off')
    ax.set_xlim([-_pointer_half_width, _plot_x_max + _pointer_half_width])
    ax.set_ylim(-_pointer_height - _text_height - _plot_y_margin,
                _plot_y_max + _text_height + _plot_y_margin)

    if upper == INF_BOUND:  # so there is a lower bound
        _draw_in_or_out_bound_section(ax, 0, _plot_x_max / 2., _out_of_bound_color)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2., _plot_x_max / 2., _in_bound_color)
        plt.text(_plot_x_max / 2., _plot_y_max + _text_height,
                 f"lower={lower}",
                 horizontalalignment='center',
                 verticalalignment='bottom')
        if abs(value - lower) < _relative_diff_from_bound:
            pointer_plot_coord = _plot_x_max / 2.
            _draw_bound_highlight(ax, _plot_x_max / 2.)

        elif value >= lower:
            pointer_plot_coord = 3. * _plot_x_max / 4.
        else:
            pointer_plot_coord = 1. * _plot_x_max / 4.
        pointer_color = _in_bound_color if value >= lower else _out_of_bound_color
        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        # plt.show()
        return

    if lower == -INF_BOUND:  # so there is an upper bound
        _draw_in_or_out_bound_section(ax, 0, _plot_x_max / 2., _in_bound_color)
        _draw_in_or_out_bound_section(ax, _plot_x_max / 2., _plot_x_max / 2., _out_of_bound_color)
        plt.text(_plot_x_max / 2., _plot_y_max + _text_height,
                 f"upper={upper}",
                 horizontalalignment='center',
                 verticalalignment='bottom')
        if value <= upper:
            pointer_plot_coord = 1. * _plot_x_max / 4.
            pointer_color = _in_bound_color
        else:
            pointer_plot_coord = 3. * _plot_x_max / 4.
            pointer_color = _out_of_bound_color
        _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)
        # plt.show()
        return

    func_val_to_plot_coord = partial(_val_to_plot_coord, lower=lower, upper=upper)
    value_in_plot_coord = func_val_to_plot_coord(value)

    # Draw the rectangles for below bounds, in bounds, and above bounds

    # in bounds is always the same
    _draw_in_or_out_bound_section(ax, _lower_plot, _upper_plot - _lower_plot, _in_bound_color)

    # below bound
    if value_in_plot_coord >= 0.0:
        _draw_in_or_out_bound_section(ax, 0, _lower_plot, _out_of_bound_color)
    else:
        _draw_in_or_out_bound_section(ax, 0, _lower_plot / 3., _out_of_bound_color)
        _draw_in_or_out_bound_section(ax, 2. * _lower_plot / 3., _lower_plot / 3.,
                                      _out_of_bound_color)
        _draw_ellipsis(ax, 0.0)

    # upper bound
    if value_in_plot_coord <= _plot_x_max:
        _draw_in_or_out_bound_section(ax, _upper_plot, _lower_plot, _out_of_bound_color)
    else:
        _draw_in_or_out_bound_section(ax, _upper_plot, _lower_plot / 3., _out_of_bound_color)
        _draw_in_or_out_bound_section(ax, _upper_plot + 2. * _lower_plot / 3.,
                                      _lower_plot / 3., _out_of_bound_color)
        _draw_ellipsis(ax, _upper_plot)

    # upper and lower labels
    plt.text(func_val_to_plot_coord(lower), _plot_y_max + _text_height,
             f"{lower}",
             horizontalalignment='center',
             weight='bold',
             verticalalignment='bottom')
    plt.text(func_val_to_plot_coord(upper), _plot_y_max + _text_height, f"{upper}",
             horizontalalignment='center',
             verticalalignment='bottom')

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
        # rectangle = patches.Rectangle((
        #                                 func_val_to_plot_coord(lower) -_near_bound_highlight_half_width,
        #                                 _near_bound_highlight_half_width_y_min),
        #                                 2 * _near_bound_highlight_half_width,
        #                                 _near_bound_highlight_half_width_y_max,
        #                                 edgecolor='black',
        #                                 facecolor='yellow',
        #                                 alpha=_near_bound_highlight_alpha)
        # ax.add_patch(rectangle)



    # pointer and pointer label
    if value_in_plot_coord < 0.0:
        pointer_plot_coord = _plot_x_max / 18.0
    elif value_in_plot_coord > _plot_x_max:
        pointer_plot_coord = _plot_x_max - _plot_x_max / 18.0
    else:
        pointer_plot_coord = value_in_plot_coord
    pointer_color = _in_bound_color if (lower <= value <= upper) else _out_of_bound_color
    _draw_pointer_and_label(ax, pointer_plot_coord, pointer_color, value)

    # plt.show()


# in_or_out_of_bounds_plot(0.0000000001, 2.000000001, 4.)
# in_or_out_of_bounds_plot(-2.5, 2., None)
# in_or_out_of_bounds_plot(3, None, 2.0 )
