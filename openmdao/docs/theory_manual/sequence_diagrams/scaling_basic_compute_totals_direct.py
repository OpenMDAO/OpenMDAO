"""
Figure built manually in SVG.

Note, these require the svgwrite package (and optionally, the svglib package to convert to pdf).
"""
from __future__ import print_function
import subprocess

from svgwrite import Drawing

filename = 'scaling_compute_totals_direct.svg'
color_phys = '#85C1E9'
color_scaled = '#EC7063'
main_font_size = 20

dwg = Drawing(filename, (2500, 2000), debug=True)

top_text = dwg.add(dwg.g(font_size=main_font_size, style="font-family: arial;"))

locs = ['NL Inputs',
        'NL Outputs',
        'NL Residuals',
        'LN Inputs',
        'LN Outputs',
        'LN Residuals',
        'Jacobian']

x = 650
y = 50
delta_x = 180
vertical_locs = []
for loc in locs:
    top_text.add(dwg.text(loc, (x - len(loc)*4, y)))
    vertical_locs.append(x)
    x += delta_x

legend_text = dwg.add(dwg.g(font_size=main_font_size, style="font-family: arial;"))
legend_text.add(dwg.text('Phys', (x-1500, y-10), fill=color_phys))
legend_text.add(dwg.text('Scaled', (x-1500, y+20), fill=color_scaled))

v_lines = dwg.add(dwg.g(stroke_width=7.0, stroke=color_phys, fill='none'))
v_lines_scaled = dwg.add(dwg.g(stroke_width=7.0, stroke=color_scaled, fill='none'))

for loc in vertical_locs:
    v_lines.add(dwg.line(start=(loc, y+15), end=(loc, 1200)))

extra_text = dwg.add(dwg.g(font_size=main_font_size - 3, style="font-family: arial;"))
extra_text.add(dwg.text('fwd', (vertical_locs[5] + 55, 820)))

locs = [('Problem.compute_totals()', None, []),
        ('_TotalJacInfo.compute_totals()', None, []),
        ('System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('Group._linearize()', None, []),
        ('  ExplicitComponent._linearize()', None, []),
        ('  ExplicitComponent._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('  Paraboloid.compute_partials()', ((4, 7), ), []),
        ('  ExplicitComponent._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('AssembledJacobian._update()', None, []),
        ('System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('DirectSolver._linearize()', None, []),
        ('scipy.sparse.linalg.splu', None, []),
        ('(Loop over right-hand-sides)', None, []),
        ('  _TotalJacInfo.single_input_setter()', [(0, 6)], []),
        ('  System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('  Group._solve_linear()', None, []),
        ('  DirectSolver.solve()', None, []),
        ('  Group._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('  scipy.sparse.linalg.inv', ((6, 7), (7, 5)), ['stagger']),
        ('  Group._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('  System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('  _TotalJacInfo.single_jac_setter()', [(5, 0)], []),
]

left_text = dwg.add(dwg.g(font_size=main_font_size, style="font-family: arial;"))
h_lines = dwg.add(dwg.g(stroke_width=0.7, stroke='black', fill='none'))

x = base_x = 40
y = base_y = 120
delta_y = 40
y_mids = []
for loc_tup in locs:

    loc, arrows, form = loc_tup

    offset = (len(loc) - len(loc.lstrip())) * 15

    if 'italic' in form:
        left_text.add(dwg.text(loc, (x + offset, y), style="font-style: italic;"))
    else:
        left_text.add(dwg.text(loc, (x + offset, y)))

    y_mid = y - 5.0
    y_mids.append(y_mid)
    if arrows:
        grid_pts = [x + 10 * len(loc)] + vertical_locs

        num_arrow = len(arrows)

        # Arrowheads
        for i, arrow in enumerate(arrows):
            start = grid_pts[arrow[0]]
            end = grid_pts[arrow[1]]

            if 'stagger' in form:
                del_y = 3
                y_use = y_mid - del_y + i*2*del_y/(num_arrow-1)
            else:
                y_use = y_mid

            line = dwg.line(start=(start, y_use), end=(end, y_use))
            h_lines.add(line)

            # Arrowhead
            if end > start:
                ar_l = 10
            else:
                ar_l = -10

            ar_h = 7.5
            pts = ((end-ar_l, y_use+ar_h), (end, y_use), (end-ar_l, y_use-ar_h))
            arrow = dwg.polyline(pts)
            h_lines.add(arrow)

    y += delta_y


# Phys vs scaling indicator
scaled_regions = [
                  (1, (2, 5)),
                  (1, (7, 9)),
                  (1, (14, 17)),
                  (1, (19, 20)),
                  (2, (2, 5)),
                  (2, (7, 9)),
                  (2, (14, 17)),
                  (2, (19, 20)),
                  (4, (2, 5)),
                  (4, (7, 9)),
                  (4, (14, 17)),
                  (4, (19, 20)),
                  (5, (2, 5)),
                  (5, (7, 9)),
                  (5, (14, 17)),
                  (5, (19, 20)),
                  ]

for region in scaled_regions:
    x_line = vertical_locs[region[0]]
    start = base_y + region[1][0] * delta_y - 5
    end = base_y + region[1][1] * delta_y - 5
    line = dwg.line(start=(x_line, start), end=(x_line, end))
    v_lines_scaled.add(line)

dwg.save()

print('done')