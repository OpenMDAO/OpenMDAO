"""
Figure built manually in SVG.

Note, these require the svgwrite package (and optionally, the svglib package to convert to pdf).
"""
from __future__ import print_function
import subprocess

from svgwrite import Drawing

filename = 'scaling_run_model.svg'
color_phys = '#85C1E9'
color_scaled = '##85C1E9'
main_font_size = 20

dwg = Drawing(filename, (2500, 2000), debug=True)

top_text = dwg.add(dwg.g(font_size=main_font_size, style="font-family: arial;"))

locs = ['NL Inputs',
        'NL Outputs',
        'NL Residuals']

x = 900
y = 50
delta_x = 400
vertical_locs = []
for loc in locs:
    top_text.add(dwg.text(loc, (x - len(loc)*4, y)))
    vertical_locs.append(x)
    x += delta_x

legend_text = dwg.add(dwg.g(font_size=main_font_size, style="font-family: arial;"))
legend_text.add(dwg.text('Phys', (x-300, y-10), fill=color_phys))
legend_text.add(dwg.text('Scaled', (x-300, y+20), fill=color_scaled))

v_lines = dwg.add(dwg.g(stroke_width=7.0, stroke=color_phys, fill='none'))
v_lines_scaled = dwg.add(dwg.g(stroke_width=7.0, stroke=color_scaled, fill='none'))

for loc in vertical_locs:
    v_lines.add(dwg.line(start=(loc, y+15), end=(loc, 1200)))

extra_text = dwg.add(dwg.g(font_size=main_font_size - 3, style="font-family: arial;"))
extra_text.add(dwg.text('Unit Conversion', (vertical_locs[0] + 10, 650)))

locs = [('Problem.run_model()', None, []),
        ('Model.run_solve_nonlinear()', None, []),
        ('System.scaled_context_all()', ((0, 2), (0, 3)), []),
        ('System.solve_nonlinear()', None, []),
        ('NonlinearRunOnce.solve()', None, []),
        ('NonlinearRunOnce.gs_iter()', None, []),
        ('  Group._transfer()', None, []),
        ("  DefaultVector.scale('norm')", ((0, 1), ), []),
        ('  DefaultTransfer.transfer()', ((2, 1), ), []),
        ("  DefaultVector.scale('phys')", ((0, 1), ), []),
        ('  ExplicitComponent.solve_nonlinear()', None, []),
        ('  ExplicitComponent._unscaled_context()', ((0, 2), (0, 3)), []),
        ('  Paraboloid.compute()', ((1, 2), ), []),
        ('  ExplicitComponent._unscaled_context()', ((0, 2), (0, 3)), ['italic']),
        ('System.scaled_context_all()', ((0, 2), (0, 3)), ['italic']),
        ('Problem.run_model()', None, ['italic']),
        ]

left_text = dwg.add(dwg.g(font_size=main_font_size, style="font-family: arial;"))
h_lines = dwg.add(dwg.g(stroke_width=0.7, stroke='black', fill='none'))

x = base_x = 40
y = base_y = 120
delta_y = 60
y_mids = []
for loc_tup in locs:

    loc, arrows, form = loc_tup

    offset = (len(loc) - len(loc.lstrip())) * 20

    if 'italic' in form:
        left_text.add(dwg.text(loc, (x + offset, y), style="font-style: italic;"))
    else:
        left_text.add(dwg.text(loc, (x + offset, y)))

    y_mid = y - 5.0
    y_mids.append(y_mid)
    if arrows:
        grid_pts = [x + 10 * len(loc)] + vertical_locs

        # Arrowheads
        for arrow in arrows:
            start = grid_pts[arrow[0]]
            end = grid_pts[arrow[1]]

            line = dwg.line(start=(start, y_mid), end=(end, y_mid))
            h_lines.add(line)

            # Arrowhead
            if end > start:
                ar_l = 10
            else:
                ar_l = -10

            ar_h = 7.5
            pts = ((end-ar_l, y_mid+ar_h), (end, y_mid), (end-ar_l, y_mid-ar_h))
            arrow = dwg.polyline(pts)
            h_lines.add(arrow)

    y += delta_y


# Phys vs scaling indicator
scaled_regions = [(0, (7, 9)),
                  (1, (2, 11)),
                  (1, (13, 14)),
                  (2, (2, 11)),
                  (2, (13, 14)),
                  ]

for region in scaled_regions:
    x_line = vertical_locs[region[0]]
    start = base_y + region[1][0] * delta_y
    end = base_y + region[1][1] * delta_y
    line = dwg.line(start=(x_line, start), end=(x_line, end))
    v_lines_scaled.add(line)

dwg.save()

print('done')