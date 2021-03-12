"""
Figure built manually in SVG.

Note, these require the svgwrite package (and optionally, the svglib package to convert to pdf).
"""
import subprocess

from svgwrite import Drawing

filename = 'scaling_compute_totals_gmres.svg'
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
    v_lines.add(dwg.line(start=(loc, y+15), end=(loc, 1300)))

extra_text = dwg.add(dwg.g(font_size=main_font_size - 3, style="font-family: arial;"))
extra_text.add(dwg.text('fwd mode', (150, 50)))
extra_text.add(dwg.text('unit conversion', (vertical_locs[3] + 15, 870)))

locs = [('Problem.compute_totals()', None, []),
        ('_TotalJacInfo.compute_totals()', None, []),
        ('System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('Group._linearize()', None, []),
        ('  ExplicitComponent._linearize()', None, []),
        ('  ExplicitComponent._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('  Paraboloid.compute_partials()', ((4, 7), ), []),
        ('  ExplicitComponent._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('Loop over right-hand-sides', None, []),
        ('  _TotalJacInfo.single_input_setter()', [(0, 6)], []),
        ('  System.scaled_context_all()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('  Group._solve_linear()', None, []),
        ('  ScipyKrylov.solve()', [(6, 0)], []),
        ('  scipy.linalg.sparse.gmres()', None, []),
        ('    ScipyKrylov.matvec()', ((0, 5), ), []),
        ('    Group._apply_linear()', None, []),
        ("      DefaultVector.scale_to_norm()", ((0, 4), ), []),
        ('      DefaultTransfer.transfer()', ((5, 4), ), []),
        ("      DefaultVector.scale_to_phys()", ((0, 4), ), []),
        ('      ExplicitComponent._apply_linear()', None, []),
        ('      DictionaryJacobian._apply()', ((4, 6), ), []),
        ('      ExplicitComponent._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['stagger']),
        ('      (Multiply jac keys)', ((7, 6), (5, 6)), ['stagger']),
        ('      ExplicitComponent._unscaled_context()', ((0, 2), (0, 3), (0, 5), (0, 6)), ['italic', 'stagger']),
        ('    ScipyKrylov.matvec()', ((6, 0), ), ['italic']),
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
        grid_pts = [x + 10 * len(loc) + 10] + vertical_locs

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
                  (1, (7, 8)),
                  (1, (11, 22)),
                  (1, (24, 26)),
                  (2, (2, 5)),
                  (2, (7, 8)),
                  (2, (11, 22)),
                  (2, (24, 26)),
                  (4, (2, 5)),
                  (4, (7, 8)),
                  (4, (11, 22)),
                  (4, (24, 26)),
                  (5, (2, 5)),
                  (5, (7, 8)),
                  (5, (11, 22)),
                  (5, (24, 26)),
                  (3, (17, 19)),
                  ]

for region in scaled_regions:
    x_line = vertical_locs[region[0]]
    start = base_y + region[1][0] * delta_y - 5
    end = base_y + region[1][1] * delta_y - 5
    line = dwg.line(start=(x_line, start), end=(x_line, end))
    v_lines_scaled.add(line)

dwg.save()

print('done')