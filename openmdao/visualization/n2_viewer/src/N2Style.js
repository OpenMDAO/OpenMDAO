/**
 * Manage CSS styles for various elements.
 * @typedef N2Style
 */
class N2Style {

    constructor(svgStyle, fontSize) {
        this.svgStyle = svgStyle;

        this.setupSolverStyles();
        this.updateSvgStyle(fontSize);
    }

    /** Make the data in N2Style.solverStyleData more readable */
    setupSolverStyles() {
        this.solvers = {};
        let solverTypes = ['linear', 'nonLinear'];

        for (let sdata of N2Style.solverStyleData) {
            for (let i = 0; i < 2; ++i) {
                if (sdata[i]) {
                    this.solvers[sdata[i]] = {
                        'name': sdata[i],
                        'type': solverTypes[i],
                        'class': 'solver_' + sdata[i].toLowerCase().replace(/[: ]/g, '_').replace(/_+/g, '_'),
                        'style': {
                            'fill': sdata[2],
                            'cursor': 'pointer',
                            'fill-opacity': '.8'
                        }
                    }
                }
            }
        }

        // Object.freeze(this.solvers);
    }

    /**
     * Replace the entire content of the SVG style section with new styles.
     * Doing a wholesale replace is easier than finding each style element,
     * deleting, and inserting a new one.
     * @param {number} fontSize In pixel units.
     */
    updateSvgStyle(fontSize) {
        // Define as JSON first
        let newCssJson = {
            'rect': {
                'stroke': N2Style.color.treeStroke
            },
            'g.unknown > rect': {
                'fill': N2Style.color.unknownExplicit,
                'fill-opacity': '.8'
            },
            'g.unknown_implicit > rect': {
                'fill': N2Style.color.unknownImplicit,
                'fill-opacity': '.8'
            },
            'g.param > rect': {
                'fill': N2Style.color.param,
                'fill-opacity': '.8'
            },
            'g.unconnected_param > rect': {
                'fill': N2Style.color.unconnectedParam,
                'fill-opacity': '.8'
            },
            'g.subsystem > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.group
            },
            'g.component > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.component
            },
            'g.param_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.paramGroup
            },
            'g.unknown_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.unknownGroup
            },
            'g.minimized > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.collapsed
            },
            'text': {
                //'dominant-baseline: middle',
                //'dy: .35em',
            },
            '#svgId g.partition_group > text': {
                'text-anchor': 'end',
                'pointer-events': 'none',
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
            },
            'text.offgridLabel': {
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
            },
            '#svgId g.solver_group > text': {
                'text-anchor': 'end',
                'pointer-events': 'none',
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
            },
            'g.component_box > rect': {
                'stroke': N2Style.color.componentBox,
                'stroke-width': '2',
                'fill': 'none',
            },
            '.bordR1, .bordR2, .bordR3, .bordR4, .ssMid, .grpMid, .svMid, .vsMid, .vMid, .sgrpMid, .grpsMid': {
                'stroke': 'none',
                'stroke-width': '0',
                'fill-opacity': '1',
            },
            '[class^=n2_hover_elements]': {
                'pointer-events': 'none',
            },
            '.background': {
                'fill': N2Style.color.background,
            },
            '.horiz_line, .vert_line': {
                'stroke': N2Style.color.gridline,
            }
        }

        for (let solverName in this.solvers) {
            let solver = this.solvers[solverName];
            newCssJson['g.' + solver.class + ' > rect'] = solver.style;
        }

        // Iterate over the JSON object just created and turn it into
        // CSS style sheet text.
        let newCssText = '';
        for (let selector in newCssJson) {
            newCssText += selector + ' {\n';
            for (let attrib in newCssJson[selector]) {
                newCssText += '    ' + attrib + ': ' + newCssJson[selector][attrib] + ';\n';
            }
            newCssText += '}\n\n';
        }

        this.svgStyle.innerHTML = newCssText;
    }

    /**
     * Determine the name of the CSS class based on the name of the solver.
     * @param {boolean} showLinearSolverNames Whether to use the linear or non-linear solver name.
     * @param {Object} solverNames
     * @param {string} solverNames.linear The linear solver name.
     * @param {string} solverNames.nonLinear The non-linear solver name.
     * @return {string} The CSS class of the solver, or for "other" if not found.
     */
    getSolverClass(showLinearSolverNames, solverNames) {
        let solverName = showLinearSolverNames? solverNames.linear : solverNames.nonLinear;
        return this.solvers[solverName]? this.solvers[solverName].class :
            this.solvers.other.class;
    }

    /**
     * Based on the element's type and conditionally other info, determine
     * what CSS style is associated.
     * @return {string} The name of an existing CSS class.
     */
    getNodeClass(element) {
        if (element.isMinimized) return 'minimized';

        switch (element.type) {
            case 'param':
                if (Array.isPopulatedArray(element.children)) return 'param_group';
                return 'param';

            case 'unconnected_param':
                if (Array.isPopulatedArray(element.children)) return 'param_group';
                return 'unconnected_param';

            case 'unknown':
                if (Array.isPopulatedArray(element.children)) return 'unknown_group';
                if (element.implicit) return 'unknown_implicit';
                return 'unknown';

            case 'root':
                return 'subsystem';

            case 'subsystem':
                if (element.subsystem_type == 'component') return 'component';
                return 'subsystem';

            default:
                throw ('CSS class not found for element ' + element);
        }

    }
}

// From Isaias Reyes
N2Style.color = {
    'connection': 'black',
    'unknownImplicit': '#c7d06d',
    'unknownExplicit': '#9ec4c7',
    'componentBox': '#555',
    'background': '#eee',
    'gridline': 'white',
    'treeStroke': '#eee',
    'unknownGroup': '#888',
    'param': '#32afad',
    'paramGroup': 'Orchid',
    'group': '#3476a2',
    'component': 'DeepSkyBlue',
    'collapsed': '#555',
    'unconnectedParam': '#f42e0c',
    'highlightHovered': 'blue',
    'redArrow': 'salmon',
    'greenArrow': 'seagreen',
};

Object.freeze(N2Style.color); // Make it the equivalent of a constant

/* 
* This is how we want to map solvers to colors and CSS classes
*    Linear             Nonlinear
*    ---------          ---------
* 0. None               None
* 1. LN: LNBJ           NL: NLBJ
* 2. LN: SCIPY
* 3. LN: RUNONCE        NL: RUNONCE
* 4. LN: Direct
* 5. LN: PETScKrylov
* 6. LN: LNBGS          NL: NLBGS
* 7. LN: USER
* 8.                    NL: Newton
* 9.                    BROYDEN
* 10. solve_linear      solve_nonlinear
* 11. other             other

* Later add these for linesearch ?
* LS: AG
* LS: BCHK
*/
N2Style.solverStyleData = [
    // [ linear, non-Linear, color ]
    ['None', 'None', '#8dd3c7'],
    ["LN: LNBJ", "NL: NLBJ", '#ffffb3'],
    ["LN: SCIPY", null, '#bebada'],
    ["LN: RUNONCE", "NL: RUNONCE", '#fb8072'],
    ["LN: Direct", null, '#80b1d3'],
    ["LN: PETScKrylov", null, '#fdb462'],
    ["LN: LNBGS", "NL: NLBGS", '#b3de69'],
    ["LN: USER", null, '#fccde5'],
    [null, "NL: Newton", '#d9d9d9'],
    [null, "BROYDEN", '#bc80bd'],
    ["solve_linear", "solve_nonlinear", '#ccebc5'],
    ["other", "other", '#ffed6f']
];

Object.freeze(N2Style.solverStyleData); // Make it the equivalent of a constant