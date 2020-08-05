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
                        'class': 'solver_' +
                            sdata[i]
                                .toLowerCase()
                                .replace(/[: ]/g, '_')
                                .replace(/_+/g, '_'),
                        'style': {
                            'fill': sdata[2],
                            'cursor': 'pointer',
                            'fill-opacity': '.8',
                        },
                    };
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
                'stroke': N2Style.color.treeStroke,
            },
            '#tree > g.output > rect': {
                'fill': N2Style.color.outputExplicit,
                'fill-opacity': '.8',
            },
            '#tree > g.output_implicit > rect': {
                'fill': N2Style.color.outputImplicit,
                'fill-opacity': '.8',
            },
            '#tree > g.input > rect': {
                'fill': N2Style.color.input,
                'fill-opacity': '.8',
            },
            '#tree > g.unconnected_input > rect': {
                'fill': N2Style.color.unconnectedInput,
                'fill-opacity': '.8',
            },
            '#tree > g.subsystem > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.group,
            },
            '#tree > g.component > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.component,
            },
            '#tree > g.input_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.inputGroup,
            },
            '#tree > g.output_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.outputGroup,
            },
            '#tree > g.minimized > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.collapsed,
            },
            '#tree > g.autoivc_input > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': N2Style.color.autoivcInput,
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
            },
        };

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
                newCssText +=
                    '    ' + attrib + ': ' + newCssJson[selector][attrib] + ';\n';
            }
            newCssText += '}\n\n';
            if (selector === 'g.subsystem > rect') {
            }
        }

        this.svgStyle.html(newCssText);
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
        let solverName = showLinearSolverNames
            ? solverNames.linear
            : solverNames.nonLinear;
        return this.solvers[solverName]
            ? this.solvers[solverName].class
            : this.solvers.other.class;
    }

    /**
       * Based on the element's type and conditionally other info, determine
       * what CSS style is associated.
       * @return {string} The name of an existing CSS class.
       */
    getNodeClass(element) {
        if (element.isMinimized) return 'minimized';

        switch (element.type) {
            case 'input':
                if (Array.isPopulatedArray(element.children)) return 'input_group';
                return 'input';

            case 'unconnected_input':
                if (Array.isPopulatedArray(element.children)) return 'input_group';
                return 'unconnected_input';

            case 'autoivc_input':
                return 'autoivc_input';

            case 'output':
                if (Array.isPopulatedArray(element.children)) return 'output_group';
                if (element.implicit) return 'output_implicit';
                return 'output';

            case 'root':
                return 'subsystem';

            case 'subsystem':
                if (element.subsystem_type == 'component') return 'component';
                return 'subsystem';

            default:
                throw 'CSS class not found for element ' + element;
        }
    }
}

// From Isaias Reyes
N2Style.color = {
    'connection': 'gray',
    'outputImplicit': '#C7D06D',
    'outputExplicit': '#9FC4C6',
    'componentBox': '#555',
    'background': '#eee',
    'gridline': 'white',
    'treeStroke': '#eee',
    'outputGroup': '#888',
    'input': '#30B0AD',
    'inputGroup': 'Orchid',
    'group': '#6092B5',
    'component': '#02BFFF',
    'collapsed': '#555555',
    'unconnectedInput': '#F42F0D',
    'inputArrow': 'salmon',
    'outputArrow': 'seagreen',
    'declaredPartial': 'black',
    'autoivcInput': '#F42F0D'
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
    ['LN: LNBJ', 'NL: NLBJ', '#ffffb3'],
    ['LN: SCIPY', null, '#bebada'],
    ['LN: RUNONCE', 'NL: RUNONCE', '#fb8072'],
    ['LN: Direct', null, '#80b1d3'],
    ['LN: PETScKrylov', null, '#fdb462'],
    ['LN: LNBGS', 'NL: NLBGS', '#b3de69'],
    ['LN: USER', null, '#fccde5'],
    [null, 'NL: Newton', '#d9d9d9'],
    [null, 'BROYDEN', '#bc80bd'],
    ['solve_linear', 'solve_nonlinear', '#ccebc5'],
    ['other', 'other', '#ffed6f'],
];

N2Style.solverStyleObject = [
    {
        'ln': 'None',
        'nl': 'None',
        'color': '#8dd3c7',
    },
    {
        'ln': 'LN: LNBJ',
        'nl': 'NL: NLBJ',
        'color': '#ffffb3',
    },
    {
        'ln': 'LN: SCIPY',
        'nl': null,
        'color': '#bebada',
    },
    {
        'ln': 'LN: RUNONCE',
        'nl': 'NL: RUNONCE',
        'color': '#fb8072',
    },
    {
        'ln': 'LN: Direct',
        'nl': null,
        'color': '#80b1d3',
    },
    {
        'ln': 'LN: PETScKrylov',
        'nl': null,
        'color': '#fdb462',
    },
    {
        'ln': 'LN: LNBGS',
        'nl': 'NL: NLBGS',
        'color': '#b3de69',
    },
    {
        'ln': 'LN: USER',
        'nl': null,
        'color': '#fccde5',
    },
    {
        'ln': null,
        'nl': 'NL: Newton',
        'color': '#d9d9d9',
    },
    {
        'ln': null,
        'nl': 'BROYDEN',
        'color': '#bc80bd',
    },
    {
        'ln': 'solve_linear',
        'nl': 'solve_nonlinear',
        'color': '#ccebc5',
    },
    {
        'ln': 'other',
        'nl': 'other',
        'color': '#ffed6f',
    },
];

Object.freeze(N2Style.solverStyleData); // Make it the equivalent of a constant
