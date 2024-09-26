// <<hpp_insert gen/Style.js>>

/**
 * Manage CSS styles for various elements. Adds handling for solvers and OM-specifc types.
 * @typedef OmStyle
 */
class OmStyle extends Style {
    // Define colors for each element type. Selected by Isaias Reyes
    static color = {
        ...Style.color,
        'outputImplicit': '#C7D06D',
        'outputExplicit': '#9FC4C6',
        'desvar': '#c5b0d5',
        'component': '#02BFFF',
        'declaredPartial': 'black',
        'autoivcInput': '#ff7000'
    };

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
    * 9.                    NL: BROYDEN
    * 10. solve_linear      solve_nonlinear
    * 11. other             other

    * Later add these for linesearch ?
    * LS: AG
    * LS: BCHK
    */
    static solverStyleData = [
     // [linear,            non-Linear,        color    ]
        ['None',            'None',            '#8dd3c7'],
        ['LN: LNBJ',        'NL: NLBJ',        '#ffffb3'],
        ['LN: SCIPY',       null,              '#bebada'],
        ['LN: RUNONCE',     'NL: RUNONCE',     '#fb8072'],
        ['LN: Direct',      null,              '#80b1d3'],
        ['LN: PETScKrylov', null,              '#fdb462'],
        ['LN: LNBGS',       'NL: NLBGS',       '#b3de69'],
        ['LN: USER',        null,              '#fccde5'],
        [null,              'NL: Newton',      '#d9d9d9'],
        [null,              'NL: BROYDEN',     '#bc80bd'],
        ['solve_linear',    'solve_nonlinear', '#ccebc5'],
        ['other',           'other',           '#ffed6f'],
    ];

    /** The solverStyleData array is split into objects with keys 'ln', 'nl', and 'color' */
    static solverStyleObject = [];

    /**
     * Initialize the OmStyle object.
     * @param {Object} svgStyle A reference to the SVG style section, which will be rewritten.
     * @param {Number} fontSize The font size to apply to text styles.
     */
    constructor(svgStyle, fontSize) {
        super(svgStyle, fontSize);
    }

    /** Make the data in OmStyle.solverStyleData more readable */
    _init() {
        this.solvers = {};
        const solverTypes = ['linear', 'nonLinear'];

        for (const sdata of OmStyle.solverStyleData) {
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
    }

    /**
     * Associate selectors with various style attributes. Adds support for OM components,
     * subsystems, implicit/explicit outputs, solvers, and Auto-IVC inputs.
     * @param {Number} fontSize The font size to apply to text styles.
     * @returns {Object} An object with selectors as keys, and values that are also objects,
     *     with style attributes as keys and values as their settings.
     */
    _createStyleObj(fontSize) {
        const newCssJson = super._createStyleObj(fontSize);

        const OmCssJson = {
            '#tree > g.output_implicit > rect': {
                'fill': OmStyle.color.outputImplicit,
                'fill-opacity': '.8',
            },
            '#tree > g.output > rect': {
                'fill': OmStyle.color.outputExplicit,
                'fill-opacity': '.8',
            },
            '#tree > g.subsystem > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': OmStyle.color.group,
            },
            '#tree > g.component > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': OmStyle.color.component,
            },
            '#svgId g.solver_group > text': {
                'text-anchor': 'end',
                'pointer-events': 'none',
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
            },
            '#tree > g.autoivc_input > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': OmStyle.color.autoivcInput,
            },
        };

        for (const solverName in this.solvers) {
            const solver = this.solvers[solverName];

            OmCssJson[`g.${solver.class} > rect`] = solver.style;
        }

        return {...newCssJson, ...OmCssJson};
    }

    /**
     * Determine the name of the CSS class based on the name of the solver.
     * @param {Object} solverNames
     * @param {string} solverNames.linear The linear solver name.
     * @param {string} solverNames.nonLinear The non-linear solver name.
     * @return {string} The CSS class of the solver, or for "other" if not found.
     */
    getSolverClass(solverNames) {
        const solverName = OmTreeNode.showLinearSolverNames? solverNames.linear : solverNames.nonLinear;
        return this.solvers[solverName]? this.solvers[solverName].class : this.solvers.other.class;
    }

    /**
     * Based on the element's type and conditionally other info, determine
     * what CSS style is associated.
     * @param {OmTreeNode} node The item to check.
     * @return {string} The name of an existing CSS class.
     */
    getNodeClass(node) {
        if (node.draw.minimized) return 'minimized';

        switch (node.type) {
            case 'input':
                if (Array.isPopulatedArray(node.children)) return 'input_group';
                return 'input';

            case 'unconnected_input':
                if (Array.isPopulatedArray(node.children)) return 'input_group';
                return 'unconnected_input';

            case 'autoivc_input':
                return 'autoivc_input';

            case 'filter':
                return 'filter';

            case 'output':
                if (Array.isPopulatedArray(node.children)) return 'output_group';
                if (node.implicit) return 'output_implicit';
                return 'output';

            case 'root':
                return 'subsystem';

            case 'subsystem':
                if (node.subsystem_type == 'component') return 'component';
                return 'subsystem';

            default:
                throw `CSS class not found for node type: ${node.type}`
        }
    }
}

// Load solverStyleObject with values from solverStyleData
for (const solver of OmStyle.solverStyleData) {
    OmStyle.solverStyleObject.push({ ln: solver[0], nl: solver[1], color: solver[2] });
}

