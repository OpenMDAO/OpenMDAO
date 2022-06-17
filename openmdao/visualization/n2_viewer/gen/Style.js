/**
 * Manage CSS styles for various elements.
 * @typedef Style
 */
class Style {
    /** Define colors for each element type. Selected by Isaias Reyes */
    static color = {
        'connection': 'gray',
        'background': '#eee',
        'gridline': 'white',
        'treeStroke': '#eee',
        'outputGroup': '#888',
        'output': '#888',
        'input': '#30B0AD',
        'inputGroup': 'Orchid',
        'group': '#6092B5',
        'collapsed': '#555555',
        'collapsedText': 'white',
        'filter': '#555555',
        'unconnectedInput': '#F42F0D',
        'inputArrow': 'salmon',
        'outputArrow': 'seagreen',
        'variableBox': '#555',
    };

    /**
     * Initialize the Style object.
     * @param {Object} svgStyle A reference to the SVG style section, which will be rewritten.
     * @param {Number} fontSize The font size to apply to text styles.
     */
    constructor(svgStyle, fontSize) {
        this.svgStyle = svgStyle;

        this._init();
        this.updateSvgStyle(fontSize);
    }

    /** A stub for subclasses to perform initialization. */
    _init() { }

    /**
     * Associate selectors with various style attributes.
     * @param {Number} fontSize The font size to apply to text styles.
     * @returns {Object} An object with selectors as keys, and values that are also objects,
     *     with style attributes as keys and values as their settings.
     */
    _createStyleObj(fontSize) {
        const newCssJson = {
            'rect': {
                'stroke': Style.color.treeStroke,
            },
            '#tree > g.input > rect': {
                'fill': Style.color.input,
                'fill-opacity': '.8',
            },
            '#tree > g.unconnected_input > rect': {
                'fill': Style.color.unconnectedInput,
                'fill-opacity': '.8',
            },
            '#tree > g.input_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': Style.color.inputGroup,
            },
            '#tree > g.output > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': Style.color.output,
            },
            '#tree > g.group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': Style.color.group,
            },
            '#tree > g.output_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': Style.color.outputGroup,
            },
            '#tree > g.minimized > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': Style.color.collapsed,
            },
            '#tree > g.minimized > text': {
                'fill': Style.color.collapsedText,
            },
            /* 'text': {
                //'dominant-baseline: middle',
                //'dy: .35em',
            }, */
            'g.model_tree_grp > text': {
                'text-anchor': 'end',
                'pointer-events': 'none',
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
            },
            'text.offgridLabel': {
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
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
                'fill': Style.color.background,
            },
            '.horiz_line, .vert_line': {
                'stroke': Style.color.gridline,
            },
            "g.model_tree_grp > rect[id*='_FILTER_'] + text": {
                'font-style': 'italic'
            },
            'g.variable_box > rect': {
                'stroke': Style.color.variableBox,
                'stroke-width': '2',
                'fill': 'none',
            }
        };

        return newCssJson;
    }

    /**
     * Replace the entire content of the SVG style section with new styles.
     * Doing a wholesale replace is easier than finding each style element,
     * deleting, and inserting a new one.
     * @param {number} fontSize In pixel units.
     */
    updateSvgStyle(fontSize) {
        // Define as JSON first
        const newCssJson = this._createStyleObj(fontSize);

        // Iterate over the JSON object just created and turn it into
        // CSS style sheet text.
        let newCssText = '';
        for (const selector in newCssJson) {
            newCssText += `${selector} {\n`;
            for (const attrib in newCssJson[selector]) {
                newCssText +=
                    `    ${attrib}: ${newCssJson[selector][attrib]};\n`;
            }
            newCssText += '}\n\n';
        }

        this.svgStyle.html(newCssText);
    }

    /**
       * Based on the element's type and conditionally other info, determine
       * what CSS style is associated.
       * @param {TreeNode} node The item to check.
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

            case 'filter':
                return 'filter';

            case 'output':
                if (Array.isPopulatedArray(node.children)) return 'output_group';
                return 'output';

            case 'root':
                return 'group';

            case 'group':
                return 'group';

            default:
                throw `CSS class not found for node type: ${node.type}`
        }
    }
}
