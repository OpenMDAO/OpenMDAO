/**
 * The outermost N2 class.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {N2Layout} layout Sizes and positions of visible elements.
 * @property {Object} parentDiv
 * @property {Object} d3ContentDiv The div containing all of the diagram's content.
 * @property {Object} svgDiv The div containing the SVG element.
 * @property {Object} svg The SVG element.
 * @property {Object} svgStyle Object where SVG style changes can be made.
 * @property {Object} toolTip Div to display tooltips.
 * @property {Boolean} showPath
 * @property {Array} backButtonHistory
 * @property {Array} forwardButtonHistory
 * @property {number} chosenCollapseDepth
 */
class N2Diagram {
    constructor(modelJSON) {
        this.model = new ModelData(modelJSON);
        this.showPath = false;

        this.setupContentDivs();
        this.transitionStartDelay = N2Diagram.defaultTransitionStartDelay;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];
        this.chosenCollapseDepth = -1;

        this.updateSvgStyle(N2Layout.fontSizePx);

        this.layout = new N2Layout(this.model, this.model.root);
    }

    /**
     * Find the divs for D3 content in the existing document, and add a style section.
     */
    setupContentDivs() {
        this.parentDiv = document.getElementById("ptN2ContentDivId");

        this.d3ContentDiv = this.parentDiv.querySelector("#d3_content_div");
        this.svgDiv = d3.select("#svgDiv");
        this.svg = d3.select("#svgId");

        this.svgStyle = document.createElement("style");
        this.svgStyle.setAttribute('title', 'svgStyle');
        this.parentDiv.querySelector("#svgId").appendChild(this.svgStyle);

        this.toolTip = d3.select(".tool-tip");
        this.arrowMarker = d3.select("#arrow");
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
                'stroke': PT_STROKE_COLOR
            },
            'g.unknown > rect': {
                'fill': UNKNOWN_EXPLICIT_COLOR,
                'fill-opacity': '.8'
            },
            'g.unknown_implicit > rect': {
                'fill': UNKNOWN_IMPLICIT_COLOR,
                'fill-opacity': '.8'
            },
            'g.param > rect': {
                'fill': PARAM_COLOR,
                'fill-opacity': '.8'
            },
            'g.unconnected_param > rect': {
                'fill': UNCONNECTED_PARAM_COLOR,
                'fill-opacity': '.8'
            },
            'g.subsystem > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': GROUP_COLOR
            },
            'g.component > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': COMPONENT_COLOR
            },
            'g.param_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': PARAM_GROUP_COLOR
            },
            'g.unknown_group > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': UNKNOWN_GROUP_COLOR
            },
            'g.minimized > rect': {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': COLLAPSED_COLOR
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
            '#svgId g.solver_group > text': {
                'text-anchor': 'end',
                'pointer-events': 'none',
                'font-family': 'helvetica, sans-serif',
                'font-size': fontSize + 'px',
            },
            'g.component_box > rect': {
                'stroke': N2_COMPONENT_BOX_COLOR,
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
                'fill': N2_BACKGROUND_COLOR,
            },
            '.horiz_line, .vert_line': {
                'stroke': N2_GRIDLINE_COLOR,
            }
        }

        // TODO: Get rid of these globals
        linearSolverNames.forEach(function(name) {
            newCssJson['g.' + linearSolverClasses[name] + ' > rect'] = {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': linearSolverColors[name]
            };
        });

        nonLinearSolverNames.forEach(function(name) {
            newCssJson['g.' + nonLinearSolverClasses[name] + ' > rect'] = {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': nonLinearSolverColors[name]
            };
        });
        
        // Iterate over the JSON object just created and turn it into
        // CSS style sheet text.
        let newCssText = '';
        Object.keys(newCssJson).forEach(function (selector) {
            newCssText += selector + ' {\n';
            Object.keys(newCssJson[selector]).forEach(function (attrib) {
                newCssText += '    ' + attrib + ': ' + newCssJson[selector][attrib] + ';\n';
            })
            newCssText += '}\n\n';
        });

        this.svgStyle.innerHTML = newCssText;
    }

    /**
     * Save the SVG to a filename selected by the user.
     * TODO: Use a proper file dialog instead of a simple prompt.
     */
    saveSvg() {
        let svgData = this.svg.node().outerHTML;
    
        // Add name spaces.
        if (!svgData.match(/^<svg[^>]+xmlns="http\:\/\/www\.w3\.org\/2000\/svg"/)) {
            svgData = svgData.replace(/^<svg/, '<svg xmlns="http://www.w3.org/2000/svg"');
        }
        if (!svgData.match(/^<svg[^>]+"http\:\/\/www\.w3\.org\/1999\/xlink"/)) {
            svgData = svgData.replace(/^<svg/, '<svg xmlns:xlink="http://www.w3.org/1999/xlink"');
        }
    
        // Add XML declaration
        svgData = '<?xml version="1.0" standalone="no"?>\r\n' + svgData;
    
        svgData = vkbeautify.xml(svgData);
        let svgBlob = new Blob([svgData], { type: "image/svg+xml;charset=utf-8" });
        let svgUrl = URL.createObjectURL(svgBlob);
        let downloadLink = document.createElement("a");
        downloadLink.href = svgUrl;
        let svgFileName = prompt("Filename to save SVG as", 'partition_tree_n2.svg');
        downloadLink.download = svgFileName;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
    }
}

N2Diagram.defaultTransitionStartDelay = 100;