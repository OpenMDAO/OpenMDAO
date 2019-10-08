/**
 * The outermost N2 class.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {N2Layout} layout Sizes and positions of visible elements.
 * @property {Object} zoomedElement The element the diagram is currently based on.
 * @property {Object} zoomedElementPrev Reference to last zoomedElement.
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
 * @property {Object} n2TopGroup
 * @property {Object} n2Groups References to <g> SVG elements.
 */
class N2Diagram {
    constructor(modelJSON) {
        this.model = new ModelData(modelJSON);
        this.zoomedElement = this.zoomedElementPrev = zoomedElement = this.model.root;

        this.showPath = false;

        this.setupContentDivs();
        this.transitionStartDelay = N2Diagram.defaultTransitionStartDelay;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];
        this.chosenCollapseDepth = -1;

        this.updateSvgStyle(N2Layout.fontSizePx);

        this.layout = new N2Layout(this.model, this.zoomedElement);

        this.oldPtN2Initialize();

        this.updateClickedIndices();

        this.matrix = new N2Matrix(this.layout.visibleNodes, this.model, this.n2Groups);
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
        linearSolverNames.forEach(function (name) {
            newCssJson['g.' + linearSolverClasses[name] + ' > rect'] = {
                'cursor': 'pointer',
                'fill-opacity': '.8',
                'fill': linearSolverColors[name]
            };
        });

        nonLinearSolverNames.forEach(function (name) {
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

    /**
     * Replace the current zoomedElement, but preserve its value.
     * @param {Object} newZoomedElement Replacement zoomed element.
     */
    updateZoomedElement(newZoomedElement) {
        this.zoomedElementPrev = this.zoomedElement;

        // TODO: Stop updating the global zoomedElement when we
        // implement a different place to put it.
        this.zoomedElement = zoomedElement = newZoomedElement;

        this.layout.zoomedElement = this.zoomedElement;
    }

    /** Actions that still need to be integrated better, but currently grouped here so
     * they can be called at the right time.
     */
    oldPtN2Initialize() {
        // ids given just so it is easier to see in Chrome dev tools when debugging
        this.n2TopGroup = this.svg.append('g').attr('id', 'N2');
        this.pTreeGroup = this.svg.append('g').attr('id', 'tree');
        this.pSolverTreeGroup = this.svg.append('g').attr('id', 'solver_tree');

        this.n2BackgroundRect = this.n2TopGroup.append('rect')
            .attr('class', 'background')
            .attr('width', WIDTH_N2_PX)
            .attr('height', N2Layout.heightPx);

        this.n2Groups = {};
        ['elements', 'gridlines', 'componentBoxes', 'arrows', 'dots'].forEach(function (gName) {
            this.n2Groups[gName] = this.n2TopGroup.append('g').attr('id', 'n2' + gName);
        }.bind(this));
    }

    /** Set up for an animated transition by setting and remembering where things were.
     * TODO: Get rid of the globals
     */
    updateClickedIndices() {
        enterIndex = exitIndex = 0;
        if (lastClickWasLeft) { //left click
            if (leftClickIsForward) {
                exitIndex = lastLeftClickedElement.rootIndex -
                    this.zoomedElementPrev.rootIndex;
            }
            else {
                enterIndex = this.zoomedElementPrev.rootIndex -
                    lastLeftClickedElement.rootIndex;
            }
        }
    }

    /**
     * Refresh the diagram when something has visually changed.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    update(computeNewTreeLayout = true) {
        this.parentDiv.querySelector("#currentPathId").innerHTML =
            "PATH: root" + ((this.zoomedElement.parent) ? "." : "") +
            this.zoomedElement.absPathName;
        this.parentDiv.querySelector("#backButtonId").disabled =
            (this.backButtonHistory.length == 0) ? "disabled" : false;
        this.parentDiv.querySelector("#forwardButtonId").disabled =
            (this.forwardButtonHistory.length == 0) ? "disabled" : false;
        this.parentDiv.querySelector("#upOneLevelButtonId").disabled =
            (this.zoomedElement === this.model.root) ? "disabled" : false;
        this.parentDiv.querySelector("#returnToRootButtonId").disabled =
            (this.zoomedElement === this.model.root) ? "disabled" : false;

        // Compute the new tree layout.
        if (computeNewTreeLayout) {
            this.layout = new N2Layout(this.model, this.zoomedElement);
            this.updateClickedIndices();
            this.matrix = new N2Matrix(this.layout.visibleNodes, this.model, this.n2Groups);
        }
    }
}

N2Diagram.defaultTransitionStartDelay = 100;