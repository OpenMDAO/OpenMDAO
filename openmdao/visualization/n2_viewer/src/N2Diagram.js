/**
 * The outermost N2 class.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {N2Style} style Manages N2-related styles and functions.
 * @property {N2Layout} layout Sizes and positions of visible elements.
 * @property {N2Matrix} matrix Manages the grid of model elements.
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
 * @property {Object} scales Scalers in the X and Y directions to associate the relative
 *   position of an element to actual pixel coordinates.
 * @property {Object} transitCoords
 */
class N2Diagram {
    constructor(modelJSON) {
        this.model = new ModelData(modelJSON);
        this.zoomedElement = this.zoomedElementPrev = zoomedElement = this.model.root;

        this.showPath = false;

        this.setupContentDivs();
        this.transitionStartDelay = N2TransitionDefaults.startDelay;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];
        this.chosenCollapseDepth = -1;

        this.style = new N2Style(this.svgStyle, N2Layout.defaults.size.font);
        this.layout = new N2Layout(this.model, this.zoomedElement);

        this.oldPtN2Initialize();

        this.updateClickedIndices();

        this.matrix = new N2Matrix(this.layout.visibleNodes, this.model, this.layout, this.n2Groups);

        this.scales = {
            'unit': 'px',
            'model': {
                'x': d3.scaleLinear().range([0, this.layout.size.partitionTree.width]),
                'y': d3.scaleLinear().range([0, this.layout.size.partitionTree.height])
            },
            'solver': {
                'x': d3.scaleLinear().range([0, this.layout.size.solverTree.width]),
                'y': d3.scaleLinear().range([0, this.layout.size.solverTree.height])
            },
            'previous': {
                'model': {
                    'x': null,
                    'y': null
                },
                'solver': {
                    'x': null,
                    'y': null
                },
            },
            'firstRun': true
        }

        this.transitCoords = {
            'unit': 'px',
            'model': { 'x': 0, 'y': 0 },
            'solver': { 'x': 0, 'y': 0 },
            'previous': {
                'model': { 'x': 0, 'y': 0 },
                'solver': { 'x': 0, 'y': 0 }
            }
        }
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
            .attr('width', this.layout.size.diagram.width)
            .attr('height', this.layout.size.diagram.height);

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

    /** Make sure UI controls reflect history and current reality. */
    updateUI() {
        this.parentDiv.querySelector('#currentPathId').innerHTML =
            'PATH: root' + ((this.zoomedElement.parent) ? '.' : '') +
            this.zoomedElement.absPathName;
        this.parentDiv.querySelector('#backButtonId').disabled =
            (this.backButtonHistory.length == 0) ? 'disabled' : false;
        this.parentDiv.querySelector('#forwardButtonId').disabled =
            (this.forwardButtonHistory.length == 0) ? 'disabled' : false;
        this.parentDiv.querySelector('#upOneLevelButtonId').disabled =
            (this.zoomedElement === this.model.root) ? 'disabled' : false;
        this.parentDiv.querySelector('#returnToRootButtonId').disabled =
            (this.zoomedElement === this.model.root) ? 'disabled' : false;

        for (let i = 2; i <= this.model.maxDepth; ++i) {
            this.parentDiv.querySelector('#idCollapseDepthOption' + i).style.display =
                (i <= this.zoomedElement.depth) ? 'none' : 'block';
        }
    }

    /**
     * Make a copy of the previous transit coordinates and linear scalers before
     * setting new ones.
     */
    preservePreviousScale() {
        // Preserve previous coordinates
        Object.assign(this.transitCoords.previous.model, this.transitCoords.model);
        Object.assign(this.transitCoords.previous.solver, this.transitCoords.solver);

        // Preserve previous linear scales
        this.scales.previous.model.x = this.scales.model.x.copy();
        this.scales.previous.model.y = this.scales.model.y.copy();
        this.scales.previous.solver.x = this.scales.solver.x.copy();
        this.scales.previous.solver.y = this.scales.solver.y.copy();
    }

    updateScale() {
        if (!this.scales.firstRun) {
            this.preservePreviousScale();
        }

        this.transitCoords.model.x = (this.zoomedElement.x ? this.layout.size.partitionTree.width -
            this.layout.size.parentNodeWidth : this.layout.size.partitionTree.width) /
            (1 - this.zoomedElement.x);
        this.transitCoords.model.y = this.layout.size.diagram.height / this.zoomedElement.height;

        this.scales.model.x
            .domain([this.zoomedElement.x, 1])
            .range([this.zoomedElement.x ? this.layout.size.parentNodeWidth : 0,
            this.layout.size.partitionTree.width]);
        this.scales.model.y
            .domain([this.zoomedElement.y, this.zoomedElement.y + this.zoomedElement.height])
            .range([0, this.layout.size.diagram.height]);

        this.transitCoords.solver.x = (this.zoomedElement.xSolver ?
            this.layout.size.solverTree.width - this.layout.size.parentNodeWidth :
            this.layout.size.solverTree.width) / (1 - this.zoomedElement.xSolver);
        this.transitCoords.solver.y = this.layout.size.diagram.height / this.zoomedElement.heightSolver;

        this.scales.solver.x
            .domain([this.zoomedElement.xSolver, 1])
            .range([this.zoomedElement.xSolver ? this.layout.size.parentNodeWidth :
                0, this.layout.size.solverTree.width]);
        this.scales.solver.y
            .domain([this.zoomedElement.ySolver,
            this.zoomedElement.ySolver + this.zoomedElement.heightSolver])
            .range([0, this.layout.size.diagram.height]);

        if (this.scales.firstRun) { // first run, duplicate what we just calculated
            this.scales.firstRun = false;
            this.preservePreviousScale();

            //Update svg dimensions before ComputeLayout() changes layout.size.partitionTree.width
            this.svgDiv.style("width",
                (this.layout.size.partitionTree.width + this.layout.size.partitionTreeGap + this.layout.size.diagram.width +
                    this.layout.size.solverTree.width + 2 * this.layout.size.svgMargin + this.layout.size.partitionTreeGap) +
                this.layout.size.unit)
                .style("height", (this.layout.size.partitionTree.height + 2 * this.layout.size.svgMargin) +
                    this.layout.size.unit);
            this.svg.attr("width", this.layout.size.partitionTree.width + this.layout.size.partitionTreeGap +
                this.layout.size.diagram.width + this.layout.size.solverTree.width + 2 * this.layout.size.svgMargin + this.layout.size.partitionTreeGap)
                .attr("height", this.layout.size.partitionTree.height + 2 * this.layout.size.svgMargin);

            this.n2TopGroup.attr("transform", "translate(" +
                (this.layout.size.partitionTree.width + this.layout.size.partitionTreeGap + this.layout.size.svgMargin) +
                "," + this.layout.size.svgMargin + ")");
            this.pTreeGroup.attr("transform", "translate(" +
                this.layout.size.svgMargin + "," + this.layout.size.svgMargin + ")");

            this.pSolverTreeGroup.attr("transform", "translate(" +
                (this.layout.size.partitionTree.width + this.layout.size.partitionTreeGap + this.layout.size.diagram.width +
                    this.layout.size.svgMargin + this.layout.size.partitionTreeGap) + "," + this.layout.size.svgMargin + ")");
        }
    }

    /** Update svg dimensions with transition after a new N2Layout changes
     * layout.size.partitionTree.width
     */
    updateTransitionInfo() {
        sharedTransition = d3.transition()
            .duration(N2TransitionDefaults.duration)
            .delay(this.transitionStartDelay); // do this after intense computation
        this.transitionStartDelay = N2TransitionDefaults.startDelay;

        this.svgDiv.transition(sharedTransition)
            .style("width", (this.layout.size.partitionTree.width + this.layout.size.partitionTreeGap +
                this.layout.size.diagram.width + this.layout.size.solverTree.width + 2 * this.layout.size.svgMargin + this.layout.size.partitionTreeGap) +
                this.layout.size.unit)
            .style("height", (this.layout.size.partitionTree.height + 2 * this.layout.size.svgMargin) +
                this.layout.size.unit);

        this.svg.transition(sharedTransition)
            .attr("width", this.layout.size.partitionTree.width +
                this.layout.size.partitionTreeGap + this.layout.size.diagram.width + this.layout.size.solverTree.width + 2 *
                this.layout.size.svgMargin + this.layout.size.partitionTreeGap)
            .attr("height", this.layout.size.partitionTree.height + 2 * this.layout.size.svgMargin);

        this.n2TopGroup.transition(sharedTransition)
            .attr("transform", "translate(" + (this.layout.size.partitionTree.width +
                this.layout.size.partitionTreeGap + this.layout.size.svgMargin) + "," + this.layout.size.svgMargin + ")");

        this.pTreeGroup.transition(sharedTransition)
            .attr("transform", "translate(" + this.layout.size.svgMargin + "," + this.layout.size.svgMargin + ")");

        this.n2BackgroundRect.transition(sharedTransition)
            .attr("width", this.layout.size.diagram.width).attr("height", this.layout.size.partitionTree.height);

        this.pSolverTreeGroup.transition(sharedTransition)
            .attr("transform", "translate(" + (this.layout.size.partitionTree.width +
                this.layout.size.partitionTreeGap + this.layout.size.diagram.width + this.layout.size.svgMargin + this.layout.size.partitionTreeGap) + "," +
                this.layout.size.svgMargin + ")");
    }

    /**
     * Refresh the diagram when something has visually changed.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    update(computeNewTreeLayout = true) {
        this.updateUI();

        // Compute the new tree layout if necessary.
        if (computeNewTreeLayout) {
            this.layout = new N2Layout(this.model, this.zoomedElement);
            this.updateClickedIndices();
            this.matrix = new N2Matrix(this.layout.visibleNodes, this.model, this.layout, this.n2Groups);
        }

        this.updateScale();
        this.updateTransitionInfo();
    }
}