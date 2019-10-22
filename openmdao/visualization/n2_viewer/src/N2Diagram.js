/**
 * The outermost N2 class.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {N2Style} style Manages N2-related styles and functions.
 * @property {N2Layout} layout Sizes and positions of visible elements.
 * @property {N2Matrix} matrix Manages the grid of model elements.
 * @property {Object} zoomedElement The element the diagram is currently based on.
 * @property {Object} zoomedElementPrev Reference to last zoomedElement.
 * @property {Object} dom Container for references to web page elements.
 * @property {Object} dom.parentDiv
 * @property {Object} dom.d3ContentDiv The div containing all of the diagram's content.
 * @property {Object} dom.svgDiv The div containing the SVG element.
 * @property {Object} dom.svg The SVG element.
 * @property {Object} dom.svgStyle Object where SVG style changes can be made.
 * @property {Object} dom.toolTip Div to display tooltips.
 * @property {Object} dom.n2TopGroup
 * @property {Object} dom.n2Groups References to <g> SVG elements.
 * @property {Boolean} showPath
 * @property {Array} backButtonHistory
 * @property {Array} forwardButtonHistory
 * @property {number} chosenCollapseDepth
 * @property {Object} scales Scalers in the X and Y directions to associate the relative
 *   position of an element to actual pixel coordinates.
 * @property {Object} transitCoords
 */
class N2Diagram {
    constructor(modelJSON) {
        this.model = new ModelData(modelJSON);
        this.zoomedElement = this.zoomedElementPrev = this.model.root;

        this.showPath = false;

        // Find the divs for D3 content in the existing document, and add a style section.
        let parentDiv = document.getElementById("ptN2ContentDivId");
        this.dom = {
            'parentDiv': parentDiv,
            'd3ContentDiv': parentDiv.querySelector("#d3_content_div"),
            'svgDiv': d3.select("#svgDiv"),
            'svg': d3.select("#svgId"),
            'svgStyle': document.createElement("style"),
            'toolTip': d3.select(".tool-tip"),
            'arrowMarker': d3.select("#arrow")
        };

        // Append the new style section.
        this.dom.svgStyle.setAttribute('title', 'svgStyle');
        this.dom.parentDiv.querySelector("#svgId").appendChild(this.dom.svgStyle);

        this.transitionStartDelay = N2TransitionDefaults.startDelay;

        this.backButtonHistory = [];
        this.forwardButtonHistory = [];
        this.chosenCollapseDepth = -1;

        this.showLinearSolverNames = true;

        this.style = new N2Style(this.dom.svgStyle, N2Layout.defaults.size.font);
        this.layout = new N2Layout(this.model, this.zoomedElement);

        this._setupSvgElements();
        this._updateClickedIndices();

        this.matrix = new N2Matrix(this.layout.visibleNodes, this.model, this.layout, this.dom.n2Groups);

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
            'firstRun': true
        }

        this.prevScales = {
            'model': {
                'x': null,
                'y': null
            },
            'solver': {
                'x': null,
                'y': null
            },
        };

        this.transitCoords = {
            'unit': 'px',
            'model': { 'x': 0, 'y': 0 },
            'solver': { 'x': 0, 'y': 0 }
        };

        this.prevTransitCoords = {
            'unit': 'px',
            'model': { 'x': 0, 'y': 0 },
            'solver': { 'x': 0, 'y': 0 }
        };
    }

    /**
    * Switch back and forth between showing the linear or non-linear solver names. 
    */
    toggleSolverNameType() {
        this.showLinearSolverNames = !this.showLinearSolverNames;
    }

    /**
     * Save the SVG to a filename selected by the user.
     * TODO: Use a proper file dialog instead of a simple prompt.
     */
    saveSvg() {
        let svgData = this.dom.svg.node().outerHTML;

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
        this.zoomedElement = newZoomedElement;

        this.layout.zoomedElement = this.zoomedElement;
    }

    /** 
     * Add several elements, mostly groups, to the primary SVG element to
     * be populated later. Stored in the .doms property.
     */
    _setupSvgElements() {
        // ids given just so it is easier to see in Chrome dev tools when debugging
        this.dom.n2TopGroup = this.dom.svg.append('g').attr('id', 'N2');
        this.dom.pTreeGroup = this.dom.svg.append('g').attr('id', 'tree');
        this.dom.pSolverTreeGroup = this.dom.svg.append('g').attr('id', 'solver_tree');

        this.dom.n2BackgroundRect = this.dom.n2TopGroup.append('rect')
            .attr('class', 'background')
            .attr('width', this.layout.size.diagram.width)
            .attr('height', this.layout.size.diagram.height);

        this.dom.n2Groups = {};
        for (let gName of ['elements', 'gridlines', 'componentBoxes', 'arrows', 'dots']) {
            this.dom.n2Groups[gName] = this.dom.n2TopGroup.append('g').attr('id', 'n2' + gName);
        };
    }

    /** Set up for an animated transition by setting and remembering where things were.
     * TODO: Get rid of the globals
     */
    _updateClickedIndices() {
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
    _updateUI() {
        this.dom.parentDiv.querySelector('#currentPathId').innerHTML =
            'PATH: root' + ((this.zoomedElement.parent) ? '.' : '') +
            this.zoomedElement.absPathName;
        this.dom.parentDiv.querySelector('#backButtonId').disabled =
            (this.backButtonHistory.length == 0) ? 'disabled' : false;
        this.dom.parentDiv.querySelector('#forwardButtonId').disabled =
            (this.forwardButtonHistory.length == 0) ? 'disabled' : false;
        this.dom.parentDiv.querySelector('#upOneLevelButtonId').disabled =
            (this.zoomedElement === this.model.root) ? 'disabled' : false;
        this.dom.parentDiv.querySelector('#returnToRootButtonId').disabled =
            (this.zoomedElement === this.model.root) ? 'disabled' : false;

        for (let i = 2; i <= this.model.maxDepth; ++i) {
            this.dom.parentDiv.querySelector('#idCollapseDepthOption' + i).style.display =
                (i <= this.zoomedElement.depth) ? 'none' : 'block';
        }
    }

    /**
     * Make a copy of the previous transit coordinates and linear scalers before
     * setting new ones.
     */
    _preservePreviousScale() {
        // Preserve previous coordinates
        Object.assign(this.prevTransitCoords.model, this.transitCoords.model);
        Object.assign(this.prevTransitCoords.solver, this.transitCoords.solver);

        // Preserve previous linear scales
        this.prevScales.model.x = this.scales.model.x.copy();
        this.prevScales.model.y = this.scales.model.y.copy();
        this.prevScales.solver.x = this.scales.solver.x.copy();
        this.prevScales.solver.y = this.scales.solver.y.copy();
    }

    _updateScale() {
        if (!this.scales.firstRun) this._preservePreviousScale();

        this.transitCoords.model.x = (this.zoomedElement.dims.x ?
            this.layout.size.partitionTree.width - this.layout.size.parentNodeWidth :
            this.layout.size.partitionTree.width) / (1 - this.zoomedElement.dims.x);
        this.transitCoords.model.y = this.layout.size.diagram.height /
            this.zoomedElement.dims.height;

        this.scales.model.x
            .domain([this.zoomedElement.dims.x, 1])
            .range([this.zoomedElement.dims.x ? this.layout.size.parentNodeWidth : 0,
            this.layout.size.partitionTree.width]);
        this.scales.model.y
            .domain([this.zoomedElement.dims.y, this.zoomedElement.dims.y +
                this.zoomedElement.dims.height])
            .range([0, this.layout.size.diagram.height]);

        this.transitCoords.solver.x = (this.zoomedElement.solverDims.x ?
            this.layout.size.solverTree.width - this.layout.size.parentNodeWidth :
            this.layout.size.solverTree.width) / (1 - this.zoomedElement.solverDims.x);
        this.transitCoords.solver.y = this.layout.size.diagram.height /
            this.zoomedElement.solverDims.height;

        this.scales.solver.x
            .domain([this.zoomedElement.solverDims.x, 1])
            .range([this.zoomedElement.solverDims.x ? this.layout.size.parentNodeWidth :
                0, this.layout.size.solverTree.width]);
        this.scales.solver.y
            .domain([this.zoomedElement.solverDims.y,
            this.zoomedElement.solverDims.y + this.zoomedElement.solverDims.height])
            .range([0, this.layout.size.diagram.height]);

        if (this.scales.firstRun) { // first run, duplicate what we just calculated
            this.scales.firstRun = false;
            this._preservePreviousScale();

            // Update svg dimensions before width changes
            let svgDivDims = this.layout.newSvgDivDimAttribs();
            this.dom.svgDiv
                .style("width", svgDivDims.width)
                .style("height", svgDivDims.height);

            let svgElemDims = this.layout.newSvgElemDimAttribs();
            this.dom.svg
                .attr("width", svgElemDims.width)
                .attr("height", svgElemDims.height);

            this.dom.n2TopGroup.attr("transform", "translate(" +
                (this.layout.size.partitionTree.width +
                    this.layout.size.partitionTreeGap +
                    this.layout.size.svgMargin) +
                "," + this.layout.size.svgMargin + ")");

            this.dom.pTreeGroup.attr("transform", "translate(" +
                this.layout.size.svgMargin + "," + this.layout.size.svgMargin + ")");

            this.dom.pSolverTreeGroup.attr("transform", "translate(" +
                (this.layout.size.partitionTree.width +
                    this.layout.size.partitionTreeGap +
                    this.layout.size.diagram.width +
                    this.layout.size.svgMargin +
                    this.layout.size.partitionTreeGap) + "," +
                this.layout.size.svgMargin + ")");
        }
    }

    /** Update svg dimensions with transition after a new N2Layout changes
     * layout.size.partitionTree.width
     */
    _updateTransitionInfo() {
        sharedTransition = d3.transition()
            .duration(N2TransitionDefaults.duration)
            .delay(this.transitionStartDelay); // do this after intense computation
        this.transitionStartDelay = N2TransitionDefaults.startDelay;

        this.dom.svgDiv.transition(sharedTransition)
            .style("width", (this.layout.size.partitionTree.width + this.layout.size.partitionTreeGap +
                this.layout.size.diagram.width + this.layout.size.solverTree.width + 2 * this.layout.size.svgMargin + this.layout.size.partitionTreeGap) +
                this.layout.size.unit)
            .style("height", (this.layout.size.partitionTree.height + 2 * this.layout.size.svgMargin) +
                this.layout.size.unit);

        this.dom.svg.transition(sharedTransition)
            .attr("width", this.layout.size.partitionTree.width +
                this.layout.size.partitionTreeGap + this.layout.size.diagram.width + this.layout.size.solverTree.width + 2 *
                this.layout.size.svgMargin + this.layout.size.partitionTreeGap)
            .attr("height", this.layout.size.partitionTree.height + 2 * this.layout.size.svgMargin);

        this.dom.n2TopGroup.transition(sharedTransition)
            .attr("transform", "translate(" + (this.layout.size.partitionTree.width +
                this.layout.size.partitionTreeGap + this.layout.size.svgMargin) + "," + this.layout.size.svgMargin + ")");

        this.dom.pTreeGroup.transition(sharedTransition)
            .attr("transform", "translate(" + this.layout.size.svgMargin + "," + this.layout.size.svgMargin + ")");

        this.dom.n2BackgroundRect.transition(sharedTransition)
            .attr("width", this.layout.size.diagram.width).attr("height", this.layout.size.partitionTree.height);

        this.dom.pSolverTreeGroup.transition(sharedTransition)
            .attr("transform", "translate(" + (this.layout.size.partitionTree.width +
                this.layout.size.partitionTreeGap + this.layout.size.diagram.width + this.layout.size.svgMargin + this.layout.size.partitionTreeGap) + "," +
                this.layout.size.svgMargin + ")");
    }

    _createPartitionCells() {
        let selection = this.dom.pTreeGroup.selectAll(".partition_group")
            .data(this.layout.zoomedNodes, function (node) { return node.id; });

        // Create a new SVG group for each node in zoomedNodes
        let nodeEnter = selection.enter().append("svg:g")
            .attr("class", function (d) {
                return "partition_group " + this.style.getNodeClass(d);
            }.bind(this))
            .attr("transform", function (d) {
                return "translate(" +
                    this.prevScales.model.x(d.prevDims.x) + "," +
                    this.prevScales.model.y(d.prevDims.y) + ")";
            }.bind(this))
            .on("click", function (d) { LeftClick(d, this); })
            .on("contextmenu", function (d) { RightClick(d, this); })
            .on("mouseover", function (d) {
                if (this.model.abs2prom != undefined) {
                    if (d.isParam()) {
                        return this.dom.toolTip.text(this.model.abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.isUnknown()) {
                        return this.dom.toolTip.text(this.model.abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            }.bind(this))
            .on("mouseleave", function (d) {
                if (this.model.abs2prom != undefined) {
                    return this.dom.toolTip.style("visibility", "hidden");
                }
            }.bind(this))
            .on("mousemove", function () {
                if (this.model.abs2prom != undefined) {
                    return this.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            }.bind(this));

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.prevDims.width * this.prevTransitCoords.model.x;
            }.bind(this))
            .attr("height", function (d) {
                return d.prevDims.height * this.prevTransitCoords.model.y;
            }.bind(this));

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                let anchorX = d.prevDims.width * this.prevTransitCoords.model.x -
                    this.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.prevDims.height *
                    this.prevTransitCoords.model.y / 2 + ")";
            }.bind(this))
            .style("opacity", function (d) {
                if (d.depth < this.zoomedElement.depth) return 0;
                return d.textOpacity;
            }.bind(this))
            .text(this.layout.getText);

        return { 'selection': selection, 'nodeEnter': nodeEnter };
    }

    _setupPartitionTransition(d3Refs) {
        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection).transition(sharedTransition)
            .attr("class", function (d) {
                return "partition_group " + this.style.getNodeClass(d);
            }.bind(this))
            .attr("transform", function (d) {
                return "translate(" + this.scales.model.x(d.dims.x) + "," +
                    this.scales.model.y(d.dims.y) + ")";
            }.bind(this));

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.dims.width * this.transitCoords.model.x;
            }.bind(this))
            .attr("height", function (d) {
                return d.dims.height * this.transitCoords.model.y;
            }.bind(this));

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                let anchorX = d.dims.width * this.transitCoords.model.x - this.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.dims.height * this.transitCoords.model.y / 2 + ")";
            }.bind(this))
            .style("opacity", function (d) {
                if (d.depth < this.zoomedElement.depth) return 0;
                return d.textOpacity;
            }.bind(this))
            .text(this.layout.getText);
    }

    _runPartitionTransition(selection) {
        // Transition exiting nodes to the parent's new position.
        let nodeExit = selection.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + this.scales.model.x(d.dims.x) + "," +
                    this.scales.model.y(d.dims.y) + ")";
            }.bind(this))
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.dims.width * this.transitCoords.model.x;
            }.bind(this))
            .attr("height", function (d) {
                return d.dims.height * this.transitCoords.model.y;
            }.bind(this));

        nodeExit.select("text")
            .attr("transform", function (d) {
                let anchorX = d.dims.width * this.transitCoords.model.x -
                    this.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.dims.height *
                    this.transitCoords.model.y / 2 + ")";
            }.bind(this))
            .style("opacity", 0);
    }

    _createSolverCells() {
        let selection = this.dom.pSolverTreeGroup.selectAll(".solver_group")
            .data(this.layout.zoomedSolverNodes, function (d) {
                return d.id;
            });

        let nodeEnter = selection.enter().append("svg:g")
            .attr("class", function (d) {
                let solver_class = this.style.getSolverClass(this.showLinearSolverNames,
                    { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver })
                return solver_class + " " + "solver_group " + this.style.getNodeClass(d);
            }.bind(this))
            .attr("transform", function (d) {
                let x = 1.0 - d.prevSolverDims.x - d.prevSolverDims.width;
                // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return "translate(" + this.prevScales.solver.x(x) + "," +
                    this.prevScales.solver.y(d.prevSolverDims.y) + ")";
            }.bind(this))
            .on("click", function (d) { LeftClick(d, this); })
            .on("contextmenu", function (d) { RightClick(d, this); })
            .on("mouseover", function (d) {
                if (this.model.abs2prom != undefined) {
                    if (d.isParam()) {
                        return this.dom.toolTip.text(this.model.abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.isUnknown()) {
                        return this.dom.toolTip.text(this.model.abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            }.bind(this))
            .on("mouseleave", function (d) {
                if (this.model.abs2prom != undefined) {
                    return this.dom.toolTip.style("visibility", "hidden");
                }
            }.bind(this))
            .on("mousemove", function () {
                if (this.model.abs2prom != undefined) {
                    return this.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            }.bind(this));

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.prevSolverDims.width * this.prevTransitCoords.solver.x;
            }.bind(this))
            .attr("height", function (d) {
                return d.prevSolverDims.height * this.prevTransitCoords.solver.y;
            }.bind(this));

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                let anchorX = d.prevSolverDims.width * this.prevTransitCoords.solver.x -
                    this.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.prevSolverDims.height *
                    this.prevTransitCoords.solver.y / 2 + ")";
            }.bind(this))
            .style("opacity", function (d) {
                if (d.depth < this.zoomedElement.depth) return 0;
                return d.textOpacity;
            }.bind(this))
            .text(this.layout.getSolverText.bind(this));

        return ({ 'selection': selection, 'nodeEnter': nodeEnter });
    }

    _setupSolverTransition(d3Refs) {
        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection).transition(sharedTransition)
            .attr("class", function (d) {
                let solver_class = this.style.getSolverClass(this.showLinearSolverNames,
                    { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver });
                return solver_class + " " + "solver_group " + this.style.getNodeClass(d);
            }.bind(this))
            .attr("transform", function (d) {
                let x = 1.0 - d.solverDims.x - d.solverDims.width;
                // The magic for reversing the blocks on the right side

                return "translate(" + this.scales.solver.x(x) + "," +
                    this.scales.solver.y(d.solverDims.y) + ")";
            }.bind(this));

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.solverDims.width * this.transitCoords.solver.x;
            }.bind(this))
            .attr("height", function (d) {
                return d.solverDims.height * this.transitCoords.solver.y;
            }.bind(this));

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                let anchorX = d.solverDims.width * this.transitCoords.solver.x -
                    this.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.solverDims.height *
                    this.transitCoords.solver.y / 2 + ")";
            }.bind(this))
            .style("opacity", function (d) {
                if (d.depth < this.zoomedElement.depth) return 0;
                return d.textOpacity;
            }.bind(this))
            .text(this.layout.getSolverText.bind(this));
    }

    _runSolverTransition(selection) {
        // Transition exiting nodes to the parent's new position.
        let nodeExit = selection.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + this.scales.solver.x(d.solverDims.x) + "," +
                    this.scales.solver.y(d.solverDims.y) + ")";
            }.bind(this))
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.solverDims.width * this.transitCoords.solver.x;
            }.bind(this))
            .attr("height", function (d) {
                return d.solverDims.height * this.transitCoords.solver.y;
            }.bind(this));

        nodeExit.select("text")
            .attr("transform", function (d) {
                let anchorX = d.solverDims.width * this.transitCoords.solver.x -
                    this.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.solverDims.height *
                    this.transitCoords.solver.y / 2 + ")";
            }.bind(this))
            .style("opacity", 0);
    }

    clearArrows() {
        this.dom.n2TopGroup.selectAll("[class^=n2_hover_elements]").remove();
    }

    /**
     * Refresh the diagram when something has visually changed.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    update(computeNewTreeLayout = true) {
        this._updateUI();

        // Compute the new tree layout if necessary.
        if (computeNewTreeLayout) {
            this.clearArrows();

            this.layout = new N2Layout(this.model, this.zoomedElement, this.showLinearSolverNames);
            this._updateClickedIndices();
            this.matrix = new N2Matrix(this.layout.visibleNodes, this.model, this.layout,
                this.dom.n2Groups, this.matrix.nodeSize);
        }

        this._updateScale();
        this._updateTransitionInfo();

        let d3Refs = this._createPartitionCells();
        this._setupPartitionTransition(d3Refs);
        this._runPartitionTransition(d3Refs.selection);

        d3Refs = this._createSolverCells();
        this._setupSolverTransition(d3Refs);
        this._runSolverTransition(d3Refs.selection);

        this.matrix.draw();
    }
}