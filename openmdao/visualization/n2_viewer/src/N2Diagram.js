/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the N2 matrix, and the N2 matrix grid itself are
 * all member objects.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {N2Style} style Manages N2-related styles and functions.
 * @property {N2Layout} layout Sizes and positions of visible elements.
 * @property {N2Matrix} matrix Manages the grid of visible model parameters.
 * @property {N2TreeNode} zoomedElement The element the diagram is currently based on.
 * @property {N2TreeNode} zoomedElementPrev Reference to last zoomedElement.
 * @property {Object} dom Container for references to web page elements.
 * @property {Object} dom.parentDiv The outermost div we work with.
 * @property {Object} dom.d3ContentDiv The div containing all of the diagram's content.
 * @property {Object} dom.svgDiv The div containing the SVG element.
 * @property {Object} dom.svg The SVG element.
 * @property {Object} dom.svgStyle Object where SVG style changes can be made.
 * @property {Object} dom.toolTip Div to display tooltips.
 * @property {Object} dom.n2TopGroup The outermost div of N2 itself.
 * @property {Object} dom.n2Groups References to <g> SVG elements.
 * @property {Boolean} showPath If we're currently displaying the path of the zoomed element.
 * @property {number} chosenCollapseDepth The selected depth from the drop-down.
 * @property {Object} scales Scalers in the X and Y directions to associate the relative
 *   position of an element to actual pixel coordinates.
 * @property {Object} transitCoords
 */
class N2Diagram {
    constructor(modelJSON) {
        this.model = new ModelData(modelJSON);
        this.zoomedElement = this.zoomedElementPrev = this.model.root;
        this.showPath = false;

        // Assign this way because defaultDims is read-only.
        this.dims = JSON.parse(JSON.stringify(defaultDims));

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

        this.chosenCollapseDepth = -1;

        this.showLinearSolverNames = true;

        this.style = new N2Style(this.dom.svgStyle, this.dims.size.font);
        this.layout = new N2Layout(this.model, this.zoomedElement, this.showLinearSolverNames, this.dims);
        this.ui = new N2UserInterface(this);

        this._setupSvgElements();

        this.matrix = new N2Matrix(this.model, this.layout, this.dom.n2Groups,
            true, this.ui.findRootOfChangeFunction);

        // TODO: Move to N2Layout
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

        // TODO: Move to N2Layout
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
        for (let gName of ['elements', 'gridlines', 'componentBoxes', 'arrows', 'dots', 'highlights']) {
            this.dom.n2Groups[gName] = this.dom.n2TopGroup.append('g').attr('id', 'n2' + gName);
        };
    }

    /**
     * Make a copy of the previous transit coordinates and linear scalers before
     * setting new ones.
     * TODO: Move to N2Layout
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

    // TODO: Move to N2Layout
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

    _createPartitionCells() {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = this.dom.pTreeGroup.selectAll(".partition_group")
            .data(this.layout.zoomedNodes, function (node) { return node.id; });

        // Create a new SVG group for each node in zoomedNodes
        let nodeEnter = selection.enter().append("svg:g")
            .attr("class", function (d) {
                return "partition_group " + self.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" +
                    self.prevScales.model.x(d.prevDims.x) + "," +
                    self.prevScales.model.y(d.prevDims.y) + ")";
            })
            .on("click", function (d) { self.ui.leftClick(d); })
            .on("contextmenu", function (d) { self.ui.rightClick(d, this); })
            .on("mouseover", function (d) {
                if (self.model.abs2prom != undefined) {
                    if (d.isParam()) {
                        return self.dom.toolTip.text(
                            self.model.abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.isUnknown()) {
                        return self.dom.toolTip.text(
                            self.model.abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                if (self.model.abs2prom != undefined) {
                    return self.dom.toolTip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                if (self.model.abs2prom != undefined) {
                    return self.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.prevDims.width * self.prevTransitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.prevDims.height * self.prevTransitCoords.model.y;
            });

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                let anchorX = d.prevDims.width * self.prevTransitCoords.model.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.prevDims.height *
                    self.prevTransitCoords.model.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getText);

        return { 'selection': selection, 'nodeEnter': nodeEnter };
    }

    _setupPartitionTransition(d3Refs) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", function (d) {
                return "partition_group " + self.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + self.scales.model.x(d.dims.x) + "," +
                    self.scales.model.y(d.dims.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.dims.width * self.transitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.dims.height * self.transitCoords.model.y;
            });

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                let anchorX = d.dims.width * self.transitCoords.model.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.dims.height *
                    self.transitCoords.model.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getText);
    }

    _runPartitionTransition(selection) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        // Transition exiting nodes to the parent's new position.
        let nodeExit = selection.exit().transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + self.scales.model.x(d.dims.x) + "," +
                    self.scales.model.y(d.dims.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.dims.width * self.transitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.dims.height * self.transitCoords.model.y;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                let anchorX = d.dims.width * self.transitCoords.model.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.dims.height *
                    self.transitCoords.model.y / 2 + ")";
            })
            .style("opacity", 0);
    }

    _createSolverCells() {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = self.dom.pSolverTreeGroup.selectAll(".solver_group")
            .data(self.layout.zoomedSolverNodes, function (d) {
                return d.id;
            });

        let nodeEnter = selection.enter().append("svg:g")
            .attr("class", function (d) {
                let solver_class = self.style.getSolverClass(self.showLinearSolverNames,
                    { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver });
                if (!self.showLinearSolverNames && d.hasOwnProperty("solve_subsystems") && d.solve_subsystems){
                    return solver_class + "_solve_subs" + " " + "solver_group " + self.style.getNodeClass(d);
                } else {
                    return solver_class + " " + "solver_group " + self.style.getNodeClass(d);
                }
            })
            .attr("transform", function (d) {
                let x = 1.0 - d.prevSolverDims.x - d.prevSolverDims.width;
                // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return "translate(" + self.prevScales.solver.x(x) + "," +
                    self.prevScales.solver.y(d.prevSolverDims.y) + ")";
            })
            .on("click", function (d) { self.ui.leftClick(d); })
            .on("contextmenu", function (d) { self.ui.rightClick(d, this); })
            .on("mouseover", function (d) {
                if (self.model.abs2prom != undefined) {
                    if (d.isParam()) {
                        return self.dom.toolTip.text(self.model.abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.isUnknown()) {
                        return self.dom.toolTip.text(self.model.abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                if (self.model.abs2prom != undefined) {
                    return self.dom.toolTip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                if (self.model.abs2prom != undefined) {
                    return self.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.prevSolverDims.width * self.prevTransitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.prevSolverDims.height * self.prevTransitCoords.solver.y;
            });

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                let anchorX = d.prevSolverDims.width * self.prevTransitCoords.solver.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.prevSolverDims.height *
                    self.prevTransitCoords.solver.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getSolverText.bind(self.layout));

        return ({ 'selection': selection, 'nodeEnter': nodeEnter });
    }

    _setupSolverTransition(d3Refs) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", function (d) {
                let solver_class = self.style.getSolverClass(self.showLinearSolverNames,
                    { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver });
                if (!self.showLinearSolverNames && d.hasOwnProperty("solve_subsystems") && d.solve_subsystems){
                    return solver_class + "_solve_subs" + " " + "solver_group " + self.style.getNodeClass(d);
                } else {
                    return solver_class + " " + "solver_group " + self.style.getNodeClass(d);
                }
            })
            .attr("transform", function (d) {
                let x = 1.0 - d.solverDims.x - d.solverDims.width;
                // The magic for reversing the blocks on the right side

                return "translate(" + self.scales.solver.x(x) + "," +
                    self.scales.solver.y(d.solverDims.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.solverDims.width * self.transitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.solverDims.height * self.transitCoords.solver.y;
            });

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                let anchorX = d.solverDims.width * self.transitCoords.solver.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.solverDims.height *
                    self.transitCoords.solver.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getSolverText.bind(self.layout));
    }

    _runSolverTransition(selection) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        // Transition exiting nodes to the parent's new position.
        let nodeExit = selection.exit()
        .transition(sharedTransition)
            .attr("transform", function (d) {
                return "translate(" + self.scales.solver.x(d.solverDims.x) + "," +
                    self.scales.solver.y(d.solverDims.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.solverDims.width * self.transitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.solverDims.height * self.transitCoords.solver.y;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                let anchorX = d.solverDims.width * self.transitCoords.solver.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.solverDims.height *
                    self.transitCoords.solver.y / 2 + ")";
            })
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
        this.ui.update();

        // Compute the new tree layout if necessary.
        if (computeNewTreeLayout) {
            this.clearArrows();

            this.layout = new N2Layout(this.model, this.zoomedElement,
                this.showLinearSolverNames, this.dims);
            this.ui.updateClickedIndices();
            this.matrix = new N2Matrix(this.model, this.layout,
                this.dom.n2Groups, this.ui.lastClickWasLeft,
                this.ui.findRootOfChangeFunction, this.matrix.nodeSize);
        }

        this._updateScale();
        this.layout.updateTransitionInfo(this.dom, this.transitionStartDelay);

        let d3Refs = this._createPartitionCells();
        this._setupPartitionTransition(d3Refs);
        this._runPartitionTransition(d3Refs.selection);

        d3Refs = this._createSolverCells();
        this._setupSolverTransition(d3Refs);
        this._runSolverTransition(d3Refs.selection);

        this.matrix.draw();
    }

    /**
     * Adjust the height and corresponding width of the diagram based on user input.
     * @param {number} height The new height in pixels.
     */
    verticalResize(height) {
        for (let i = 600; i <= 1000; i += 50) {
            let newText = (i == height) ? ("<b>" + i + "px</b>") : (i + "px");
            this.dom.parentDiv.querySelector("#idVerticalResize" + i + "px").innerHTML = newText;
        }
        for (let i = 2000; i <= 4000; i += 1000) {
            let newText = (i == height) ? ("<b>" + i + "px</b>") : (i + "px");
            this.dom.parentDiv.querySelector("#idVerticalResize" + i + "px").innerHTML = newText;
        }
        this.clearArrows();

        this.dims.size.diagram.height =
            this.dims.size.diagram.width =
            this.dims.size.partitionTree.height =
            this.dims.size.solverTree.height = height;

        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
        this.style.updateSvgStyle(this.layout.size.font);
        this.update();
    }

    /**
     * Adjust the font size of all text in the diagram based on user input.
     * @param {number} fontSize The new font size in pixels.
     */
    fontSizeSelectChange(fontSize) {
        for (let i = 8; i <= 14; ++i) {
            let newText = (i == fontSize) ? ("<b>" + i + "px</b>") : (i + "px");
            this.dom.parentDiv.querySelector("#idFontSize" + i + "px").innerHTML = newText;
        }
        this.dims.size.font = fontSize;
        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
        this.style.updateSvgStyle(fontSize);
        this.update();
    }

    /**
     * Since the matrix can be destroyed and recreated, use this to invoke the callback
     * rather than setting one up that points directly to a specific matrix.
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseOverOnDiagonal(cell) {
        this.matrix.mouseOverOnDiagonal(cell);
    }

    /**
     * Since the matrix can be destroyed and recreated, use this to invoke the callback
     * rather than setting one up that points directly to a specific matrix.
     */
    mouseOverOffDiagonal(cell) {
        this.matrix.mouseOverOffDiagonal(cell);
    }

    /** When the mouse leaves a cell, remove all temporary arrows. */
    mouseOut() {
        this.dom.n2TopGroup.selectAll(".n2_hover_elements").remove();
    }

    /**
     * When the mouse if left-clicked on a cell, change their CSS class
     * so they're not removed when the mouse moves out.
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseClick(cell) {
        let newClassName = "n2_hover_elements_" + cell.row + "_" + cell.col;
        let selection = this.dom.n2TopGroup.selectAll("." + newClassName);
        if (selection.size() > 0) {
            selection.remove();
        }
        else {
            this.dom.n2TopGroup
                .selectAll("path.n2_hover_elements, circle.n2_hover_elements")
                .attr("class", newClassName);
        }
    }

    /**
     * Place member mouse callbacks in an object for easy reference.
     * @returns {Object} Object containing each of the functions.
    */
    getMouseFuncs() {
        let self = this;

        let mf = {
            'overOffDiag': self.mouseOverOffDiagonal.bind(self),
            'overOnDiag': self.mouseOverOnDiagonal.bind(self),
            'out': self.mouseOut.bind(self),
            'click': self.mouseClick.bind(self)
        }

        return mf;
    }
}