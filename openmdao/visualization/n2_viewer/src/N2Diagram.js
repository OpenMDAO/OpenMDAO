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
 * @property {Object} dom.n2OuterGroup The outermost div of N2 itself.
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
        this.search = new N2Search(this.zoomedElement, this.model.root);

        this.ui = new N2UserInterface(this);

        this._setupSvgElements();

        this.matrix = new N2Matrix(this.model, this.layout, this.dom.n2Groups,
            true, this.ui.findRootOfChangeFunction);

        // this.matrixMax = this.matrix;

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
        this.zoomedElement = newZoomedElement;
        this.layout.zoomedElement = this.zoomedElement;
    }

    /** 
     * Add several elements, mostly groups, to the primary SVG element to
     * be populated later. Stored in the .doms property.
     */
    _setupSvgElements() {
        this.dom.clips = {
            'partitionTree': this.dom.svg.select("#partitionTreeClip > rect"),
            'n2Matrix': this.dom.svg.select("#n2MatrixClip > rect"),
            'solverTree': this.dom.svg.select("#solverTreeClip > rect")
        };

        // ids given just so it is easier to see in Chrome dev tools when debugging
        this.dom.n2OuterGroup = this.dom.svg.append('g')
            .attr('id', 'n2outer');

        this.dom.n2InnerGroup = this.dom.n2OuterGroup.append('g')
            .attr('id', 'n2inner');

        this.dom.pTreeGroup = this.dom.svg.append('g')
            .attr('id', 'tree')
            .attr('clip-path', 'url(#partitionTreeClip');

        this.dom.pSolverTreeGroup = this.dom.svg.append('g')
            .attr('id', 'solver_tree')
            .attr('clip-path', 'url(#solverTreeClip');

        this.dom.n2BackgroundRect = this.dom.n2InnerGroup.append('rect')
            .attr('id', 'backgroundRect')
            .attr('class', 'background');

        this.dom.n2Groups = {};
        for (let gName of ['elements', 'gridlines', 'componentBoxes',
            'dots', 'highlights', 'arrows']) {
            this.dom.n2Groups[gName] =
                this.dom.n2InnerGroup.append('g')
                    .attr('id', 'n2' + gName);
        };

        for (let clippedGrp of ['elements', 'gridlines', 'componentBoxes', 'dots']) {
            this.dom.n2Groups[clippedGrp].attr('clip-path', 'url(#n2MatrixClip)');
        }

        let ogg = {};
        for (let oName of ['top', 'left', 'right', 'bottom']) {
            ogg[oName] = this.dom.n2OuterGroup.append('g')
                .attr('id', 'n2' + oName)
                .attr('class', 'offgridLabel');
        }
        this.dom.n2Groups.offgrid = ogg;

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
        this.transitCoords.model.y = this.layout.size.partitionTree.height /
            this.zoomedElement.dims.height;

        this.scales.model.x
            .domain([this.zoomedElement.dims.x, 1])
            .range([this.zoomedElement.dims.x ? this.layout.size.parentNodeWidth : 0,
            this.layout.size.partitionTree.width]);
        this.scales.model.y
            .domain([this.zoomedElement.dims.y, this.zoomedElement.dims.y +
                this.zoomedElement.dims.height])
            .range([0, this.layout.size.partitionTree.height]);

        this.transitCoords.solver.x = (this.zoomedElement.solverDims.x ?
            this.layout.size.solverTree.width - this.layout.size.parentNodeWidth :
            this.layout.size.solverTree.width) / (1 - this.zoomedElement.solverDims.x);
        this.transitCoords.solver.y = this.layout.size.solverTree.height /
            this.zoomedElement.solverDims.height;

        this.scales.solver.x
            .domain([this.zoomedElement.solverDims.x, 1])
            .range([this.zoomedElement.solverDims.x ? this.layout.size.parentNodeWidth :
                0, this.layout.size.solverTree.width]);
        this.scales.solver.y
            .domain([this.zoomedElement.solverDims.y,
            this.zoomedElement.solverDims.y + this.zoomedElement.solverDims.height])
            .range([0, this.layout.size.solverTree.height]);

        if (this.scales.firstRun) { // first run, duplicate what we just calculated
            this.scales.firstRun = false;
            this._preservePreviousScale();

            // Update svg dimensions before size changes
            let outerDims = this.layout.newOuterDims();
            let innerDims = this.layout.newInnerDims();

            this.dom.svgDiv
                .style("width", outerDims.width + this.dims.size.unit)
                .style("height", outerDims.height + this.dims.size.unit);

            this.dom.svg
                .attr("width", outerDims.width)
                .attr("height", outerDims.height)
                .attr("transform", "translate(0 0)");

            this.dom.pTreeGroup
                .attr("height", innerDims.height)
                .attr("width", this.dims.size.partitionTree.width)
                .attr("transform", "translate(0 " + innerDims.margin + ")");

            this.dom.n2OuterGroup
                .attr("height", outerDims.height)
                .attr("width", outerDims.height)
                .attr("transform", "translate(" +
                    (this.dims.size.partitionTree.width) + " 0)");

            this.dom.n2InnerGroup.transition(sharedTransition)
                .attr("height", innerDims.height)
                .attr("width", innerDims.height)
                .attr("transform", "translate(" + innerDims.margin + " " + innerDims.margin + ")");

            this.dom.n2BackgroundRect
                .attr("width", innerDims.height)
                .attr("height", innerDims.height)
                .attr("transform", "translate(0 0)");

            this.dom.pSolverTreeGroup
                .attr("height", innerDims.height)
                .attr("transform", "translate(" + (this.dims.size.partitionTree.width +
                    innerDims.margin +
                    innerDims.height +
                    innerDims.margin) + " " +
                    innerDims.margin + ")");

            let offgridHeight = this.dims.size.font + 2;
            this.dom.n2Groups.offgrid.top
                .attr("transform", "translate(" + innerDims.margin + " 0)")
                .attr("width", innerDims.height)
                .attr("height", offgridHeight);

                /*
            this.dom.n2Groups.offgrid.top
                .append("text")
                .attr("x", 0).attr("y", 0)
                // .attr("transform", "translate(0 0)")
                //.attr("width", innerDims.height)
                //.attr("height", offgridHeight)
                //.style("fill", "black")
                .text("HOWDY.HOWDY.HOWDY.HOWDY");
                */

            this.dom.n2Groups.offgrid.bottom
                .attr("transform", "translate(0 " + innerDims.height + offgridHeight + ")")
                .attr("width", outerDims.height)
                .attr("height", offgridHeight);
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
                    self.prevScales.model.x(d.prevDims.x) + " " +
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
                return "translate(" + anchorX + " " + d.prevDims.height *
                    self.prevTransitCoords.model.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getText.bind(self.layout));

        return { 'selection': selection, 'nodeEnter': nodeEnter };
    }

    _setupPartitionTransition(d3Refs) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        this.dom.clips.partitionTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.partitionTree.height);

        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", function (d) {
                return "partition_group " + self.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                return "translate(" + self.scales.model.x(d.dims.x) + " " +
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
                return "translate(" + anchorX + " " + d.dims.height *
                    self.transitCoords.model.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getText.bind(self.layout));
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
                return solver_class + " solver_group " + self.style.getNodeClass(d);
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

        this.dom.clips.solverTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.solverTree.height);

        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", function (d) {
                let solver_class = self.style.getSolverClass(self.showLinearSolverNames,
                    { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver });
                return solver_class + " solver_group " + self.style.getNodeClass(d);
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
        this.dom.n2OuterGroup.selectAll("[class^=n2_hover_elements]").remove();
    }

    /**
     * Refresh the diagram when something has visually changed.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    update(computeNewTreeLayout = true) {
        this.ui.update();
        this.search.update(this.zoomedElement, this.model.root);

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

        let d3PartRefs = this._createPartitionCells();
        this._setupPartitionTransition(d3PartRefs);
        this._runPartitionTransition(d3PartRefs.selection);

        let d3SolverRefs = this._createSolverCells();
        this._setupSolverTransition(d3SolverRefs);
        this._runSolverTransition(d3SolverRefs.selection);

        this.matrix.draw();
    }

    /**
     * Updates the intended dimensions of the diagrams and font, but does
     * not perform rendering itself.
     * @param {number} height The base height of the diagram without margins.
     * @param {number} fontSize The new size of the font.
     */
    updateSizes(height, fontSize) {
        let gapSize = fontSize + 4;

        this.dims.size.n2matrix.margin = gapSize;
        this.dims.size.partitionTreeGap = gapSize;

        this.dims.size.n2matrix.height =
            this.dims.size.n2matrix.width = // Match base height, keep it looking square
            this.dims.size.partitionTree.height =
            this.dims.size.solverTree.height = height;

        this.dims.size.font = fontSize;
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
        this.updateSizes(height, this.dims.size.font);

        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
        this.style.updateSvgStyle(this.dims.size.font);
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

        this.updateSizes(this.dims.size.n2matrix.height, fontSize);

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
        if (this.matrix.cellExists(cell))
            this.matrix.mouseOverOnDiagonal(cell);
    }

    /**
     * Since the matrix can be destroyed and recreated, use this to invoke the callback
     * rather than setting one up that points directly to a specific matrix.
     */
    mouseOverOffDiagonal(cell) {
        if (this.matrix.cellExists(cell))
            this.matrix.mouseOverOffDiagonal(cell);
    }

    /** When the mouse leaves a cell, remove all temporary arrows. */
    mouseOut() {
        this.dom.n2OuterGroup.selectAll(".n2_hover_elements").remove();
        d3.selectAll("div.offgrid")
            .style("visibility", "hidden")
            .node().innerHTML = '';
    }

    /**
     * When the mouse if left-clicked on a cell, change their CSS class
     * so they're not removed when the mouse moves out.
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseClick(cell) {
        let newClassName = "n2_hover_elements_" + cell.row + "_" + cell.col;
        let selection = this.dom.n2OuterGroup.selectAll("." + newClassName);
        if (selection.size() > 0) {
            selection.remove();
        }
        else {
            this.dom.n2OuterGroup
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