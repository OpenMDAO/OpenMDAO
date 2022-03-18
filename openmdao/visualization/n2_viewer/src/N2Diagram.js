/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the N2 matrix, and the N2 matrix grid itself are
 * all member objects.
 * @typedef N2Diagram
 * @property {ModelData} model Processed model data received from Python.
 * @property {N2Style} style Manages N2-related styles and functions.
 * @property {N2Layout} layout Sizes and positions of visible elements.
 * @property {N2Matrix} matrix Manages the grid of visible model parameters.
 * @property {OmTreeNode} zoomedElement The element the diagram is currently based on.
 * @property {OmTreeNode} zoomedElementPrev Reference to last zoomedElement.
 * @property {Object} dom Container for references to web page elements.
 * @property {Object} dom.parentDiv The outermost div we work with.
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
        this.modelData = modelJSON;
        this.model = new OmModelData(modelJSON);
        this.zoomedElement = this.zoomedElementPrev = this.model.root;
        this.showPath = false;
        this.manuallyResized = false; // If the diagram has been sized by the user

        // Assign this way because defaultDims is read-only.
        this.dims = JSON.parse(JSON.stringify(defaultDims));

        this._referenceD3Elements();
        this.transitionStartDelay = N2TransitionDefaults.startDelay;
        this.chosenCollapseDepth = -1;
        this.showLinearSolverNames = true;
        this.showSolvers = true ;

        this.style = new N2Style(this.dom.svgStyle, this.dims.size.font);
        this.layout = new N2Layout(this.model, this.zoomedElement,
            this.showLinearSolverNames, this.showSolvers, this.dims);
        this.search = new N2Search(this.zoomedElement, this.model.root);
        this.ui = new N2UserInterface(this);
        // Keep track of arrows to show and hide them
        this.arrowMgr = new N2ArrowManager(this.dom.n2Groups);
        this.matrix = new N2Matrix(this.model, this.layout, this.dom.n2Groups,
            this.arrowMgr, true, this.ui.findRootOfChangeFunction);

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
            'model': {
                'x': 0,
                'y': 0
            },
            'solver': {
                'x': 0,
                'y': 0
            }
        };

        this.prevTransitCoords = {
            'unit': 'px',
            'model': {
                'x': 0,
                'y': 0
            },
            'solver': {
                'x': 0,
                'y': 0
            }
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
        let svgBlob = new Blob([svgData], {
            type: "image/svg+xml;charset=utf-8"
        });
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
     * Recurse and pull state info from model for saving.
     * @param {Array} dataList Array of objects with state info for each node.
     * @param {OmTreeNode} node The current node being examined.
     */
    getSubState(dataList, node = this.model.root) {
        if (node.isFilter()) return; // Ignore state for N2FilterNodes

        dataList.push(node.getStateForSave());

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.getSubState(dataList, child);
            }
        }
    }

    /**
     * Recurse and set state info into model.
     * @param {Array} dataList Array of objects with state info for each node. 
     * @param {OmTreeNode} node The node currently being restored.
     */
    setSubState(dataList, node = this.model.root) {
        if (node.isFilter()) return; // Ignore state for N2FilterNodes

        node.setStateFromLoad(dataList.pop());

        // Get rid of any existing filters before processing children, as they'll
        // be populated while processing the state of each child node.
        if (node.hasFilters()) {
            node.filter.inputs.wipe();
            node.filter.outputs.wipe();
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.setSubState(dataList, child);
            }
        }
    }

    /*
     * Recurse and return node given id.
     */
    findNodeById(id, node = this.model.root) {
        if (id == node.id) {
            return node;
        }
        else if (node.hasChildren()) {
            for (const child of node.children) {
                let found = this.findNodeById(id, child);
                if (found) {
                    return found;
                }
            }
        }
        else {
            return false;
        }
        return false;
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
     * Setup internal references to D3 objects so we can avoid running
     * d3.select() over and over later.
     */
    _referenceD3Elements() {
        this.dom = {
            'parentDiv': document.getElementById("ptN2ContentDivId"),
            'svgDiv': d3.select("#svgDiv"),
            'svg': d3.select("#svgId"),
            'svgStyle': d3.select("#svgId style"),
            'toolTip': d3.select(".tool-tip"),
            'arrowMarker': d3.select("#arrow"),
            'n2OuterGroup': d3.select('g#n2outer'),
            'n2InnerGroup': d3.select('g#n2inner'),
            'pTreeGroup': d3.select('g#tree'),
            'highlightBar': d3.select('g#highlight-bar'),
            'pSolverTreeGroup': d3.select('g#solver_tree'),
            'n2BackgroundRect': d3.select('g#n2inner rect'),
            'waiter': d3.select('#waiting-container'),
            'clips': {
                'partitionTree': d3.select("#partitionTreeClip > rect"),
                'n2Matrix': d3.select("#n2MatrixClip > rect"),
                'solverTree': d3.select("#solverTreeClip > rect")
            }
        };

        const n2Groups = {};
        this.dom.n2InnerGroup.selectAll('g').each(function () {
            const d3elem = d3.select(this);
            const name = new String(d3elem.attr('id')).replace(/n2/, '');
            n2Groups[name] = d3elem;
        })
        this.dom.n2Groups = n2Groups;

        const offgrid = {};
        this.dom.n2OuterGroup.selectAll('g.offgridLabel').each(function () {
            const d3elem = d3.select(this);
            const name = new String(d3elem.attr('id')).replace(/n2/, '');
            offgrid[name] = d3elem;
        })
        this.dom.n2Groups.offgrid = offgrid;
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

        this.transitCoords.model.x = (this.zoomedElement.draw.dims.x ?
            this.layout.size.partitionTree.width - this.layout.size.parentNodeWidth :
            this.layout.size.partitionTree.width) / (1 - this.zoomedElement.draw.dims.x);
        this.transitCoords.model.y = this.layout.size.partitionTree.height /
            this.zoomedElement.draw.dims.height;

        this.scales.model.x
            .domain([this.zoomedElement.draw.dims.x, 1])
            .range([this.zoomedElement.draw.dims.x ? this.layout.size.parentNodeWidth : 0,
            this.layout.size.partitionTree.width
            ]);
        this.scales.model.y
            .domain([this.zoomedElement.draw.dims.y, this.zoomedElement.draw.dims.y +
                this.zoomedElement.draw.dims.height
            ])
            .range([0, this.layout.size.partitionTree.height]);

        this.transitCoords.solver.x = (this.zoomedElement.draw.solverDims.x ?
            this.layout.size.solverTree.width - this.layout.size.parentNodeWidth :
            this.layout.size.solverTree.width) / (1 - this.zoomedElement.draw.solverDims.x);
        this.transitCoords.solver.y = this.layout.size.solverTree.height /
            this.zoomedElement.draw.solverDims.height;

        this.scales.solver.x
            .domain([this.zoomedElement.draw.solverDims.x, 1])
            .range([this.zoomedElement.draw.solverDims.x ? this.layout.size.parentNodeWidth :
                0, this.layout.size.solverTree.width
            ]);
        this.scales.solver.y
            .domain([this.zoomedElement.draw.solverDims.y,
            this.zoomedElement.draw.solverDims.y + this.zoomedElement.draw.solverDims.height
            ])
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

            this.dom.highlightBar
                .attr("height", innerDims.height)
                .attr("width", "8")
                .attr("transform", "translate(" + this.dims.size.partitionTree.width + 1 + " " + innerDims.margin + ")");

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

            this.dom.n2Groups.offgrid.bottom
                .attr("transform", "translate(0 " + innerDims.height + offgridHeight + ")")
                .attr("width", outerDims.height)
                .attr("height", offgridHeight);
        }
    }

    leftClickSelector(obj, node) {
        switch (this.ui.click.clickEffect) {
            case N2Click.ClickEffect.NodeInfo:
                this.ui.nodeInfoBox.pin();
                break;
            case N2Click.ClickEffect.Collapse:
                this.ui.rightClick(node, obj);
                break;
            case N2Click.ClickEffect.Filter:
                const color = d3.select(obj).select('rect').style('fill');
                this.ui.altRightClick(node, color);
                break;
            default:
                this.ui.leftClick(node);
        }
    }

    _createPartitionCells() {
        const self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = this.dom.pTreeGroup.selectAll(".partition_group")
            .data(this.layout.zoomedNodes, function (node) {
                return node.id;
            });

        // Create a new SVG group for each node in zoomedNodes
        let nodeEnter = selection.enter().append("g")
            .attr("class", function (d) {
                return `partition_group ${self.style.getNodeClass(d)}`;
            })
            .attr("transform", function (d) {
                return "translate(" +
                    self.prevScales.model.x(d.draw.prevDims.x) + " " +
                    self.prevScales.model.y(d.draw.prevDims.y) + ")";
            })
            .on("click", function (d) {
                self.leftClickSelector(this, d);
            })
            .on("contextmenu", function (d) {
                if (d3.event.altKey) {
                    const color = d3.select(this).select('rect').style('fill');
                    self.ui.altRightClick(d, color);
                }
                else {
                    self.ui.rightClick(d, this);
                }
            })
            .on("mouseover", function (d) {
                self.ui.nodeInfoBox.update(d3.event, d, d3.select(this).select('rect').style('fill'));
            })
            .on("mouseleave", function () {
                self.ui.nodeInfoBox.clear();
            })
            .on("mousemove", function () {
                self.ui.nodeInfoBox.moveNearMouse(d3.event);
            });

        nodeEnter.append("rect")
            .attr("width", function (d) {
                return d.draw.prevDims.width * self.prevTransitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.draw.prevDims.height * self.prevTransitCoords.model.y;
            })
            .attr("id", function (d) {
                return OmTreeNode.pathToId(d.path);
            })
            .attr('rx', 12)
            .attr('ry', 12);

        nodeEnter.append("text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                let anchorX = d.draw.prevDims.width * self.prevTransitCoords.model.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + " " +
                    (d.draw.prevDims.height * self.prevTransitCoords.model.y / 2) + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getText.bind(self.layout));

        return {
            'selection': selection,
            'nodeEnter': nodeEnter
        };
    }

    _setupPartitionTransition(d3Refs) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        this.dom.clips.partitionTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.partitionTree.height);

        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", function (d) {
                return `partition_group ${self.style.getNodeClass(d)}`;
            })
            .attr("transform", function (d) {
                return "translate(" + self.scales.model.x(d.draw.dims.x) + " " +
                    self.scales.model.y(d.draw.dims.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.draw.dims.width * self.transitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.draw.dims.height * self.transitCoords.model.y;
            })
            .attr('rx', 12)
            .attr('ry', 12);

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                let anchorX = d.draw.dims.width * self.transitCoords.model.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + " " + (d.draw.dims.height *
                    self.transitCoords.model.y / 2) + ")";
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
                return "translate(" + self.scales.model.x(d.draw.dims.x) + "," +
                    self.scales.model.y(d.draw.dims.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.draw.dims.width * self.transitCoords.model.x;
            })
            .attr("height", function (d) {
                return d.draw.dims.height * self.transitCoords.model.y;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                let anchorX = d.draw.dims.width * self.transitCoords.model.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + (d.draw.dims.height *
                    self.transitCoords.model.y / 2) + ")";
            })
            .style("opacity", 0);
    }

    _createSolverCells() {
        const self = this; // For callbacks that change "this". Alternative to using .bind().

        let selection = self.dom.pSolverTreeGroup.selectAll(".solver_group")
            .data(self.layout.zoomedSolverNodes, function (d) {
                return d.id;
            });

        let nodeEnter = selection.enter().append("svg:g")
            .attr("class", function (d) {
                const solver_class = self.style.getSolverClass(self.showLinearSolverNames, {
                    'linear': d.linear_solver,
                    'nonLinear': d.nonlinear_solver
                });
                return `${solver_class} solver_group ${self.style.getNodeClass(d)}`;
            })
            .attr("transform", function (d) {
                let x = 1.0 - d.draw.prevSolverDims.x - d.draw.prevSolverDims.width;
                // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return "translate(" + self.prevScales.solver.x(x) + "," +
                    self.prevScales.solver.y(d.draw.prevSolverDims.y) + ")";
            })
            .on("click", function (d) {
                self.leftClickSelector(this, d);
            })
            .on("contextmenu", function (d) {
                self.ui.rightClick(d, this);
            })
            .on("mouseover", function (d) {
                self.ui.nodeInfoBox.update(d3.event, d, d3.select(this).select('rect').style('fill'), true)

                if (self.model.abs2prom != undefined) {
                    if (d.isInput()) {
                        return self.dom.toolTip.text(self.model.abs2prom.input[d.absPathName])
                            .style("visibility", "visible");
                    }
                    if (d.isOutput()) {
                        return self.dom.toolTip.text(self.model.abs2prom.output[d.absPathName])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", function (d) {
                self.ui.nodeInfoBox.clear();

                if (self.model.abs2prom != undefined) {
                    self.dom.toolTip.style("visibility", "hidden");
                }
            })
            .on("mousemove", function () {
                self.ui.nodeInfoBox.moveNearMouse(d3.event);

                if (self.model.abs2prom != undefined) {
                    self.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeEnter.append("svg:rect")
            .attr("width", function (d) {
                return d.draw.prevSolverDims.width * self.prevTransitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.draw.prevSolverDims.height * self.prevTransitCoords.solver.y;
            })
            .attr("id", function (d) {
                return d.absPathName.replace(/\./g, '_');
            });

        nodeEnter.append("svg:text")
            .attr("dy", ".35em")
            .attr("transform", function (d) {
                let anchorX = d.draw.prevSolverDims.width * self.prevTransitCoords.solver.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.draw.prevSolverDims.height *
                    self.prevTransitCoords.solver.y / 2 + ")";
            })
            .style("opacity", function (d) {
                if (d.depth < self.zoomedElement.depth) return 0;
                return d.textOpacity;
            })
            .text(self.layout.getSolverText.bind(self.layout));

        return ({
            'selection': selection,
            'nodeEnter': nodeEnter
        });
    }

    _setupSolverTransition(d3Refs) {
        let self = this; // For callbacks that change "this". Alternative to using .bind().

        this.dom.clips.solverTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.solverTree.height);

        let nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", function (d) {
                let solver_class = self.style.getSolverClass(self.showLinearSolverNames, {
                    'linear': d.linear_solver,
                    'nonLinear': d.nonlinear_solver
                });
                return solver_class + " solver_group " + self.style.getNodeClass(d);
            })
            .attr("transform", function (d) {
                let x = 1.0 - d.draw.solverDims.x - d.draw.solverDims.width;
                // The magic for reversing the blocks on the right side

                return "translate(" + self.scales.solver.x(x) + "," +
                    self.scales.solver.y(d.draw.solverDims.y) + ")";
            });

        nodeUpdate.select("rect")
            .attr("width", function (d) {
                return d.draw.solverDims.width * self.transitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.draw.solverDims.height * self.transitCoords.solver.y;
            })
            .attr('rx', 12)
            .attr('ry', 12);

        nodeUpdate.select("text")
            .attr("transform", function (d) {
                let anchorX = d.draw.solverDims.width * self.transitCoords.solver.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.draw.solverDims.height *
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
                return "translate(" + self.scales.solver.x(d.draw.solverDims.x) + "," +
                    self.scales.solver.y(d.draw.solverDims.y) + ")";
            })
            .remove();

        nodeExit.select("rect")
            .attr("width", function (d) {
                return d.draw.solverDims.width * self.transitCoords.solver.x;
            })
            .attr("height", function (d) {
                return d.draw.solverDims.height * self.transitCoords.solver.y;
            });

        nodeExit.select("text")
            .attr("transform", function (d) {
                let anchorX = d.draw.solverDims.width * self.transitCoords.solver.x -
                    self.layout.size.rightTextMargin;
                return "translate(" + anchorX + "," + d.draw.solverDims.height *
                    self.transitCoords.solver.y / 2 + ")";
            })
            .style("opacity", 0);
    }

    /** Remove all rects in the highlight bar */
    clearHighlights() {
        const selection = this.dom.highlightBar.selectAll('rect');
        const size = selection.size();
        debugInfo(`clearHighlights: Removing ${size} highlights`);
        selection.remove();
    }

    /** Remove all pinned arrows */
    clearArrows() {
        this.arrowMgr.removeAllPinned();
        this.clearHighlights();
    }

    /** Display connection arrows for all visible inputs/outputs */
    showAllArrows() {
        for (const row in this.matrix.grid) {
            const cell = this.matrix.grid[row][row]; // Diagonal cells only
            this.matrix.drawOnDiagonalArrows(cell);
            this.arrowMgr.togglePin(cell.id, true);
        }
    }

    showDesignVars() {
        [Object.keys(modelData.design_vars), Object.keys(modelData.responses)].flat().forEach(
            item => d3.select("#" + item.replaceAll(".", "_")).classed('opt-vars', true)
            );
        d3.select('.partition_group #_auto_ivc').classed('opt-vars', true)
    }

    hideDesignVars() {
        [Object.keys(modelData.design_vars), Object.keys(modelData.responses)].flat().forEach(
            item => d3.select("#" + item.replaceAll(".", "_")).classed('opt-vars', false)
            );
        d3.select("#_auto_ivc").classed('opt-vars', false)
    }

    delay(time) {
        return new Promise(function(resolve) {
            setTimeout(resolve, time)
        });
     }

    /** Display an animation while the transition is in progress */
    showWaiter() {
        this.dom.waiter.attr('class', 'show');
    }

    /** Hide the animation after the transition completes */
    hideWaiter() {
        this.dom.waiter.attr('class', 'no-show');
    }

    /**
     * Refresh the diagram when something has visually changed.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    async update(computeNewTreeLayout = true) {
        this.showWaiter();
        await this.delay(100);

        this.ui.update();
        this.search.update(this.zoomedElement, this.model.root);

        // Compute the new tree layout if necessary.
        if (computeNewTreeLayout) {
            this.layout = new N2Layout(this.model, this.zoomedElement,
                this.showLinearSolverNames, this.showSolvers, this.dims);

            this.ui.updateClickedIndices();

            this.matrix = new N2Matrix(this.model, this.layout,
                this.dom.n2Groups, this.arrowMgr, this.ui.lastClickWasLeft,
                this.ui.findRootOfChangeFunction, this.matrix.nodeSize);
        }

        this._updateScale();
        this.layout.updateTransitionInfo(this.dom, this.transitionStartDelay, this.manuallyResized);

        let d3PartRefs = this._createPartitionCells();
        this._setupPartitionTransition(d3PartRefs);
        this._runPartitionTransition(d3PartRefs.selection);

        let d3SolverRefs = this._createSolverCells();
        this._setupSolverTransition(d3SolverRefs);
        this._runSolverTransition(d3SolverRefs.selection);

        this.arrowMgr.transition(this.matrix);
        this.matrix.draw();

        if (!d3.selection.prototype.transitionAllowed) this.hideWaiter();

        if (!this.ui.desVars) this.showDesignVars()
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
        // Don't resize if the height didn't actually change:
        if (this.dims.size.partitionTree.height == height) return;

        if (!this.manuallyResized) {
            height = this.layout.calcFitDims().height;
        }

        this.updateSizes(height, this.dims.size.font);

        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
        this.update();
    }

    /**
     * Adjust the font size of all text in the diagram based on user input.
     * @param {number} fontSize The new font size in pixels.
     */
    fontSizeSelectChange(fontSize) {
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
        if (this.matrix.cellExists(cell)) {
            this.matrix.mouseOverOnDiagonal(cell);
            this.ui.nodeInfoBox.update(d3.event, cell.obj, cell.color());
        }
    }

    /**
     * Move the node info panel around if it's visible
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseMoveOnDiagonal(cell) {
        if (this.matrix.cellExists(cell)) {
            this.ui.nodeInfoBox.moveNearMouse(d3.event);
        }
    }

    /**
     * Since the matrix can be destroyed and recreated, use this to invoke the callback
     * rather than setting one up that points directly to a specific matrix.
     */
    mouseOverOffDiagonal(cell) {
        if (this.matrix.cellExists(cell)) {
            this.matrix.mouseOverOffDiagonal(cell);
        }
    }

    /** When the mouse leaves a cell, remove all temporary arrows and highlights. */
    mouseOut() {
        this.arrowMgr.removeAllHovered();
        this.clearHighlights();
        d3.selectAll("div.offgrid").style("visibility", "hidden").html('');

        this.ui.nodeInfoBox.clear();
    }

    /**
     * When the mouse is left-clicked on a cell, change their CSS class
     * so they're not removed when the mouse moves out. Or, if in info panel
     * mode, pin the info panel.
     * @param {N2MatrixCell} cell The cell the event occured on.
     */
    mouseClick(cell) {
        if (this.ui.click.isNormal) { // If not in info-panel mode, pin/unpin arrows
            this.arrowMgr.togglePin(cell.id);
        }
        else { // Make a persistent info panel
            this.ui.nodeInfoBox.pin();
        }
    }

    /**
     * Place member mouse callbacks in an object for easy reference.
     * @returns {Object} Object containing each of the functions.
     */
    getMouseFuncs() {
        const self = this;

        const mf = {
            'overOffDiag': self.mouseOverOffDiagonal.bind(self),
            'overOnDiag': self.mouseOverOnDiagonal.bind(self),
            'moveOnDiag': self.mouseMoveOnDiagonal.bind(self),
            'out': self.mouseOut.bind(self),
            'click': self.mouseClick.bind(self)
        }

        return mf;
    }

    /**
     * Recurse through the model, and determine whether a group/component is
     * minimized or manually expanded, or an input/output hidden. If it is,
     * add it to the hiddenList array, and optionally reset its state.
     * @param {Object[]} hiddenList The provided array to populate.
     * @param {Boolean} reveal If true, make the node visible.
     * @param {OmTreeNode} node The current node to operate on.
     */
    findAllHidden(hiddenList, reveal = false, node = this.model.root) {
        // Filtered nodes are handled by their true parents
        if (node.isFilter()) return;

        if (!node.isVisible() || node.draw.minimized || node.draw.manuallyExpanded) {
            hiddenList.push({
                'node': node,
                'draw': {
                    'minimized': node.draw.minimized,
                    'hidden': node.draw.hidden,
                    'filtered': node.draw.filtered,
                    'manuallyExpanded': node.draw.manuallyExpanded
                }
            })

            if (reveal) {
                node.expand();
                node.show();
                node.draw.manuallyExpanded = false;
                node.removeSelfFromFilter();
            }
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.findAllHidden(hiddenList, reveal, child);
            }
        }
    }

    /**
     * Restore the minimized/hidden value to all the specified nodes.
     * @param {Object[]} hiddenList The list of preserved objects
     * @param {OmTreeNode} node The current node to operate on.
    */
    resetAllHidden(hiddenList, node = this.model.root) {
        // Filtered nodes are handled by their true parents
        if (!hiddenList || node.isFilter()) return;

        const foundEntry = hiddenList.find(item => item.node === node);

        // If variables were selectively hidden, force the variable selection
        // dialog to rebuild the hiddenVars array.
        if (node.hasFilters()) { node.filter.inputs.wipe(); node.filter.outputs.wipe(); }

        if (!foundEntry) { // Not found, reset values to default
            node.expand();
            node.show();
            node.draw.manuallyExpanded = false;
            node.removeSelfFromFilter();
        }
        else { // Found, restore values
            node.draw.minimized = foundEntry.draw.minimized;
            node.draw.hidden = foundEntry.draw.hidden;
            node.draw.manuallyExpanded = foundEntry.draw.manuallyExpanded;
            if (foundEntry.draw.filtered) { node.addSelfToFilter(); }
            else { node.removeSelfFromFilter(); }
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.resetAllHidden(hiddenList, child);
            }
        }
    }

    /** Unset all manually-selected node states and zoom to the root element */
    reset() {
        this.resetAllHidden([]);
        this.updateZoomedElement(this.model.root);
        N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
        this.update();
    }

    /**
     * Set the node as not minimized and manually expanded, as well as
     * all children.
     * @param {OmTreeNode} startNode The node to begin from.
     */
    manuallyExpandAll(startNode) {
        startNode.draw.minimized = false;
        startNode.draw.manuallyExpanded = true;

        if (startNode.hasChildren()) {
            for (const child of startNode.children) {
                this.manuallyExpandAll(child);
            }
        }
    }

    /**
     * Set all the children of the specified node as minimized and not manually expanded.
     * @param {OmTreeNode} startNode The node to begin from.
     * @param {Boolean} [initialNode = true] Indicate the starting node.
     */
    minimizeAll(startNode, initialNode = true) {
        if (!initialNode) {
            startNode.draw.minimized = true;
            startNode.draw.manuallyExpanded = false;
        }

        if (startNode.hasChildren()) {
            for (const child of startNode.children) {
                this.minimizeAll(child, false);
            }
        }
    }

    /**
     * Recursively minimize non-input nodes to the specified depth.
     * @param {OmTreeNode} node The node to work on.
     */
    _minimizeToDepth(node) {
        if (node.isInputOrOutput()) {
            return;
        }

        if (node.depth < this.chosenCollapseDepth) {
            node.draw.minimized = false;
            node.draw.manuallyExpanded = true;
        }
        else {
            node.draw.minimized = true;
            node.draw.manuallyExpanded = false;
        }

        if (node.hasChildren()) {
            for (let child of node.children) {
                this._minimizeToDepth(child);
            }
        }
    }

    /**
     * Set the new depth to collapse to and perform the operation.
     * @param {Number} depth If the node's depth is the same or more, collapse it.
     */
    minimizeToDepth(depth) {
        this.chosenCollapseDepth = depth;

        if (this.chosenCollapseDepth > this.zoomedElement.depth)
            this._minimizeToDepth(this.model.root);
    }

    /**
     * Using an object populated by loading and validating a JSON file, set the model
     * to the saved view.
     * @param {Object} oldState The model view to restore.
     */
    restoreSavedState(oldState) {
        // Solver toggle state.
        this.showLinearSolverNames = oldState.showLinearSolverNames;
        this.ui.setSolvers(oldState.showLinearSolverNames);
        this.showSolvers = oldState.showSolvers;

        // Zoomed node (subsystem).
        this.zoomedElement = this.findNodeById(oldState.zoomedElement);

        // Expand/Collapse state of all nodes (subsystems) in model.
        this.setSubState(oldState.expandCollapse.reverse());

        // Force an immediate display update.
        // Needed to do this so that the arrows don't slip in before the element zoom.
        this.layout = new N2Layout(this.model, this.zoomedElement,
            this.showLinearSolverNames, this.showSolvers, this.dims);
        this.ui.updateClickedIndices();
        this.matrix = new N2Matrix(this.model, this.layout,
            this.dom.n2Groups, this.arrowMgr, this.ui.lastClickWasLeft,
            this.ui.findRootOfChangeFunction, this.matrix.nodeSize);
        this._updateScale();
        this.layout.updateTransitionInfo(this.dom, this.transitionStartDelay, this.manuallyResized);

        // Arrow State
        this.arrowMgr.loadPinnedArrows(oldState.arrowState);
    }
}
