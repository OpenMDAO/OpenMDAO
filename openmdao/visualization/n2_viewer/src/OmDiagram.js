// <<hpp_insert gen/Diagram.js>>
// <<hpp_insert src/OmLayout.js>>

/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the matrix, and the matrix grid itself are
 * all member objects. OmDiagram adds handling for solvers.
 * @typedef OmDiagram
 */
class OmDiagram extends Diagram {
    constructor(modelJSON) {
        super(modelJSON);
    }

    _newLayout() {
        if (this.showLinearSolverNames === undefined)
            this.showLinearSolverNames = true;

        if (this.showSolvers === undefined)
            this.showSolvers = true;

        return new OmLayout(this.model, this.zoomedElement, this.dims,
            this.showLinearSolverNames, this.showSolvers).init();
    }

    /**
     * Switch back and forth between showing the linear or non-linear solver names.
     */
    toggleSolverNameType() {
        this.showLinearSolverNames = !this.showLinearSolverNames;
    }

    /**
     * Setup internal references to D3 objects so we can avoid running
     * d3.select() over and over later.
     */
    _referenceD3Elements() {
        super._referenceD3Elements();
        this.dom.pSolverTreeGroup = d3.select('g#solver_tree');
        this.dom.clips.solverTree = d3.select("#solverTreeClip > rect");
    }

    _updateScale() {
        if (super._updateScale()) {
            const innerDims = this.layout.newInnerDims();

            const x = this.dims.size.partitionTree.width + innerDims.margin +
                innerDims.height + innerDims.margin;
            const y = innerDims.margin;

            this.dom.pSolverTreeGroup
                .attr("height", innerDims.height)
                .attr("transform", `translate(${x},${y})`);
        }
    }

    _createSolverCells() {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const scale = this.layout.scales.solver.prev;
        const transitCoords = this.layout.transitCoords.solver.prev;  

        const selection = this.dom.pSolverTreeGroup.selectAll(".solver_group")
            .data(this.layout.zoomedSolverNodes, d => d.id);

        const nodeEnter = selection.enter()
            .append("g")
            .attr("class", d => {
                const solver_class = self.style.getSolverClass(self.showLinearSolverNames,
                    { 'linear': d.linear_solver, 'nonLinear': d.nonlinear_solver });
                return `${solver_class} solver_group ${self.style.getNodeClass(d)}`;
            })
            .attr("transform", d => {

                const x = 1.0 - d.draw.prevSolverDims.x - d.draw.prevSolverDims.width;
                // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                return `translate(${scale.x(x)},${scale.y(d.draw.prevSolverDims.y)})`;
            })
            .on("click", d => self.leftClickSelector(this, d))
            .on("contextmenu", function (d) { self.ui.rightClick(d, this)})
            .on("mouseover", function(d) {
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
            .on("mouseleave", d => {
                self.ui.nodeInfoBox.clear();

                if (self.model.abs2prom != undefined) {
                    self.dom.toolTip.style("visibility", "hidden");
                }
            })
            .on("mousemove", () => {
                self.ui.nodeInfoBox.moveNearMouse(d3.event);

                if (self.model.abs2prom != undefined) {
                    self.dom.toolTip.style("top", (d3.event.pageY - 30) + "px")
                        .style("left", (d3.event.pageX + 5) + "px");
                }
            });

        nodeEnter.append("rect")
            .attr("width", d => d.draw.prevSolverDims.width * transitCoords.x)
            .attr("height", d => d.draw.prevSolverDims.height * transitCoords.y)
            .attr("id", d => d.absPathName.replace(/\./g, '_'));

        nodeEnter.append("text")
            .attr("dy", ".35em")
            .attr("transform", d => {
                const anchorX = d.draw.prevSolverDims.width * transitCoords.x -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX},${d.draw.prevSolverDims.height*transitCoords.y / 2})`;
            })
            .style("opacity", d => (d.depth < self.zoomedElement.depth)? 0 : d.textOpacity)
            .text(self.layout.getSolverText.bind(self.layout));

        return ({
            'selection': selection,
            'nodeEnter': nodeEnter
        });
    }

    _setupSolverTransition(d3Refs) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const scale = this.layout.scales.solver;
        const transitCoords = this.layout.transitCoords.solver;

        this.dom.clips.solverTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.solverTree.height);

        const nodeUpdate = d3Refs.nodeEnter.merge(d3Refs.selection)
            .transition(sharedTransition)
            .attr("class", d => {
                const solver_class = self.style.getSolverClass(self.showLinearSolverNames, {
                    'linear': d.linear_solver,
                    'nonLinear': d.nonlinear_solver
                });
                return solver_class + " solver_group " + self.style.getNodeClass(d);
            })
            .attr("transform", d => {
                const x = 1.0 - d.draw.solverDims.x - d.draw.solverDims.width;
                // The magic for reversing the blocks on the right side

                return `translate(${scale.x(x)},${scale.y(d.draw.solverDims.y)})`;
            });

        nodeUpdate.select("rect")
            .attr("width", d => d.draw.solverDims.width * transitCoords.x)
            .attr("height", d => d.draw.solverDims.height * transitCoords.y)
            .attr('rx', 12)
            .attr('ry', 12);

        nodeUpdate.select("text")
            .attr("transform", d => {
                const anchorX = d.draw.solverDims.width * transitCoords.x -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX},${d.draw.solverDims.height * transitCoords.y / 2})`;
            })
            .style("opacity", d => (d.depth < self.zoomedElement.depth)? 0 : d.textOpacity)
            .text(self.layout.getSolverText.bind(self.layout));
    }

    _runSolverTransition(selection) {
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const scale = this.layout.scales.solver; 
        const transitCoords = this.layout.transitCoords.solver;

        // Transition exiting nodes to the parent's new position.
        const nodeExit = selection.exit()
            .transition(sharedTransition)
            .attr("transform", d =>
                `translate(${scale.x(d.draw.solverDims.x)},${scale.y(d.draw.solverDims.y)})`)
            .remove();

        nodeExit.select("rect")
            .attr("width", d => d.draw.solverDims.width * transitCoords.x)
            .attr("height", d => d.draw.solverDims.height * transitCoords.y);

        nodeExit.select("text")
            .attr("transform", d => {
                const anchorX = d.draw.solverDims.width * transitCoords.x -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX},${d.draw.solverDims.height * transitCoords.y / 2})`;
            })
            .style("opacity", 0);
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

    /** Add HTML elements coupled to the visible nodes in the solver tree. */
    _newTreeCells() {
        super._newTreeCells();

        const d3SolverRefs = this._createSolverCells();
        this._setupSolverTransition(d3SolverRefs);
        this._runSolverTransition(d3SolverRefs.selection);
    }

    /**
     * Refresh the diagram when something has visually changed. Adds the ability to handle
     * design vars for OpenMDAO.
     * @param {Boolean} [computeNewTreeLayout = true] Whether to rebuild the layout and
     *  matrix objects.
     */
    async update(computeNewTreeLayout = true) {
        await super.update(computeNewTreeLayout);

        if (!this.ui.desVars) this.showDesignVars()
    }

    /**
     * Updates the intended dimensions of the diagrams and font, but does
     * not perform rendering itself.
     * @param {number} height The base height of the diagram without margins.
     * @param {number} fontSize The new size of the font.
     */
    updateSizes(height, fontSize) {
        super.updateSizes(height, fontSize);
        this.dims.size.solverTree.height = height;
    }

    /**
     * Using an object populated by loading and validating a JSON file, set the model
     * to the saved view. Adds solver-handling capability to the superclass function.
     * @param {Object} oldState The model view to restore.
     */
     restoreSavedState(oldState) {
        // Solver toggle state.
        this.showLinearSolverNames = oldState.showLinearSolverNames;
        this.ui.setSolvers(oldState.showLinearSolverNames);
        this.showSolvers = oldState.showSolvers;
        
        super.restoreSavedState(oldState);       
     }
}
