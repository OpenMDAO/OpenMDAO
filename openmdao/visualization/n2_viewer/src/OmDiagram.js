// <<hpp_insert gen/Diagram.js>>
// <<hpp_insert src/OmLayout.js>>
// <<hpp_insert src/OmStyle.js>>
// <<hpp_insert src/OmUserInterface.js>>
// <<hpp_insert src/OmMatrix.js>>

/**
 * Manage all components of the application. The model data, the CSS styles, the
 * user interface, the layout of the matrix, and the matrix grid itself are
 * all member objects. OmDiagram adds handling for solvers.
 * @typedef OmDiagram
 */
class OmDiagram extends Diagram {
    /**
     * Set initial values.
     * @param {Object} modelJSON The decompressed model structure.
     */
    constructor(modelJSON) {
        super(modelJSON, false);

        // Solver tree initial dimensions are the same as the model tree
        this.dims.size.solverTree = {...this.dims.size.partitionTree};

        this._init();
    }

    _newModelData() {
        this.model = new OmModelData(this.modelData);
    }

    /** Override Diagram._newLayout() to create an OmLayout object. */
    _newLayout() {

        if (this.showSolvers === undefined)
            this.showSolvers = true;

        return new OmLayout(this.model, this.zoomedElement, this.dims, this.showSolvers);
    }

    /** Create a new OmMatrix object. Overrides superclass method. */
    _newMatrix(lastClickWasLeft, prevCellSize = null) {
        return new OmMatrix(this.model, this.layout, this.dom.diagGroups,
            this.arrowMgr, lastClickWasLeft, this.ui.findRootOfChangeFunction, prevCellSize);
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
     _init() {
        this.style = new OmStyle(this.dom.svgStyle, this.dims.size.font);
        this.layout = this._newLayout();
        this.ui = new OmUserInterface(this);
        this.matrix = this._newMatrix(true);
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

    /**
     * Add SVG groups & contents coupled to the visible nodes in the trees.
     * Select all <g> elements that have class "solver_group". If any already
     * exist, join to their associated nodes in the solver tree. If no
     * existing <g> matches a displayable node, add it to the "enter"
     * selection so the <g> can be created. If a <g> exists but there is
     * no longer a displayable node for it, put it in the "exit" selection so
     * it can be removed:
     */
    _updateTreeCells() {
        super._updateTreeCells();

        const self = this;
        const scale = this.layout.scales.solver;
        const treeSize = this.layout.treeSize.solver;

        this.dom.pSolverTreeGroup.selectAll("g.solver_group")
            .data(this.layout.zoomedSolverNodes, d => d.id)
            .join(
                enter => self._addNewSolverCells(enter, scale, treeSize),
                update => self._updateExistingSolverCells(update, scale, treeSize),
                exit => self._removeOldSolverCells(exit, scale, treeSize)
            )
    }

    /**
     * Using the visible nodes in the solver tree as data points, create SVG objects to
     * represent each one. Dimensions are obtained from the precalculated layout.
     * @param {Selection} enter The selection to add <g> elements and children to.
     * @param {Scale} scale Linear scales of the diagram width and height.
     * @param {Dimensions} treeSize Actual width and height of the tree in pixels.
     */
    _addNewSolverCells(enter, scale, treeSize) {
        const self = this; // For callbacks that change "this".

        // Create a <g> for each node in zoomedSolverNodes that doesn't already have one.
        // Dimensions are obtained from the previous geometry so the new nodes can appear
        // to transition to the new size together with the existing nodes.
        const enterSelection = enter
            .append("g")
            .attr("class", d => {
                const solver_class = self.style.getSolverClass({ 'linear': d.linear_solver,
                    'nonLinear': d.nonlinear_solver });
                return `${solver_class} solver_group ${self.style.getNodeClass(d)}`;
            })
            .on("click", (e,d) => self.leftClickSelector(e, d))
            .on("contextmenu", (e,d) => self.ui.rightClick(e, d))
            .on("mouseover", (e,d) => {
                self.ui.showInfoBox(e, d);

                if (self.model.abs2prom != undefined) {
                    if (d.isInput()) {
                        return self.dom.toolTip.text(self.model.abs2prom.input[d.path])
                            .style("visibility", "visible");
                    }
                    if (d.isOutput()) {
                        return self.dom.toolTip.text(self.model.abs2prom.output[d.path])
                            .style("visibility", "visible");
                    }
                }
            })
            .on("mouseleave", () => {
                self.ui.removeInfoBox();

                if (self.model.abs2prom != undefined) {
                    self.dom.toolTip.style("visibility", "hidden");
                }
            })
            .on("mousemove", e => {
                self.ui.moveInfoBox(e);

                if (self.model.abs2prom != undefined) {
                    self.dom.toolTip.style("top", (e.pageY - 30) + "px")
                        .style("left", (e.pageX + 5) + "px");
                }
            });

        enterSelection
            .transition(sharedTransition)
            .attr("transform", d => {
                // The magic for reversing the blocks on the right side
                // The solver tree goes from the root on the right and expands to the left
                const x = 1.0 - d.draw.solverDims.x - d.draw.solverDims.width;

                return `translate(${scale.x(x)},${scale.y(d.draw.solverDims.y)})`;
            })

        enterSelection // Add the visible rectangle
            .append("rect")
            .transition(sharedTransition)
            .attr("width", d => d.draw.solverDims.width * treeSize.width)
            .attr("height", d => d.draw.solverDims.height * treeSize.height)
            .attr("id", d => d.path.replace(/\./g, '_'))
            .attr('rx', 12)
            .attr('ry', 12);

        enterSelection // Add a label
            .append("text")
            .text( d => d.getSolverText())
            .style('visibility', 'hidden')
            .attr("dy", ".35em")
            .attr("transform", d => {
                const anchorX = d.draw.solverDims.width * treeSize.width -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX},${d.draw.solverDims.height * treeSize.height / 2})`;
            })
            .style("opacity", d => (d.depth < self.zoomedElement.depth)? 0 : d.textOpacity)
            .transition(sharedTransition)
            .on('end', function() { d3.select(this).style('visibility', 'visible'); } )

        return enterSelection;
    }

    /**
     * Update the geometry for existing <g> with a transition.
     * @param {Selection} update The selected group of existing solver tree <g> elements.
     * @param {Scale} scale Linear scales of the diagram width and height.
     * @param {Dimensions} treeSize Actual width and height of the tree in pixels.
     */
     _updateExistingSolverCells(update, scale, treeSize) {
        const self = this; // For callbacks that change "this".

        this.dom.clips.solverTree
            .transition(sharedTransition)
            .attr('height', this.dims.size.solverTree.height);

        // New location for each group
        const mergedSelection = update
            .attr("class", d => {
                const solver_class = self.style.getSolverClass({ 'linear': d.linear_solver,
                    'nonLinear': d.nonlinear_solver });
                return `${solver_class} solver_group ${self.style.getNodeClass(d)}`;
            })
            .transition(sharedTransition)
            .attr("transform", d => {
                // The magic for reversing the blocks on the right side
                const x = 1.0 - d.draw.solverDims.x - d.draw.solverDims.width;

                return `translate(${scale.x(x)},${scale.y(d.draw.solverDims.y)})`;
            });

        // Resize each rectangle      
        mergedSelection
            .select("rect")
            .attr("width", d => d.draw.solverDims.width * treeSize.width)
            .attr("height", d => d.draw.solverDims.height * treeSize.height);

         // Move the text label
        mergedSelection
            .select("text")
            .text( d => d.getSolverText())
            .attr("transform", d => {
                const anchorX = d.draw.solverDims.width * treeSize.width -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX},${d.draw.solverDims.height * treeSize.height / 2})`;
            })
            .style("opacity", d => (d.depth < self.zoomedElement.depth)? 0 : d.textOpacity);

        return mergedSelection;
    }

    /**
     * Remove <g> that no longer have displayable nodes associated with them, and
     * transition them away.
     * @param {Selection} exit The selected group of solver tree <g> elements to remove.
     * @param {Scale} scale Linear scales of the diagram width and height.
     * @param {Dimensions} treeSize Actual width and height of the tree in pixels.
     */
    _removeOldSolverCells(exit, scale, treeSize) {
        const self = this; // For callbacks that change "this".

        // Transition exiting nodes to the parent's new position.
        const exitSelection = exit.transition(sharedTransition);

        if (this.showSolvers) {
            exitSelection.attr("transform", d =>
                `translate(${scale.x(d.draw.solverDims.x)},${scale.y(d.draw.solverDims.y)})`)
        }

        exitSelection.select("rect")
            .attr("width", d => d.draw.solverDims.width * treeSize.width)
            .attr("height", d => d.draw.solverDims.height * treeSize.height);

        exitSelection.select("text")
            .attr("transform", d => {
                const anchorX = d.draw.solverDims.width * treeSize.width -
                    self.layout.size.rightTextMargin;
                return `translate(${anchorX},${d.draw.solverDims.height * treeSize.height / 2})`;
            })
            .style("opacity", 0);

        exitSelection.on('end', function() {d3.select(this).remove(); })

        return exitSelection;
    }

    /** Add the opt-vars class to design variable elements to set the fill color */
    showDesignVars() {
        [Object.keys(modelData.design_vars), Object.keys(modelData.responses)].flat().forEach(
            path => d3.select(`#${TreeNode.pathToId(path)}`).classed('opt-vars', true)
            );
        d3.select('.model_tree_grp #_auto_ivc').classed('opt-vars', true)
    }

    /** Remove the opt-vars class from design variable elements to use the default fill color */
    hideDesignVars() {
        [Object.keys(modelData.design_vars), Object.keys(modelData.responses)].flat().forEach(
            path => d3.select(`#${TreeNode.pathToId(path)}`).classed('opt-vars', false)
            );
        d3.select("#_auto_ivc").classed('opt-vars', false)
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
        OmTreeNode.showLinearSolverNames = oldState.showLinearSolverNames;
        this.ui.setSolvers(oldState.showLinearSolverNames);
        this.showSolvers = oldState.showSolvers;
        
        super.restoreSavedState(oldState);       
     }
}
