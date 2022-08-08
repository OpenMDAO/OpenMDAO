// <<hpp_insert gen/UserInterface.js>>
// <<hpp_insert src/OmToolbar.js>>
// <<hpp_insert src/OmNodeInfo.js>>
// <<hpp_insert src/OmLegend.js>>

/**
 * Handle input events for the matrix and toolbar.
 * @typedef OmUserInterface
 */

class OmUserInterface extends UserInterface {
    /**
     * Initialize properties, set up the collapse-depth menu, and set up other
     * elements of the toolbar.
     * @param {OmDiagram} diag A reference to the main diagram.
     */
    constructor(diag) {
        super(diag);
    }

    /**
     * Separate these calls from the constructor so that subclasses can
     * set values before execution.
     */
    _init() {
        this.desVars = true;
        this.legend = new OmLegend(this.diag.modelData);
        this.toolbar = new OmToolbar(this);
    }

    /**
     * Create a regular node info window if the hovered object is not a solver,
     * or a solver node info window if it is.
     * @param {Selection} svgNodeGroup Selected object that triggered the event.
     */
    _newInfoBox(svgNodeGroup) {
        this.nodeInfoBox = svgNodeGroup.classed('solver_group')? 
            new OmSolverNodeInfo(this) : new OmNodeInfo(this);        
    }

    /** Determine the style of the inactive handle (overriding superclass) */
    _inactiveResizerHandlerStyle() {
        return this.diag.showSolvers?
            'inactive-resizer-handle' : 'inactive-resizer-handle-without-solvers'
    }

    /**
     * Wipe the current solvers legend area and populate with the other type.
     * @param {Boolean} linear True to use linear solvers, false for non-linear.
     */
    setSolvers(linear) {
        // Update the diagram
        OmTreeNode.showLinearSolverNames = linear;

        // update the legend
        this.legend.updateSolvers();

        if (this.legend.shown)
            this.legend.show();
        this.diag.update();
    }

    /** Display the solver tree. */
    showSolvers() {
        this.diag.showSolvers = true;
        this.diag.update();
        d3.select('#n2-resizer-handle').attr('class', 'inactive-resizer-handle')
    }

    /** Hide the solver tree. */
    hideSolvers() {
        this.diag.showSolvers = false;
        this.diag.update();
        d3.select('#n2-resizer-handle').attr('class', 'inactive-resizer-handle-without-solvers')
    }

    /** Show or hide design variables. */
    toggleDesVars() {
        if (this.desVars) {
            this.diag.showDesignVars();
            this.desVars = false;
        } else {
            this.diag.hideDesignVars();
            this.desVars = true;
        }

        d3.select('#desvars-button').attr('class',
            this.desVars ? 'fas icon-fx-2' : 'fas icon-fx-2 active-tab-icon');
    }

    /**
     * Minimize the specified node and recursively minimize its children.
     * @param {OmTreeNode} node The current node to operate on.
     */
     _collapseOutputs(node) {
        if (node.subsystem_type && node.subsystem_type == 'component') {
            node.minimize();
        }
        if (node.hasChildren()) {
            for (const child of node.children) {
                this._collapseOutputs(child);
            }
        }
    }

    /** Save the model state to a file. Adds solver support to the base class. */
    saveState() {
        // Solver toggle state.
        const extraData = {
            'showLinearSolverNames': OmTreeNode.showLinearSolverNames,
            'showSolvers': this.diag.showSolvers,            
        }

        super.saveState(extraData);
    }
}
