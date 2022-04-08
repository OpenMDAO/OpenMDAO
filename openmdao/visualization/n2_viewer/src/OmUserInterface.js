// <<hpp_insert gen/UserInterface.js>>

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

    /** Determine the style of the inactive handle (overriding superclass) */
    _inactiveResizerHandlerStyle() {
        return self.diag.showSolvers?
            'inactive-resizer-handle' : 'inactive-resizer-handle-without-solvers'
    }

    /**
     * Wipe the current solvers legend area and populate with the other type.
     * @param {Boolean} linear True to use linear solvers, false for non-linear.
     */
    setSolvers(linear) {

        // Update the diagram
        this.diag.showLinearSolverNames = linear;

        // update the legend
        this.legend.toggleSolvers(this.diag.showLinearSolverNames);

        if (this.legend.shown)
            this.legend.show(
                this.diag.showLinearSolverNames,
                this.diag.style.solvers
            );
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
}
