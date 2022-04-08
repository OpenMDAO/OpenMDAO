// <<hpp_insert gen/Toolbar.js>>

/**
 * Manage the set of buttons and tools at the left of the diagram.
 * @typedef OmToolbar
 * @property {Boolean} hidden Whether the toolbar is visible or not.
 */
class OmToolbar extends Toolbar {
    /**
     * Set up the event handlers for mouse hovering and clicking.
     * @param {OmUserInterface} ui Reference to the main interface object
     */
    constructor(ui) {
        super(ui);
    }

    /**
     * Associate all of the buttons on the toolbar with a method in OmUserInterface.
     * @param {OmUserInterface} ui A reference to the UI object
     */
    _setupButtonFunctions(ui) {
        super._setupButtonFunctions(ui);
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const tooltipBox = d3.select(".tool-tip");

        this._addButton(new ToolbarButtonClick('#linear-solver-button', tooltipBox,
            "Control solver tree display",
            () => { ui.setSolvers(true); ui.showSolvers(); }));

        this._addButton(new ToolbarButtonClick('#linear-solver-button-2', tooltipBox,
            "Show linear solvers",
            (e, target) => {
                ui.setSolvers(true);
                ui.showSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#non-linear-solver-button', tooltipBox,
            "Show non-linear solvers",
            (e, target) => {
                ui.setSolvers(false);
                ui.showSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#no-solver-button', tooltipBox,
            "Hide solvers",
            (e, target) => {
                ui.hideSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonToggle('#desvars-button', tooltipBox,
            ["Show optimization variables", "Hide optimization variables"],
            () => ui.desVars, () => ui.toggleDesVars()))
            .setHelpInfo("Toggle optimization variables");
    }
}
