// <<hpp_insert gen/Toolbar.js>>

/**
 * Manage the set of buttons and tools at the left of the diagram.
 * @typedef OmToolbar
 * @property {Boolean} hidden Whether the toolbar is visible or not.
 */
class OmToolbar extends Toolbar {
    /**
     * Set up the event handlers for mouse hovering and clicking.
     * @param {N2UserInterface} n2ui Reference to the main interface object
     */
    constructor(n2ui) {
        super(n2ui);
    }

    /**
     * Associate all of the buttons on the toolbar with a method in N2UserInterface.
     * @param {N2UserInterface} n2ui A reference to the UI object
     */
    _setupButtonFunctions(n2ui) {
        super._setupButtonFunctions(n2ui);
        const self = this; // For callbacks that change "this". Alternative to using .bind().
        const tooltipBox = d3.select(".tool-tip");

        this._addButton(new ToolbarButtonClick('#linear-solver-button', tooltipBox,
            "Control solver tree display",
            () => { n2ui.setSolvers(true); n2ui.showSolvers(); }));

        this._addButton(new ToolbarButtonClick('#linear-solver-button-2', tooltipBox,
            "Show linear solvers",
            (e, target) => {
                n2ui.setSolvers(true);
                n2ui.showSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#non-linear-solver-button', tooltipBox,
            "Show non-linear solvers",
            (e, target) => {
                n2ui.setSolvers(false);
                n2ui.showSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonClick('#no-solver-button', tooltipBox,
            "Hide solvers",
            (e, target) => {
                n2ui.hideSolvers();
                self._setRootButton(target);
            }));

        this._addButton(new ToolbarButtonToggle('#desvars-button', tooltipBox,
            ["Show optimization variables", "Hide optimization variables"],
            () => n2ui.desVars, () => n2ui.toggleDesVars()))
            .setHelpInfo("Toggle optimization variables");
    }
}
