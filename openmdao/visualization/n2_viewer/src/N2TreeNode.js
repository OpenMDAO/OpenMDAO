/**
 * Essentially the same as a JSON object from the model tree,
 * with some utility functions.
 * @typedef {N2TreeNode}
*/
class N2TreeNode {

    /**
     * Absorb all the properties of the provided JSON object from the model tree.
     * @param {Object} origNode The node to work from.
     */
    constructor(origNode) {
        // Merge all of the props from the original JSON tree node to us.
        Object.assign(this, origNode);

        // From old ClearConnections():
        this.targetsParamView = new Set();
        this.targetsHideParams = new Set();

        this.changeBlankSolverNamesToNone();
    }

    /** 
     * Determine if the children array exists and has members.
     * @param {string} [childrenPropName = 'children'] Usually children, but
     *   sometimes 'subsystem_children'
     * @return {boolean} True if the children property is an Array and length > 0.
    */
    hasChildren(childrenPropName = 'children') {
        return (Array.isPopulatedArray(this[childrenPropName]));
    }

    /**
     * Solver names may be empty, so set them to "None" instead.
     */
    changeBlankSolverNamesToNone() {
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";
    }

    /** True if this.type is 'param' or 'unconnected_param'. */
    isParam() {
        return this.type.match(paramRegex);
    }

    /** True if this.type is 'param', 'unconnected_param', or 'unknown'. */
    isParamOrUnknown() {
        return this.type.match(paramOrUnknownRegex);
    }

    isSubsystem() {
        return (this.type == 'subsystem');
    }
}