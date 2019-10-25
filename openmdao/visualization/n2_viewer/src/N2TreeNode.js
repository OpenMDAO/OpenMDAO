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

        // Solver names may be empty, so set them to "None" instead.
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";

        this.rootIndex = -1;
        this.dims = { 'x': 0, 'y': 0, 'width': 1, 'height': 1 };
        this.prevDims = { 'x': 1e-6, 'y': 1e-6, 'width': 1e-6, 'height': 1e-6 };
        this.solverDims = { 'x': 0, 'y': 0, 'width': 1, 'height': 1 };
        this.prevSolverDims = { 'x': 1e-6, 'y': 1e-6, 'width': 1e-6, 'height': 1e-6 };
        this.isMinimized = false;
    }

    toggleMinimize() {
        this.isMinimized = !this.isMinimized;
        return this.isMinimized;
    }

    /**
     * Create a backup of our position and other info.
     * @param {boolean} solver Whether to use .dims or .solverDims.
     * @param {number} leafNum Identify this as the nth leaf of the tree
     */
    preserveDims(solver, leafNum) {
        let dimProp = solver ? 'dims' : 'solverDims';
        let prevDimProp = solver ? 'prevDims' : 'prevSolverDims';

        Object.assign(this[prevDimProp], this[dimProp]);

        if (this.rootIndex < 0) this.rootIndex = leafNum;
        this.prevRootIndex = this.rootIndex;
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

    /** True if this.type is 'param' or 'unconnected_param'. */
    isParam() {
        return this.type.match(paramRegex);
    }

    isConnectedParam() {
        return (this.type == 'param');
    }

    isUnconnectedParam() {
        return (this.type == 'unconnected_param');
    }

    /** True if this.type is 'unknown'. */
    isUnknown() {
        return (this.type == 'unknown');
    }

    /** True if this.type is 'param', 'unconnected_param', or 'unknown'. */
    isParamOrUnknown() {
        return this.type.match(paramOrUnknownRegex);
    }

    /** True is this.type is 'subsystem' */
    isSubsystem() {
        return (this.type == 'subsystem');
    }
}