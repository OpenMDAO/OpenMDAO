/**
 * Essentially the same as a JSON object from the model tree,
 * with some utility functions.
 * @typedef {N2TreeNode}
 * @property {Object} dims The size and location of the node when drawn.
 * @property {Object} prevDims The previous value of dims.
 * @property {Object} solverDims The size and location of the node within the solver tree.
 * @property {Object} prevSolverDims The previous value of solverDims.
 * @property {Object} isMinimized Whether this node or a parent has been collapsed.
 * @property {number} depth
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

    /** Run when a node is collapsed/restored. */
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

    /** True if this is a parameter and connected. */
    isConnectedParam() {
        return (this.type == 'param');
    }

    /** True if this a paramater and unconnected. */
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

    /**
     * Compare the supplied node, and recurse through children if it doesn't match.
     * @returns {Boolean} True if a match is found.
     */
    _hasNodeInChildren(compareNode) {
        if (this === compareNode) {
            return true;
        }

        if (this.hasChildren()) {
            for (let child of node.children) {
                if (child._hasNodeInChildren(compareNode)) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Look for the supplied node in the lineage of this one. First check
     * parents, then children.
     * @param {N2TreeNode} compareNode The node to look for.
     * @returns {Boolean} True if the node is found, otherwise false.
    */
    hasNode(compareNode) {
        for (let obj = this; obj != null; obj = obj.parent) {
            if (obj === compareNode) {
                return true;
            }
        }
        return this._hasNodeInChildren(compareNode);
    }

    _getNodesInChildrenWithCycleArrows(arr) {
        if (this.cycleArrows) { arr.push(this); }

        if (this.hasChildren()) {
            for (let child of node.children) {
                child._getNodesInChildrenWithCycleArrows(arr);
            }
        }
    }

    getNodesWithCycleArrows() {
        let arr = [];

        //start with parent.. the children will get the current object to avoid duplicates
        for (let obj = this.parent; obj != null; obj = obj.parent) {
            if (obj.cycleArrows) { arr.push(obj); }
        }
        this._getNodesInChildrenWithCycleArrows(arr);

        return arr;
    }
}