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
        this.sourceParentSet = new Set();
        this.targetParentSet = new Set();

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
            for (let child of this.children) {
                if (child._hasNodeInChildren(compareNode)) {
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Look for the supplied node in the lineage of this one.
     * @param {N2TreeNode} compareNode The node to look for.
     * @param {N2TreeNode} [parentLimit = null] Stop searching at this common parent.
     * @returns {Boolean} True if the node is found, otherwise false.
    */
    hasNode(compareNode, parentLimit = null) {
        // Check parents first.
        for (let obj = this; obj != null && obj !== parentLimit; obj = obj.parent) {
            if (obj === compareNode) {
                return true;
            }
        }

        // Check children if not found in parents.
        return this._hasNodeInChildren(compareNode);
    }

    /**
     * Add ourselves to the supplied array if we contain a cycleArrows property.
     * @param {Array} arr The array to add to.
     */
    _getNodesInChildrenWithCycleArrows(arr) {
        if (this.cycleArrows) { arr.push(this); }

        if (this.hasChildren()) {
            for (let child of this.children) {
                child._getNodesInChildrenWithCycleArrows(arr);
            }
        }
    }

    /**
     * Populate an array with nodes in our lineage that contain a cycleArrows member.
     * @returns {Array} The array containing all the found nodes with cycleArrows.
     */
    getNodesWithCycleArrows() {
        let arr = [];

        // Check parents first.
        for (let obj = this.parent; obj != null; obj = obj.parent) {
            if (obj.cycleArrows) { arr.push(obj); }
        }

        // Check all descendants as well.
        this._getNodesInChildrenWithCycleArrows(arr);

        return arr;
    }

    /**
     * Find the closest parent shared by two nodes; farthest should be tree root.
     * @param {N2TreeNode} other Another node to compare parents with.
     * @returns {N2TreeNode} The first common parent found.
     */
    nearestCommonParent(other) {
        for (let myParent = this.parent; myParent != null; myParent = myParent.parent )
            for (let otherParent = other.parent; otherParent != null; otherParent = otherParent.parent)
                if (myParent === otherParent) return myParent;

        // Should never get here because root is parent of all
        debugInfo("No common parent found between two nodes: ", this, other);
        return null;
    }
}