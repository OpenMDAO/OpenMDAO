/**
 * Essentially the same as a JSON object from the model tree,
 * with some utility functions.
 * @typedef {N2TreeNode}
 * @property {Object} dims The size and location of the node when drawn.
 * @property {Object} prevDims The previous value of dims.
 * @property {Object} solverDims The size and location of the node within the solver tree.
 * @property {Object} prevSolverDims The previous value of solverDims.
 * @property {Boolean} isMinimized Whether this node or a parent has been collapsed.
 * @property {Number} nameWidthPx The width of the name in pixels as computed by N2Layout.
 * @property {Boolean} manuallyExpanded If this node was right-clicked.
 * @property {Number} depth The index of the column this node appears in.
 */
class N2TreeNode {

    /**
     * Absorb all the properties of the provided JSON object from the model tree.
     * @param {Object} origNode The node to work from.
     */
    constructor(origNode) {
        // Merge all of the props from the original JSON tree node to us.
        Object.assign(this, origNode);

        this.sourceParentSet = new Set();
        this.targetParentSet = new Set();
        this.nameWidthPx = 1; // Set by N2Layout
        this.numLeaves = 0; // Set by N2Layout
        this.isMinimized = false;
        this.manuallyExpanded = false;
        this.childNames = new Set(); // Set by ModelData
        this.depth = -1; // Set by ModelData
        this.parent = null; // Set by ModelData
        this.id = -1; // Set by ModelData
        this.absPathName = ''; // Set by ModelData
        this.numDescendants = 0; // Set by ModelData

        // Solver names may be empty, so set them to "None" instead.
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";

        this.rootIndex = -1;
        this.dims = {
            'x': 1e-6,
            'y': 1e-6,
            'width': 1,
            'height': 1
        };
        this.prevDims = {
            'x': 1e-6,
            'y': 1e-6,
            'width': 1e-6,
            'height': 1e-6
        };
        this.solverDims = {
            'x': 1e-6,
            'y': 1e-6,
            'width': 1,
            'height': 1
        };
        this.prevSolverDims = {
            'x': 1e-6,
            'y': 1e-6,
            'width': 1e-6,
            'height': 1e-6
        };

    }

    /** Run when a node is collapsed. */
    minimize() {
        this.isMinimized = true;
    }

    /** Run when a node is restored. */
    expand() {
        this.isMinimized = false;
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

    /** True if this.type is 'input' or 'unconnected_input'. */
    isInput() {
        return this.type.match(inputRegex);
    }

    /** True if this is an input and connected. */
    isConnectedInput() { return (this.type == 'input'); }

    /** True if this an input and unconnected. */
    isUnconnectedInput() { return (this.type == 'unconnected_input'); }

    /** True if this is an input whose source is an auto-ivc'd output */
    isAutoIvcInput() { return (this.type == 'autoivc_input');}

    /** True if this.type is 'output'. */
    isOutput() { return (this.type == 'output'); }

    /** True if this is the root node in the model */
    isRoot() { return (this.type == 'root'); }

    /** True if this is an output and it's not implicit */
    isExplicitOutput() { return (this.isOutput() && !this.implicit); }

    /** True if this is an output and it is implicit */
    isImplicitOutput() { return (this.isOutput() && this.implicit); }

    /** True if this.type is 'input', 'unconnected_input', or 'output'. */
    isInputOrOutput() { return this.type.match(inputOrOutputRegex); }

    /** True is this.type is 'subsystem' */
    isSubsystem() { return (this.type == 'subsystem'); }

    /** True if it's a subsystem and this.subsystem_type is 'group' */
    isGroup() { return ( this.isSubsystem() && this.subsystem_type == 'group'); }

    /** True if it's a subsystem and this.subsystem_type is 'component' */
    isComponent() { return ( this.isSubsystem() && this.subsystem_type == 'component'); }

    /** Not connectable if this is a input group or parents are minimized. */
    isConnectable() {
        if (this.isInputOrOutput() && !(this.hasChildren() ||
                this.parent.isMinimized || this.parentComponent.isMinimized)) return true;

        return this.isMinimized;
    }

    /** Return false if the node is minimized or hidden */
    isVisible() {
        return ! (this.varIsHidden || this.isMinimized);
    }

    /**
     * Look for the supplied node in the set of child names.
     * @returns {Boolean} True if a match is found, otherwise false.
     */
    hasNodeInChildren(compareNode) {
        return this.childNames.has(compareNode.absPathName);
    }

    /** Look for the supplied node in the parentage of this one.
     * @param {N2TreeNode} compareNode The node to look for.
     * @param {N2TreeNode} [parentLimit = null] Stop searching at this common parent.
     * @returns {Boolean} True if the node is found, otherwise false.
     */
    hasParent(compareNode, parentLimit = null) {
        for (let obj = this.parent; obj != null && obj !== parentLimit; obj = obj.parent) {
            if (obj === compareNode) {
                return true;
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
        if (this.type == 'root') return true;

        if ( this === compareNode) return true;

        // Check parents first.
        if (this.hasParent(compareNode, parentLimit)) return true;

        return this.hasNodeInChildren(compareNode);
    }

    /**
     * Add ourselves to the supplied array if we contain a cycleArrows property.
     * @param {Array} arr The array to add to.
     */
    _getNodesInChildrenWithCycleArrows(arr) {
        if (this.cycleArrows) {
            arr.push(this);
        }

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
            if (obj.cycleArrows) {
                arr.push(obj);
            }
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
        for (let myParent = this.parent; myParent != null; myParent = myParent.parent)
            for (let otherParent = other.parent; otherParent != null; otherParent = otherParent.parent)
                if (myParent === otherParent) return myParent;

        // Should never get here because root is parent of all
        debugInfo("No common parent found between two nodes: ", this, other);
        return null;
    }

    /**
     * If the node has a lot of descendants and it wasn't manually expanded,
     * minimize it.
     * @param {Number} depthCount The number of nodes at the next depth down.
     * @returns {Boolean} True if minimized here, false otherwise.
     */
    minimizeIfLarge(depthCount) {
        if ( ! (this.isRoot() || this.manuallyExpanded) &&
            ( this.depth >= (this.isComponent()?
                Precollapse.cmpDepthStart : Precollapse.grpDepthStart) &&
                this.numDescendants > Precollapse.threshold &&
                this.children.length > Precollapse.children - this.depth &&
                depthCount > Precollapse.depthLimit )) {
            debugInfo(`Precollapsing node ${this.absPathName}`)
            this.minimize();
            return true;
        }

        return false;
    }
}
