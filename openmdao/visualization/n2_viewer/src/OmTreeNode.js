// <<hpp_insert gen/TreeNode.js>>

/**
 * Extend NodeDisplayData by handling solvers.
 * @typedef NodeDisplayData
 */
class OmNodeDisplayData extends NodeDisplayData {
    constructor() {
        super();
        this.nameSolverWidthPx = 1; // Solver-side label width pixels as computed by N2Layout
        this.solverDims = { x: 1e-6, y: 1e-6, width: 1, height: 1 };
        this.prevSolverDims = { x: 1e-6, y: 1e-6, width: 1e-6, height: 1e-6 };
    }

    /**
     * Copy the current model or solver dims to the previous ones.
     * @param {Boolean} solver True if the solver dims should be copied.
     */
    preserveDims(solver) {
        if (solver) {
            for (const prop in this.solverDims) {
                this.prevSolverDims[prop] = this.solverDims[prop];
            }
        }
        else {
            super.preserveDims();
        }
    }
}

/**
 * Extend TreeNode by adding support for feedback cycle arrows and solvers.
 * @typedef OmTreeNode
 */
class OmTreeNode extends TreeNode {
    constructor(origNode, attribNames) {
        super(origNode, attribNames);

        // Solver names may be empty, so set them to "None" instead.
        if (this.linear_solver == "") this.linear_solver = "None";
        if (this.nonlinear_solver == "") this.nonlinear_solver = "None";
    }

    get absPathName() { return this.path; }
    set absPathName(newName) { this.path = newName; return newName; }

    _newDisplayData() { return new OmNodeDisplayData(); }

    addFilterChild(attribNames) {
        if (this.isComponent()) { super.addFilterChild(attribNames); }
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
            for (const child of this.children) {
                if (child instanceof OmTreeNode)
                    child._getNodesInChildrenWithCycleArrows(arr);
            }
        }
    }

    /**
     * Populate an array with nodes in our lineage that contain a cycleArrows member.
     * @returns {Array} The array containing all the found nodes with cycleArrows.
     */
    getNodesWithCycleArrows() {
        const arr = [];

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
     * Create a backup of our position and other info.
     * @param {boolean} solver Whether to use .dims or .solverDims.
     * @param {number} leafNum Identify this as the nth leaf of the tree
     */
     preserveDims(solver, leafNum) {
        this.draw.preserveDims(solver);

        if (this.rootIndex < 0) this.rootIndex = leafNum;
        this.prevRootIndex = this.rootIndex;
    }
}
