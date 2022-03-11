// <<hpp_insert src/OmTreeNode.js>>
// <<hpp_insert gen/NodeConnection.js>>

/**
 * Extend functionality of NodeConnection by handling feedback cycle arrows.
 * @typedef OmNodeConnection
 */
class OmNodeConnection extends NodeConnection {
    /**
     * Process the connection and warn if any problems are found.
     * @param {Object} conn The original connection object with src and tgt properties.
     * @param {Object} nodes References to all TreeNodes in the model. Keys are path
     *  names, the values are TreeNode objects.
     * @param {String[]} sysPathnames Array of pathnames that the indices in the cycle
     *  arrow arrays correspond to.
     */
    constructor(conn, nodes, sysPathnames = null) {
        super(conn, nodes);
        if (sysPathnames) this._processCycleArrows(sysPathnames, nodes);
    }

    /**
     * The cycle_arrows object in each connection is an array of length-2 arrays,
     * each of which is an index into the sysPathnames array. Using that array we
     * can resolve the indexes to pathnames to the associated objects.
     * @param {String[]} sysPathnames Array of pathnames that the indices in the cycle
     *  arrow arrays correspond to.
     * @param {Object} nodes References to all TreeNodes in the model. Keys are path
     *  names, the values are TreeNode objects.
     */
    _processCycleArrows(sysPathnames, nodes) {
        if (Array.isPopulatedArray(this.conn.cycle_arrows)) {
            const cycleArrowsArray = [];

            // Find the named TreeNode objects at each end and create a new connection object
            for (const cycleArrow of this.conn.cycle_arrows) {
                if (cycleArrow.length != 2) {
                    console.warn(`cycleArrowsSplitArray length not 2, got ${cycleArrow.length}: ${cycleArrow}`);
                }
                else {
                    const srcPath = sysPathnames[cycleArrow[0]];
                    const tgtPath = sysPathnames[cycleArrow[1]];

                    if (!srcPath in nodes) {
                        console.warn(`Cannot find cycle arrow source object ${srcPath}`);
                    }
                    else {
                        if (!tgtPath in nodes) {
                            console.warn(`Cannot find cycle arrow target object ${tgtPath}`);
                        }
                        else {
                            const arrow = {'begin': nodes[srcPath], 'end': nodes[tgtPath]};
                            cycleArrowsArray.push(arrow);
                        }
                    }
                }
            }

            if (!this.tgtObj.parent.cycleArrows) { this.tgtObj.parent.cycleArrows = []; }

            this.tgtObj.parent.cycleArrows.push({
                'src': this.srcObj,
                'arrows': cycleArrowsArray
            });
        }
    }
}
