
// <<hpp_insert src/OmTreeNode.js>>
// <<hpp_insert gen/NodeConnection.js>>

class OmNodeConnection extends NodeConnection {
    constructor(conn, nodes, sysPathnames) {
        super(conn, nodes);
        this._processCycleArrows(sysPathnames, nodes);
    }

    /**
     * The cycle_arrows object in each connection is an array of length-2 arrays,
     * each of which is an index into the sysPathnames array. Using that array we
     * can resolve the indexes to pathnames to the associated objects.
     * @param {Object} conn Reference to the connection to operate on.
     */
    _processCycleArrows(sysPathnames, nodes) {
        if (Array.isPopulatedArray(this.conn.cycle_arrows)) {
            const cycleArrowsArray = [];
            for (const cycleArrow of this.conn.cycle_arrows) {
                if (cycleArrow.length != 2) {
                    console.warn(`cycleArrowsSplitArray length not 2, got ${cycleArrow.length}: ${cycleArrow}`);
                    continue;
                }

                const srcPathname = sysPathnames[cycleArrow[0]];
                const tgtPathname = sysPathnames[cycleArrow[1]];

                const arrowBeginObj = nodes[srcPathname];
                if (!arrowBeginObj) {
                    console.warn(`Cannot find cycle arrows begin object ${srcPathname}`);
                    continue;
                }

                const arrowEndObj = nodes[tgtPathname];
                if (!arrowEndObj) {
                    console.warn(`Cannot find cycle arrows end object ${tgtPathname}`);
                    continue;
                }

                cycleArrowsArray.push({
                    "begin": arrowBeginObj,
                    "end": arrowEndObj
                });
            }

            if (!this.tgtObj.parent.hasOwnProperty("cycleArrows")) {
                this.tgtObj.parent.cycleArrows = [];
            }
            this.tgtObj.parent.cycleArrows.push({
                "src": this.srcObj,
                "arrows": cycleArrowsArray
            });
        }
    }
}
