// <<hpp_insert gen/TreeNode.js>>

/**
 * Processes and manages a connection between nodes.
 * @typedef NodeConnection
 * @property {Object} conn The original connection object with src and tgt properties.
 * @property {TreeNode} srcObj The start of the connection.
 * @property {TreeNode} tgtObj The end of the connection.
 */
class NodeConnection {
    /**
     * Process the connection and warn if any problems are found.
     * @param {Object} conn The original connection object with src and tgt properties.
     * @param {Object} nodes References to all TreeNodes in the model. Keys are path
     *  names, the values are TreeNode objects.
     */
    constructor(conn, nodes) {
        this.conn = conn;

        this.srcObj = nodes[conn.src];
        this.srcObj.connTargets.add(conn.tgt);
        const srcObjParents = [];
        if (!this.srcObj) {
            console.warn(`Cannot find connection source ${conn.src}.`);
        }
        else {
            // Collect all parents of the source node.
            srcObjParents.push(this.srcObj);
            for (let parentObj = this.srcObj.parent; parentObj != null; parentObj = parentObj.parent) {
                srcObjParents.push(parentObj);
            }
        }

        this.tgtObj = nodes[conn.tgt];
        this.tgtObj.connSources.add(conn.src);
        const tgtObjParents = [];
        if (!this.tgtObj) {
            console.warn(`Cannot find connection target ${conn.tgt}.`);
        }
        else {
            // Collect all parents of the target node.
            tgtObjParents.push(this.tgtObj);
            for (let parentObj = this.tgtObj.parent; parentObj != null; parentObj = parentObj.parent) {
                tgtObjParents.push(parentObj);
            }
        }
    
        // Make sure all parents of the source and target are aware of the
        // connection. Required for handling collapsed or offscreen node connections.
        for (const srcParent of srcObjParents) {
            for (const tgtParent of tgtObjParents) {
                if (tgtParent.path != '')
                    srcParent.targetParentSet.add(tgtParent);

                if (srcParent.path != '')
                    tgtParent.sourceParentSet.add(srcParent);
            }
        }
    }
}
