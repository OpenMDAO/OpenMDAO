
// <<hpp_insert gen/TreeNode.js>>

class NodeConnection {
    constructor(conn, nodes) {
        this.conn = conn;

        this.srcObj = nodes[conn.src];
        const srcObjParents = [];
        if (!this.srcObj) {
            console.warn(`Cannot find connection source ${conn.src}.`);
        }
        else {
            if (!this.srcObj.isOutput()) { // source obj must be output
                console.warn(`Source ${this.srcObj.path} is not an output.`);
            }
            else {
                if (this.srcObj.hasChildren()) {
                    console.warn(`Source ${this.srcObj.path} has children.`);
                }
                else {
                    srcObjParents.push(this.srcObj);
                    for (let obj = this.srcObj.parent; obj != null; obj = obj.parent) {
                        srcObjParents.push(obj);
                    }
                }
            }
        }

        this.tgtObj = nodes[conn.tgt];
        const tgtObjParents = [];
        if (!this.tgtObj) {
            console.warn(`Cannot find connection target ${conn.tgt}.`);
        }
        else {
            // Target obj must be an input
            if (!this.tgtObj.isInput()) {
                console.warn(`Target ${this.tgtObj.path} is not an input.`);
            }
            else {
                if (this.tgtObj.hasChildren()) {
                    console.warn(`Target ${this.tgtObj.path} has children.`);
                }
                else {
                    tgtObjParents.push(this.tgtObj);
                    for (let parentObj = this.tgtObj.parent; parentObj != null; parentObj = parentObj.parent) {
                        tgtObjParents.push(parentObj);
                    }

                    for (const srcParent of srcObjParents) {
                        for (const tgtParent of tgtObjParents) {
                            if (tgtParent.absPathName != "")
                                srcParent.targetParentSet.add(tgtParent);

                            if (srcParent.absPathName != "")
                                tgtParent.sourceParentSet.add(srcParent);
                        }
                    }
                }
            }
        }
    }
}
