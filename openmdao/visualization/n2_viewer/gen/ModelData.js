// <<hpp_insert libs/pako_inflate.min.js>>
// <<hpp_insert libs/json5_2.2.0.min.js>>
// <<hpp_insert gen/N2TreeNode.js>>

const defaultAttribNames = {
    name: 'name',              // Human-readable label for the node
    type: 'type',              // Label for the node's classification property
    descendants: 'children',   // Property that contains the node's descendants
    links: 'connections_list', // Property the contains the list of connections, in
                               // a [{src: "srcname1", tgt: "tgtname1", ...}] format
};

class ModelData {
    constructor(modelJSON, attribNames = defaultAttribNames) {
        // console.log(modelJSON);

        this._attribNames = attribNames;
        this.conns = modelJSON[attribNames.links];
        this.maxDepth = 1;
        this.nodePaths = {};
        this.nodeIds = [];
        this.depthCount = [];

        this._init(modelJSON);

        this.root = this.tree = modelJSON.tree = this._adoptNodes(modelJSON.tree);
        this._setParentsAndDepth(this.root, null, 1);
        this._computeConnections();

    }

    _init(modelJSON) { }

    static uncompressModel(b64str) {
        const compressedData = atob(b64str);
        const jsonStr = window.pako.inflate(compressedData, { to: 'string' });

        // JSON5 can handle Inf and NaN
        return JSON5.parse(jsonStr); 
    }

    _newNode(element, attribNames) {
        return new N2TreeNode(element, attribNames);
    }

    /**
     * Recurse over the tree and replace the JSON objects
     * provided by n2_viewer.py with N2TreeNodes.
     * @param {Object} element The current element being updated.
     */
     _adoptNodes(element) {
        const newNode = this._newNode(element, this._attribNames);

        if (newNode.hasChildren()) {
            for (let i in newNode.children) {
                newNode.children[i] = this._adoptNodes(newNode.children[i]);
                newNode.children[i].parent = newNode;
                if (exists(newNode.children[i].parentComponent))
                    newNode.children[i].parentComponent = newNode;
            }
        }

        newNode.addFilterChild(this._attribNames);

        return newNode;
    }

   /**
     * Sets parents and depth of all nodes, and determine max depth.
     * @param {N2TreeNode} node Item to process.
     * @param {N2TreeNode} parent Parent of node, null for root node.
     * @param {number} depth Numerical level of ancestry.
     */
    _setParentsAndDepth(node, parent, depth) {
        node.depth = depth;
        node.parent = parent;
        node.id = this.nodeIds.length;
        this.nodeIds.push(node);

        // Track # of nodes at each depth
        if (depth > this.depthCount.length) { this.depthCount.push(1); }
        else { this.depthCount[depth - 1]++; }

        if (node.parent) {
            node.uuid = (node.parent.uuid == '')? node.name : `${node.parent.uuid}.${node.name}`;
            this.nodePaths[node.uuid] = node;
        }

        this.maxDepth = Math.max(depth, this.maxDepth);
        node.childNames.add(node.uuid); // Add the node itself

        if (node.hasChildren()) {
            node.numDescendants = node.children.length;
            for (const child of node.children) {
                this._setParentsAndDepth(child, node, depth + 1);
                node.numDescendants += child.numDescendants;

                // Add absolute pathnames of children to a set for quick searching
                if (!node.isRoot()) { // All nodes are children of the model root
                    node.childNames.add(child.uuid);
                    for (const childName of child.childNames) {
                        node.childNames.add(childName);
                    }
                }
            }
        }
    }

    /**
     * Check the entire array of model connections for any with a target matching
     * the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a target in the connection list.
     */
     hasInputConnection(elementPath) {
        for (const conn of this.conns) {
            if (conn.tgt.match(elementPath)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Check the entire array of model connections for any with a source matching
     * the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a source in the connection list.
     */
    hasOutputConnection(elementPath) {
        for (const conn of this.conns) {
            if (conn.src.match(elementPath)) {
                return true;
            }
        }

        return false;
    }

    /**
     * Check the entire array of model connections for any with a source OR
     * target matching the specified path.
     * @param {string} elementPath The full path of the element to check.
     * @return True if the path is found as a source in the connection list.
     */
    hasAnyConnection(elementPath) {
        for (const conn of this.conns) {
            if (conn.src == elementPath || conn.tgt == elementPath)
                return true;
        }

        debugInfo(elementPath + " has no connections.");
        this.unconnectedInputs++;

        return false;
    }

    /** A stub to be overridden by a derived class */
    _additionalConnProcessing(conn, srcObj, tgtObj) { }

    /**
     * Iterate over the connections list, and find the objects that make up
     * each connection, and do some error checking. Store an array containing the
     * target object and all of its parents in the source object and all of *its*
     * parents.
     */
    _computeConnections() {
        const throwLbl = 'ModelData._computeConnections: ';

        for (const conn of this.conns) {
            // Process sources
            const srcObj = this.nodePaths[conn.src];

            if (!srcObj) {
                console.warn(throwLbl + "Cannot find connection source " + conn.src);
                continue;
            }

            const srcObjParents = [srcObj];
            if (!srcObj.isOutput()) { // source obj must be output
                console.warn(throwLbl + "Found a source that is not an output.");
                continue;
            }

            if (srcObj.hasChildren()) {
                console.warn(throwLbl + "Found a source that has children.");
                continue;
            }

            for (let obj = srcObj.parent; obj != null; obj = obj.parent) {
                srcObjParents.push(obj);
            }

            // Process targets
            const tgtObj = this.nodePaths[conn.tgt];

            if (!tgtObj) {
                console.warn(throwLbl + "Cannot find connection target " + conn.tgt);
                continue;
            }

            // Target obj must be an input
            if (!tgtObj.isInput()) {
                console.warn(throwLbl + "Found a target that is NOT a input.");
                continue;
            }
            if (tgtObj.hasChildren()) {
                console.warn(throwLbl + "Found a target that has children.");
                continue;
            }

            if (!tgtObj.parentComponent) {
                console.warn(`${throwLbl} Target object ${conn.tgt} is missing a parent component.`);
                continue;
            }

            const tgtObjParents = [tgtObj];
            for (let parentObj = tgtObj.parent; parentObj != null; parentObj = parentObj.parent) {
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

            this._additionalConnProcessing(conn, srcObj, tgtObj);
        }
    }

    /**
     * Add all leaf descendents of specified node to the array.
     * @param {N2TreeNode} node Current node to work on.
     * @param {N2TreeNode[]} objArray Array to add to.
     */
     _addLeaves(node, objArray) {
        if (!node.isInput()) {
            objArray.push(node);
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this._addLeaves(child, objArray);
            }
        }
    }
}
