// <<hpp_insert libs/pako_inflate.min.js>>
// <<hpp_insert libs/json5_2.2.0.min.js>>
// <<hpp_insert gen/TreeNode.js>>
// <<hpp_insert gen/NodeConnection.js>>

const defaultAttribNames = {
    name: 'name',              // Human-readable label for the node
    type: 'type',              // Label for the node's classification property
    descendants: 'children',   // Property that contains the node's descendants
    links: 'connections_list', // Property the contains the list of connections, in
                               // a [{src: "srcname1", tgt: "tgtname1", ...}] format
};

/**
 * Manages the data in the model and the connections between nodes.
 * @typedef ModelData
 * @property {Object[]} conns Connections: array of objects in format {src: 'path', tgt: 'path'}
 * @property {NodeConnection[]} connObjs The array of processed connection objects.
 * @property {Number} maxDepth The farthest distance of any descendant from the root node.
 * @property {Object} nodePaths An object with keys that are the pathnames of nodes, and values
 *  that are the associated TreeNode.
 * @property {TreeNode[]} nodeIds An array whose indices are the ids of their assicated values.
 * @property {Number[]} depthCount The tally of nodes at each depth.
 * @property {TreeNode} root The starting node in the tree.
 */
class ModelData {
    constructor(modelJSON, attribNames = defaultAttribNames) {

        this._attribNames = attribNames;
        this.conns = modelJSON[attribNames.links];
        this.connObjs = [];
        this.maxDepth = 1;
        this.nodePaths = {};
        this.nodeIds = [];
        this.depthCount = [];

        this._init(modelJSON);

        this.root = this.tree = modelJSON.tree = this._adoptNodes(modelJSON.tree);
        this._setParentsAndDepth(this.root, null, 1);

        for (const conn of this.conns) {
            this.connObjs.push(this._newConnectionObj(conn));
        }

    }

    /**
     * Tasks to perform early from the superclass constructor.
     * @param {Object} modelJSON The model object generated from a JSON string.
     */
    _init(modelJSON) { }

    /**
     * Given a string with base64-encoded, zlib-compressed JSON data, uncompress it
     * and return an Object created from the data.
     * @param {String} b64str The string with base64-encoded, zlib-compressed JSON.
     * @returns {Object} Created from supplied JSON.
     */
    static uncompressModel(b64str) {
        const compressedData = atob(b64str);
        const jsonStr = window.pako.inflate(compressedData, { to: 'string' });

        // JSON5 can handle Inf and NaN
        return JSON5.parse(jsonStr); 
    }

    /**
     * Create a new TreeNode object for this type of model tree. Can be overridden
     * to create different type of node objects derived from TreeNode.
     * @param {Object} element A simple object that will be REPLACED with the TreeNode.
     * @param {Object} attribNames The customized attribute names for this model.
     */
    _newNode(element, attribNames) {
        return new TreeNode(element, attribNames);
    }

    /**
     * Create a new connection object based on the src and tgt values in conn.
     * Can be overwritten to generate different type of connection objects.
     * @param {Object} conn An Object with src and tgt strings containing path names.
     * @returns {NodeConnection} The new object.
     */
    _newConnectionObj(conn) {
        return new NodeConnection(conn, this.nodePaths);
    }

    /**
     * Recurse over the tree and replace the JSON objects
     * provided by n2_viewer.py with TreeNodes.
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
     * @param {TreeNode} node Item to process.
     * @param {TreeNode} parent Parent of node, null for root node.
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
            node.path = (node.parent.path == '')? node.name : `${node.parent.path}.${node.name}`;
            this.nodePaths[node.path] = node;
        }

        this.maxDepth = Math.max(depth, this.maxDepth);
        node.childNames.add(node.path); // Add the node itself

        if (node.hasChildren()) {
            node.numDescendants = node.children.length;
            for (const child of node.children) {
                this._setParentsAndDepth(child, node, depth + 1);
                node.numDescendants += child.numDescendants;

                // Add absolute pathnames of children to a set for quick searching
                if (!node.isRoot()) { // All nodes are children of the model root
                    node.childNames.add(child.path);
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

    /**
     * Add all leaf descendents of specified node to the array.
     * @param {TreeNode} node Current node to work on.
     * @param {TreeNode[]} objArray Array to add to.
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
