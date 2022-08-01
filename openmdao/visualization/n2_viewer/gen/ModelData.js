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
        this._setDepth(this.root, 1);

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
     * @param {TreeNode} parent The node whose children array that this new node will be in.
     * @returns {TreeNode} The newly-created object.
     */
    _newNode(element, attribNames, parent) {
        return new TreeNode(element, attribNames, parent);
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
     _adoptNodes(element, parent = null) {
        const newNode = this._newNode(element, this._attribNames, parent);

        if (newNode.hasChildren()) {
            for (const i in newNode.children) {
                newNode.children[i] = this._adoptNodes(newNode.children[i], newNode);
            }
        }

        if (newNode.canFilter()) newNode.addFilterChild(this._attribNames);

        return newNode;
    }

   /**
     * Sets depth of all nodes and determines max depth.
     * @param {TreeNode} node Item to process.
     * @param {number} depth Numerical level of ancestry.
     */
    _setDepth(node, depth) {
        node.depth = depth;
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
                this._setDepth(child, depth + 1);
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

   /**
     * Recurse through the model, and determine whether a parent node is
     * minimized or manually expanded, or an input/output hidden. If it is,
     * add it to the hiddenList array, and optionally reset its state.
     * @param {Object[]} hiddenList The provided array to populate.
     * @param {Boolean} reveal If true, make the node visible.
     * @param {TreeNode} node The current node to operate on.
     */
    findAllHidden(hiddenList, reveal = false, node = this.root) {
        // Filtered nodes are handled by their true parents
        if (node.isFilter()) return;

        if (!node.isVisible() || node.draw.minimized || node.draw.manuallyExpanded) {
            hiddenList.push({
                'node': node,
                'draw': {
                    'minimized': node.draw.minimized,
                    'hidden': node.draw.hidden,
                    'filtered': node.draw.filtered,
                    'manuallyExpanded': node.draw.manuallyExpanded
                }
            })

            if (reveal) {
                node.expand();
                node.show();
                node.draw.manuallyExpanded = false;
                if (node.isFilteredVariable()) node.removeSelfFromFilter();
            }
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.findAllHidden(hiddenList, reveal, child);
            }
        }
    }

    /**
     * Restore the minimized/hidden value to all the specified nodes.
     * @param {Object[]} hiddenList The list of preserved objects
     * @param {TreeNode} node The current node to operate on.
     */
    resetAllHidden(hiddenList, node = this.root) {
        // Filtered nodes are handled by their true parents
        if (!hiddenList || node.isFilter()) return;

        const foundEntry = hiddenList.find(item => item.node === node);

        // If variables were selectively hidden, force the variable selection
        // dialog to rebuild the hiddenVars array.
        if (node.hasFilters()) { node.wipeFilters(); }

        if (!foundEntry) { // Not found, reset values to default
            node.expand();
            node.show();
            node.draw.manuallyExpanded = false;
            if (node.isFilteredVariable()) node.removeSelfFromFilter();
        }
        else { // Found, restore values
            node.draw.minimized = foundEntry.draw.minimized;
            node.draw.hidden = foundEntry.draw.hidden;
            node.draw.manuallyExpanded = foundEntry.draw.manuallyExpanded;
            if (foundEntry.draw.filtered) { node.addSelfToFilter(); }
            else { if (node.isFilteredVariable()) node.removeSelfFromFilter(); }
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.resetAllHidden(hiddenList, child);
            }
        }
    }

    /**
     * Set the node as not minimized and manually expanded, as well as
     * all children.
     * @param {TreeNode} startNode The node to begin from.
     */
     manuallyExpandAll(startNode) {
        startNode.draw.minimized = false;
        startNode.draw.manuallyExpanded = true;

        if (startNode.hasChildren()) {
            for (const child of startNode.children) {
                this.manuallyExpandAll(child);
            }
        }
    }

    /**
     * Set all the children of the specified node as minimized and not manually expanded.
     * @param {TreeNode} startNode The node to begin from.
     * @param {Boolean} [initialNode = true] Indicate the starting node.
     */
    minimizeAll(startNode, initialNode = true) {
        if (!initialNode) {
            startNode.draw.minimized = true;
            startNode.draw.manuallyExpanded = false;
        }

        if (startNode.hasChildren()) {
            for (const child of startNode.children) {
                this.minimizeAll(child, false);
            }
        }
    }

    /**
     * Recursively minimize non-input nodes to the specified depth.
     * @param {TreeNode} node The node to work on.
     * @param {Number} chosenCollapseDepth If the node's depth is the same or more, collapse it.
     */
    minimizeToDepth(node, chosenCollapseDepth) {
        if (node.isInputOrOutput()) {
            return;
        }

        if (node.depth < chosenCollapseDepth) {
            node.draw.minimized = false;
            node.draw.manuallyExpanded = true;
        }
        else {
            node.draw.minimized = true;
            node.draw.manuallyExpanded = false;
        }

        if (node.hasChildren()) {
            for (const child of node.children) {
                this.minimizeToDepth(child, chosenCollapseDepth);
            }
        }
    }


}
