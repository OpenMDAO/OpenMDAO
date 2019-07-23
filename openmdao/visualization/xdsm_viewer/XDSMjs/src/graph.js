'use strict';
var UID = "_U_";
var UNAME = "User";
var MULTI_TYPE = "_multi";

var STATUS = {
  UNKNOWN: 'UNKNOWN',
  PENDING: 'PENDING',
  RUNNING: 'RUNNING',
  DONE: 'DONE',
  FAILED: 'FAILED',
};

// *** Node *******************************************************************
function Node(id, name, type, status) {
  if (typeof (name) === 'undefined') {
    name = id;
  }
  if (typeof (type) === 'undefined') {
    type = 'analysis';
  }
  if (typeof (status) === 'undefined') {
    status = STATUS.UNKNOWN;
  }
  if (typeof STATUS[status] === 'undefined') {
    throw Error("Unknown status '" + status + "' for node " + name + "(id="
        + id + ")");
  }
  this.id = id;
  this.name = name;
  this.isMulti = (type.search(/_multi$/) >= 0);
  this.type = this.isMulti ? type.substr(0, type.length - MULTI_TYPE.length)
      : type;
  this.status = status;
}

Node.prototype.isMdo = function() {
  return this.type === "mdo";
};

Node.prototype.getScenarioId = function() {
  if (this.isMdo()) {
    var idxscn = this.name.indexOf("_scn-");
    if (idxscn === -1) {
      console.log("Warning: MDO Scenario not found. "
          + "Bad type or name for node: " + JSON.stringify(this));
      return null;
    }
    return this.name.substr(idxscn + 1);
  }
  return null;
};

// *** Edge *******************************************************************
function Edge(from, to, nameOrVars, row, col, isMulti) {
  this.id = "link_" + from + "_" + to;
  if (typeof (nameOrVars) === "string") {
    this.name = nameOrVars;
    this.vars = {};
    const vars = this.name.split(',');
    vars.forEach((n, i) => this.vars[i] = n.trim());
  } else { // vars = {id: name, ...}
    this.vars = nameOrVars;
    const names = [];
    for (const k in this.vars) {
      if (this.vars.hasOwnProperty(k)) {
        names.push(this.vars[k]);
      }
    }
    this.name = names.join(", ");
  }
  this.row = row;
  this.col = col;
  this.iotype = row < col ? "in" : "out";
  this.isMulti = isMulti;
}

Edge.prototype.addVar = function(nameOrVar) {
  if (typeof(nameOrVar) === "string") {
    if (this.name==="") {
      this.name=nameOrVar;
    } else {
      this.name+=", "+nameOrVar;
    }
    this.vars[this.vars.length] = nameOrVar;
  } else {
    for (const k in nameOrVar) {
      if (nameOrVar.hasOwnProperty(k)) {
        this.vars[k] = nameOrVar[k];
        const names = [];
        for (const k in this.vars) {
          if (this.vars.hasOwnProperty(k)) {
            names.push(this.vars[k]);
          }
        }
        this.name = names.join(", ");
      }
    }
  }
};

Edge.prototype.removeVar = function(nameOrId) {
  let found = false;
  for (const k in this.vars) {
    if (k === nameOrId) {
      this.vars.delete(k);
      found = true;
    }
  }
  if (!found) {
    const newvars = {};
    for (const k in this.vars) {
      if (this.vars[k] === nameOrId) {
        found = true;
      } else {
        newvars[k] = this.vars[k];
      }
    }
    this.vars = newvars;
  }
  if (found) {
    const names = [];
    for (const k in this.vars) {
      if (this.vars.hasOwnProperty(k)) {
        names.push(this.vars[k]);
      }
    }
    this.name = names.join(", ");
  }
};

// *** Graph ******************************************************************
function Graph(mdo, refname, noDefaultDriver) {
  this.nodes = [new Node(UID, UNAME, "driver")];
  if (noDefaultDriver) {
    this.nodes = [];
  }

  this.edges = [];
  this.chains = [];
  this.refname = refname || "";

  var numbering = Graph.number(mdo.workflow);
  var numPrefixes = numbering.toNum;
  this.nodesByStep = numbering.toNode;

  mdo.nodes.forEach(function(item) {
    var num = numPrefixes[item.id];
    this.nodes.push(new Node(item.id, num ? num + ":" + item.name : item.name,
        item.type, item.status));
  }, this);
  this.uid = this.nodes[0].id;

  if (mdo.edges) {
    mdo.edges.forEach(function(item) {
      this.addEdge(item.from, item.to, item.vars?item.vars:item.name);
    }, this);
  }

  if (mdo.workflow) {
    this._makeChaining(mdo.workflow);
  }
}

Graph.NODE_STATUS = STATUS;

Graph.prototype._makeChaining = function(workflow) {
  var echain = Graph.expand(workflow);
  echain.forEach(function(leafChain) {
    if (leafChain.length < 2) {
      throw new Error("Bad process chain (" + leafChain.length + "elt)");
    } else {
      this.chains.push([]);
      var ids = this.nodes.map(function(elt) {
        return elt.id;
      });
      leafChain.forEach(function(item, j) {
        if (j !== 0) {
          var idA = ids.indexOf(leafChain[j - 1]);
          if (idA < 0) {
            throw new Error("Process chain element (" + leafChain[j - 1]
                + ") not found");
          }
          var idB = ids.indexOf(leafChain[j]);
          if (idB < 0) {
            throw new Error("Process chain element (" + leafChain[j]
                + ") not found");
          }
          if (idA !== idB) {
            this.chains[this.chains.length - 1].push([idA, idB]);
          }
        }
      }, this);
    }
  }, this);
};

Graph.prototype.idxOf = function(nodeId) {
  const idx = this.nodes.map(function(elt) {
    return elt.id;
  }).indexOf(nodeId);
  if (idx<0) {
    throw new Error("Graph.idxOf: "+nodeId+" not found in "+JSON.stringify(this.nodes));
  }
  return idx;
};

Graph.prototype.getNode = function(nodeId) {
  var theNode;
  this.nodes.forEach(function(node) {
    if (node.id === nodeId) {
      theNode = node;
    }
  }, this);
  return theNode;
};

Graph.prototype.getNodeFromIndex = function(idx) {
  var node;
  if (idx >= 0 && idx < this.nodes.length) {
    node = this.nodes[idx];
  } else {
    throw new Error("Index out of range : " + idx + " not in [0, "
        + (this.nodes.length - 1) + "]");
  }
  return node;
};

Graph.prototype.addNode = function(node) {
  this.nodes.push(new Node(node['id'], node['name'], node['kind']));
};

Graph.prototype.removeNode = function(index) {
  var self = this;
  // Update edges
  var edges = this.findEdgesOf(index);
  edges.toRemove.forEach(function(edge) {
    var idx = self.edges.indexOf(edge);
    if (idx > -1) {
      self.edges.splice(idx, 1);
    }
  }, this);
  edges.toShift.forEach(function(edge) {
    if (edge.row > 1) {
      edge.row -= 1;
    }
    if (edge.col > 1) {
      edge.col -= 1;
    }
  }, this);

  // Update nodes
  this.nodes.splice(index, 1);
};

Graph.prototype.moveLeft = function(index) {
  if (index>1) {
    const tmp = this.nodes[index-1];
    this.nodes[index-1] = this.nodes[index];
    this.nodes[index] = tmp;
  }
};

Graph.prototype.moveRight = function(index) {
  if (index<this.nodes.length-1) {
    const tmp = this.nodes[index+1];
    this.nodes[index+1] = this.nodes[index];
    this.nodes[index] = tmp;
  }
};

Graph.prototype.addEdge = function(nodeIdFrom, nodeIdTo, nameOrVar) {
  var idA = this.idxOf(nodeIdFrom);
  var idB = this.idxOf(nodeIdTo);
  var isMulti = this.nodes[idA].isMulti || this.nodes[idB].isMulti;
  this.edges
      .push(new Edge(nodeIdFrom, nodeIdTo, nameOrVar, idA, idB, isMulti));
};

Graph.prototype.removeEdge = function(nodeIdFrom, nodeIdTo) {
  const edge = this.findEdge(nodeIdFrom, nodeIdTo);
  this.edges.splice(edge.index, 1);
};

Graph.prototype.addEdgeVar = function(nodeIdFrom, nodeIdTo, nameOrVar) {
  const edge = this.findEdge(nodeIdFrom, nodeIdTo).element;
  if (edge) {
    console.log(nameOrVar);
    edge.addVar(nameOrVar);
  } else {
    this.addEdge(nodeIdFrom, nodeIdTo, nameOrVar);
  }
};

Graph.prototype.removeEdgeVar = function(nodeIdFrom, nodeIdTo, nameOrId) {
  const ret = this.findEdge(nodeIdFrom, nodeIdTo);
  const edge = ret.element;
  const index = ret.index;
  if (edge) {
    edge.removeVar(nameOrId);
  }
  if (edge.name === "") {
    this.edges.splice(index, 1);
  }
};

Graph.prototype.findEdgesOf = function(nodeIdx) {
  var toRemove = [];
  var toShift = [];
  this.edges.forEach(function(edge) {
    if ((edge.row === nodeIdx) || (edge.col === nodeIdx)) {
      toRemove.push(edge);
    } else if ((edge.row > nodeIdx) || (edge.col > nodeIdx)) {
      toShift.push(edge);
    }
  }, this);
  return {
    toRemove: toRemove,
    toShift: toShift,
  };
};

Graph.prototype.findEdge = function(nodeIdFrom, nodeIdTo) {
  let element;
  let index = -1;
  const idxFrom = this.idxOf(nodeIdFrom);
  const idxTo = this.idxOf(nodeIdTo);
  this.edges.some(function(edge, i) {
    if ((edge.row === idxFrom) && (edge.col === idxTo)) {
      if (element) {
        throw Error('edge have be uniq between two nodes, but got: '+
            JSON.stringify(element)+ ' and '+JSON.stringify(edge));
      }
      element = edge;
      index = i;
      return true;
    }
  }, this);
  return {element, index};
};

function _expand(workflow) {
  var ret = [];
  var prev;
  workflow.forEach(function(item) {
    if (item instanceof Array) {
      if (item[0].hasOwnProperty('parallel')) {
        if (prev) {
          ret = ret.slice(0, ret.length - 1).concat(
              item[0].parallel.map(function(elt) {
                return [prev].concat(_expand([elt]), prev);
              }));
        } else {
          throw new Error("Bad workflow structure : "
              + "cannot parallel loop without previous starting point.");
        }
      } else if (prev) {
        ret = ret.concat(_expand(item), prev);
      } else {
        ret = ret.concat(_expand(item));
      }
      prev = ret[ret.length - 1];
    } else if (item.hasOwnProperty('parallel')) {
      if (prev) {
        ret = ret.slice(0, ret.length - 1).concat(
            item.parallel.map(function(elt) {
              return [prev].concat(_expand([elt]));
            }));
      } else {
        ret = ret.concat(item.parallel.map(function(elt) {
          return _expand([elt]);
        }));
      }
      prev = undefined;
    } else {
      var i = ret.length - 1;
      var flagParallel = false;
      while (i >= 0 && (ret[i] instanceof Array)) {
        ret[i] = ret[i].concat(item);
        i -= 1;
        flagParallel = true;
      }
      if (!flagParallel) {
        ret.push(item);
      }
      prev = item;
    }
  });
  return ret;
}

Graph._isPatchNeeded = function(toBePatched) {
  var lastElts = toBePatched.map(function(arr) {
    return arr[arr.length - 1];
  });
  var lastElt = lastElts[0];
  for (var i = 0; i < lastElts.length; i++) {
    if (lastElts[i] !== lastElt) {
      return true;
    }
  }
  return false;
};

Graph._patchParallel = function(expanded) {
  var toBePatched = [];
  expanded.forEach(function(elt) {
    if (elt instanceof Array) {
      toBePatched.push(elt);
    } else if (Graph._isPatchNeeded(toBePatched)) {
      toBePatched.forEach(function(arr) {
        arr.push(elt);
      }, this);
    }
  }, this);
};

Graph.expand = function(item) {
  var expanded = _expand(item);
  var result = [];
  var current = [];
  // first pass to add missing 'end link' in case of parallel branches at the
  // end of a loop
  // [a, [b, d], [b, c], a] -> [a, [b, d, a], [b, c, a], a]
  Graph._patchParallel(expanded);
  // [a, aa, [b, c], d] -> [[a, aa, b], [b,c], [c, d]]
  expanded.forEach(function(elt) {
    if (elt instanceof Array) {
      if (current.length > 0) {
        current.push(elt[0]);
        result.push(current);
        current = [];
      }
      result.push(elt);
    } else {
      if (result.length > 0 && current.length === 0) {
        var lastChain = result[result.length - 1];
        var lastElt = lastChain[lastChain.length - 1];
        current.push(lastElt);
      }
      current.push(elt);
    }
  }, this);
  if (current.length > 0) {
    result.push(current);
  }
  return result;
};

Graph.number = function(workflow, num) {
  num = (typeof num === 'undefined') ? 0 : num;
  var toNum = {};
  var toNode = [];

  function setStep(step, nodeId) {
    if (step in toNode) {
      toNode[step].push(nodeId);
    } else {
      toNode[step] = [nodeId];
    }
  }

  function setNum(nodeId, beg, end) {
    if (end === undefined) {
      num = String(beg);
      setStep(beg, nodeId);
    } else {
      num = end + "-" + beg;
      setStep(end, nodeId);
    }
    if (nodeId in toNum) {
      toNum[nodeId] += "," + num;
    } else {
      toNum[nodeId] = num;
    }
  }

  function _number(wks, num) {
    var ret = 0;
    if (wks instanceof Array) {
      if (wks.length === 0) {
        ret = num;
      } else if (wks.length === 1) {
        ret = _number(wks[0], num);
      } else {
        var head = wks[0];
        var tail = wks.slice(1);
        var beg = _number(head, num);
        if (tail[0] instanceof Array) {
          var end = _number(tail[0], beg);
          setNum(head, beg, end);
          beg = end + 1;
          tail.shift();
        }
        ret = _number(tail, beg);
      }
    } else if ((wks instanceof Object) && 'parallel' in wks) {
      var nums = wks.parallel.map(function(branch) {
        return _number(branch, num);
      });
      ret = Math.max.apply(null, nums);
    } else {
      setNum(wks, num);
      ret = num + 1;
    }
    return ret;
  }

  _number(workflow, num);
  // console.log('toNodes=', JSON.stringify(toNode));
  // console.log('toNum=',JSON.stringify(toNum));
  return {
    toNum: toNum,
    toNode: toNode,
  };
};

export default Graph;
