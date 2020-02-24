
const UID = '_U_';
const UNAME = 'User';
const MULTI_TYPE = '_multi';

const STATUS = {
  UNKNOWN: 'UNKNOWN',
  PENDING: 'PENDING',
  RUNNING: 'RUNNING',
  DONE: 'DONE',
  FAILED: 'FAILED',
};

// *** Node *******************************************************************
function Node(id, pname, ptype, pstatus, psubxdsm) {
  const name = pname || id;
  const type = ptype || 'analysis';
  const status = pstatus || STATUS.UNKNOWN;
  if (typeof STATUS[status] === 'undefined') {
    throw Error(`Unknown status '${status}' for node ${name}(id=${id})`);
  }
  this.id = id;
  this.name = name;
  this.isMulti = (type.search(/_multi$/) >= 0);
  this.type = this.isMulti ? type.substr(0, type.length - MULTI_TYPE.length)
    : type;
  this.status = status;
  this.subxdsm = psubxdsm;
}

Node.prototype.isComposite = function isComposite() {
  return this.type === 'mdo' || this.type === 'sub-optimization'
    || this.type === 'group' || this.type === 'implicit-group';
};

Node.prototype.getSubXdsmId = function getSubXdsmId() {
  if (this.isComposite()) {
    // Deprecated
    const idxscn = this.name.indexOf('_scn-');
    if (idxscn === -1) {
      // console.log(`${'Warning: MDO Scenario not found. '
      //   + 'Bad type or name for node: '}${JSON.stringify(this)}`);
    } else {
      console.log("Use of <name>_scn-<id> pattern in node.name to detect sub scenario 'scn-<id>'"
        + ' is deprecated. Use node.subxdsm property instead (i.e. node.subxdsm = <id>)');
      return this.name.substr(idxscn + 1);
    }
    if (this.subxdsm) {
      return this.subxdsm;
    }
    console.log(`${'Warning: Sub XDSM id not found. '
        + 'Bad type or name for node: '}${JSON.stringify(this)}`);
  }
  return null;
};

// *** Edge *******************************************************************
function Edge(from, to, nameOrVars, row, col, isMulti) {
  this.id = `link_${from}_${to}`;
  if (typeof (nameOrVars) === 'string') {
    this.name = nameOrVars;
    this.vars = {};
    const vars = this.name.split(',');
    vars.forEach((n, i) => { this.vars[i] = n.trim(); });
  } else { // vars = {id: name, ...}
    this.vars = nameOrVars;
    const names = [];
    for (const k in this.vars) {
      if (Object.prototype.hasOwnProperty.call(this.vars, k)) {
        names.push(this.vars[k]);
      }
    }
    this.name = names.join(', ');
  }
  this.row = row;
  this.col = col;
  this.iotype = row < col ? 'in' : 'out';
  this.isMulti = isMulti;
}

Edge.prototype.addVar = function addVar(nameOrVar) {
  if (typeof (nameOrVar) === 'string') {
    if (this.name === '') {
      this.name = nameOrVar;
    } else {
      this.name += `, ${nameOrVar}`;
    }
    this.vars[this.vars.length] = nameOrVar;
  } else {
    for (const k in nameOrVar) {
      if (Object.prototype.hasOwnProperty.call(nameOrVar, k)) {
        this.vars[k] = nameOrVar[k];
        const names = [];
        for (const v in this.vars) {
          if (Object.prototype.hasOwnProperty.call(this.vars, v)) {
            names.push(this.vars[v]);
          }
        }
        this.name = names.join(', ');
      }
    }
  }
};

Edge.prototype.removeVar = function removeVar(nameOrId) {
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
      if (Object.prototype.hasOwnProperty.call(this.vars, k)) {
        names.push(this.vars[k]);
      }
    }
    this.name = names.join(', ');
  }
};

// *** Graph ******************************************************************
function Graph(mdo, refname, noDefaultDriver) {
  this.nodes = [new Node(UID, UNAME, 'driver')];
  if (noDefaultDriver) {
    this.nodes = [];
  }

  this.edges = [];
  this.chains = [];
  this.refname = refname || '';

  const numbering = Graph.number(mdo.workflow);
  const numPrefixes = numbering.toNum;
  this.nodesByStep = numbering.toNode;

  mdo.nodes.forEach((item) => {
    const num = numPrefixes[item.id];
    this.nodes.push(new Node(item.id, num ? `${num}:${item.name}` : item.name,
      item.type, item.status, item.subxdsm));
  }, this);
  this.uid = this.nodes[0].id;

  if (mdo.edges) {
    mdo.edges.forEach((item) => {
      this.addEdge(item.from, item.to, item.vars ? item.vars : item.name);
    }, this);
  }

  if (mdo.workflow) {
    this._makeChaining(mdo.workflow);
  }
}

Graph.NODE_STATUS = STATUS;

Graph.prototype._makeChaining = function _makeChaining(workflow) {
  const echain = Graph.expand(workflow);
  echain.forEach((leafChain) => {
    if (leafChain.length < 2) {
      throw new Error(`Bad process chain (${leafChain.length}elt)`);
    } else {
      this.chains.push([]);
      const ids = this.nodes.map((elt) => elt.id);
      leafChain.forEach((item, j) => {
        if (j !== 0) {
          const idA = ids.indexOf(leafChain[j - 1]);
          if (idA < 0) {
            throw new Error(`Process chain element (${leafChain[j - 1]}) not found`);
          }
          const idB = ids.indexOf(leafChain[j]);
          if (idB < 0) {
            throw new Error(`Process chain element (${leafChain[j]}) not found`);
          }
          if (idA !== idB) {
            this.chains[this.chains.length - 1].push([idA, idB]);
          }
        }
      }, this);
    }
  }, this);
};

Graph.prototype.idxOf = function idxOf(nodeId) {
  const idx = this.nodes.map((elt) => elt.id).indexOf(nodeId);
  if (idx < 0) {
    throw new Error(`Graph.idxOf: ${nodeId} not found in ${JSON.stringify(this.nodes)}`);
  }
  return idx;
};

Graph.prototype.getNode = function getNode(nodeId) {
  let theNode;
  this.nodes.forEach((node) => {
    if (node.id === nodeId) {
      theNode = node;
    }
  }, this);
  return theNode;
};

Graph.prototype.getNodeFromIndex = function getNodeFromIndex(idx) {
  let node;
  if (idx >= 0 && idx < this.nodes.length) {
    node = this.nodes[idx];
  } else {
    throw new Error(`Index out of range : ${idx} not in [0, ${
      this.nodes.length - 1}]`);
  }
  return node;
};

Graph.prototype.addNode = function addNode(node) {
  this.nodes.push(new Node(node.id, node.name, node.kind));
};

Graph.prototype.removeNode = function removeNode(index) {
  const self = this;
  // Update edges
  const edges = this.findEdgesOf(index);
  edges.toRemove.forEach((edge) => {
    const idx = self.edges.indexOf(edge);
    if (idx > -1) {
      self.edges.splice(idx, 1);
    }
  }, this);
  edges.toShift.forEach((edge) => {
    if (edge.row > 1) {
      // eslint-disable-next-line no-param-reassign
      edge.row -= 1;
    }
    if (edge.col > 1) {
      // eslint-disable-next-line no-param-reassign
      edge.col -= 1;
    }
  }, this);

  // Update nodes
  this.nodes.splice(index, 1);
};

Graph.prototype.moveLeft = function moveLeft(index) {
  if (index > 1) {
    const tmp = this.nodes[index - 1];
    this.nodes[index - 1] = this.nodes[index];
    this.nodes[index] = tmp;
  }
};

Graph.prototype.moveRight = function moveRight(index) {
  if (index < this.nodes.length - 1) {
    const tmp = this.nodes[index + 1];
    this.nodes[index + 1] = this.nodes[index];
    this.nodes[index] = tmp;
  }
};

Graph.prototype.addEdge = function addEdge(nodeIdFrom, nodeIdTo, nameOrVar) {
  const idA = this.idxOf(nodeIdFrom);
  const idB = this.idxOf(nodeIdTo);
  const isMulti = this.nodes[idA].isMulti || this.nodes[idB].isMulti;
  this.edges
    .push(new Edge(nodeIdFrom, nodeIdTo, nameOrVar, idA, idB, isMulti));
};

Graph.prototype.removeEdge = function removeEdge(nodeIdFrom, nodeIdTo) {
  const edge = this.findEdge(nodeIdFrom, nodeIdTo);
  this.edges.splice(edge.index, 1);
};

Graph.prototype.addEdgeVar = function addEdgeVar(nodeIdFrom, nodeIdTo, nameOrVar) {
  const edge = this.findEdge(nodeIdFrom, nodeIdTo).element;
  if (edge) {
    console.log(nameOrVar);
    edge.addVar(nameOrVar);
  } else {
    this.addEdge(nodeIdFrom, nodeIdTo, nameOrVar);
  }
};

Graph.prototype.removeEdgeVar = function removeEdgeVar(nodeIdFrom, nodeIdTo, nameOrId) {
  const ret = this.findEdge(nodeIdFrom, nodeIdTo);
  const edge = ret.element;
  const { index } = ret;
  if (edge) {
    edge.removeVar(nameOrId);
  }
  if (edge.name === '') {
    this.edges.splice(index, 1);
  }
};

Graph.prototype.findEdgesOf = function findEdgesOf(nodeIdx) {
  const toRemove = [];
  const toShift = [];
  this.edges.forEach((edge) => {
    if ((edge.row === nodeIdx) || (edge.col === nodeIdx)) {
      toRemove.push(edge);
    } else if ((edge.row > nodeIdx) || (edge.col > nodeIdx)) {
      toShift.push(edge);
    }
  }, this);
  return {
    toRemove,
    toShift,
  };
};

Graph.prototype.findEdge = function findEdge(nodeIdFrom, nodeIdTo) {
  let element;
  let index = -1;
  const idxFrom = this.idxOf(nodeIdFrom);
  const idxTo = this.idxOf(nodeIdTo);
  this.edges.some((edge, i) => {
    if ((edge.row === idxFrom) && (edge.col === idxTo)) {
      if (element) {
        throw Error(`edge have be uniq between two nodes, but got: ${
          JSON.stringify(element)} and ${JSON.stringify(edge)}`);
      }
      element = edge;
      index = i;
      return true;
    }
    return false;
  }, this);
  return { element, index };
};

function _expand(workflow) {
  let ret = [];
  let prev;
  workflow.forEach((item) => {
    if (item instanceof Array) {
      if (Object.prototype.hasOwnProperty.call(item[0], 'parallel')) {
        if (prev) {
          ret = ret.slice(0, ret.length - 1).concat(
            item[0].parallel.map((elt) => [prev].concat(_expand([elt]), prev)),
          );
        } else {
          throw new Error('Bad workflow structure : '
            + 'cannot parallel loop without previous starting point.');
        }
      } else if (prev) {
        ret = ret.concat(_expand(item), prev);
      } else {
        ret = ret.concat(_expand(item));
      }
      prev = ret[ret.length - 1];
    } else if (Object.prototype.hasOwnProperty.call(item, 'parallel')) {
      if (prev) {
        ret = ret.slice(0, ret.length - 1).concat(
          item.parallel.map((elt) => [prev].concat(_expand([elt]))),
        );
      } else {
        ret = ret.concat(item.parallel.map((elt) => _expand([elt])));
      }
      prev = undefined;
    } else {
      let i = ret.length - 1;
      let flagParallel = false;
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

Graph._isPatchNeeded = function _isPatchNeeded(toBePatched) {
  const lastElts = toBePatched.map((arr) => arr[arr.length - 1]);
  const lastElt = lastElts[0];
  for (let i = 0; i < lastElts.length; i += 1) {
    if (lastElts[i] !== lastElt) {
      return true;
    }
  }
  return false;
};

Graph._patchParallel = function _patchParallel(expanded) {
  const toBePatched = [];
  expanded.forEach((elt) => {
    if (elt instanceof Array) {
      toBePatched.push(elt);
    } else if (Graph._isPatchNeeded(toBePatched)) {
      toBePatched.forEach((arr) => {
        arr.push(elt);
      }, this);
    }
  }, this);
};

Graph.expand = function expand(item) {
  const expanded = _expand(item);
  const result = [];
  let current = [];
  // first pass to add missing 'end link' in case of parallel branches at the
  // end of a loop
  // [a, [b, d], [b, c], a] -> [a, [b, d, a], [b, c, a], a]
  Graph._patchParallel(expanded);
  // [a, aa, [b, c], d] -> [[a, aa, b], [b,c], [c, d]]
  expanded.forEach((elt) => {
    if (elt instanceof Array) {
      if (current.length > 0) {
        current.push(elt[0]);
        result.push(current);
        current = [];
      }
      result.push(elt);
    } else {
      if (result.length > 0 && current.length === 0) {
        const lastChain = result[result.length - 1];
        const lastElt = lastChain[lastChain.length - 1];
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

Graph.number = function number(workflow, pnum) {
  let num = (typeof pnum === 'undefined') ? 0 : pnum;
  const toNum = {};
  const toNode = [];

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
      num = `${end}-${beg}`;
      setStep(end, nodeId);
    }
    if (nodeId in toNum) {
      toNum[nodeId] += `,${num}`;
    } else {
      toNum[nodeId] = num;
    }
  }

  function _number(wks, nb) {
    let ret = 0;
    if (wks instanceof Array) {
      if (wks.length === 0) {
        ret = nb;
      } else if (wks.length === 1) {
        ret = _number(wks[0], nb);
      } else {
        const head = wks[0];
        const tail = wks.slice(1);
        let beg = _number(head, nb);
        if (tail[0] instanceof Array) {
          const end = _number(tail[0], beg);
          setNum(head, beg, end);
          beg = end + 1;
          tail.shift();
        }
        ret = _number(tail, beg);
      }
    } else if ((wks instanceof Object) && 'parallel' in wks) {
      const nums = wks.parallel.map((branch) => _number(branch, nb));
      ret = Math.max.apply(null, nums);
    } else {
      setNum(wks, nb);
      ret = nb + 1;
    }
    return ret;
  }

  _number(workflow, num);
  // console.log('toNodes=', JSON.stringify(toNode));
  // console.log('toNum=',JSON.stringify(toNum));
  return {
    toNum,
    toNode,
  };
};

export default Graph;
