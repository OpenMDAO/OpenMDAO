'use strict';
import {select, selectAll} from 'd3-selection';
import {rgb} from 'd3-color';
import Graph from './graph';

var PULSE_DURATION = 700;
var SUB_ANIM_DELAY = 200;
var RUNNING_COLOR = rgb("seagreen");
var FAILED_COLOR = rgb("firebrick");
var PENDING_COLOR = rgb("darkseagreen");
var DONE_COLOR = rgb("darkcyan");

function Animation(xdsms, rootId, delay) {
  this.rootId = rootId;
  if (typeof (rootId) === 'undefined') {
    this.rootId = 'root';
  }
  this.root = xdsms[this.rootId];
  this.xdsms = xdsms;
  this.duration = PULSE_DURATION;
  this.initialDelay = delay || 0;

  this._observers = [];
  this.reset();
}

Animation.STATUS = {READY: "ready",
  RUNNING_STEP: "running_step",
  RUNNING_AUTO: "running_auto",
  STOPPED: "stopped",
  DONE: "done",
  DISABLED: "disabled"};

Animation.prototype.reset = function() {
  this.curStep = 0;
  this.subAnimations = {};
  this._updateStatus(Animation.STATUS.READY);
};

Animation.prototype.start = function() {
  this._scheduleAnimation();
  this._updateStatus(Animation.STATUS.RUNNING_AUTO);
};

Animation.prototype.stop = function() {
  this._reset("all");
  this._updateStatus(Animation.STATUS.STOPPED);
};

Animation.prototype.stepPrev = function() {
  this._step("prev");
};

Animation.prototype.stepNext = function() {
  this._step("next");
};

Animation.prototype._step = function(dir) {
  var backward = (dir === "prev");
  var self = this;
  var graph = self.xdsms[self.rootId].graph;
  var nodesByStep = graph.nodesByStep;
  var incr = backward ? -1 : 1;

  // console.log("*************************************** STEP "+self.rootId);

  if ((!backward && self.done()) ||
      (backward && self.ready())) {
    return;
  }

  if (!self._subAnimationInProgress()) {
    self.curStep += incr;
    self._reset();

    var nodesAtStep = nodesByStep[self.curStep];
    nodesAtStep.forEach(function(nodeId) {
      if (self.running()) {
        nodesByStep[self.curStep - incr].forEach(function(prevNodeId) { // eslint-disable-line space-infix-ops
          if (backward) {
            self._pulseLink(0, nodeId, prevNodeId);
          } else {
            self._pulseLink(0, prevNodeId, nodeId);
          }
          var gnode = "g.id" + prevNodeId;
          self._pulse(0, gnode + " > rect", "out");
        });
      }
      var gnode = "g.id" + nodeId;
      self._pulse(0, gnode + " > rect", "in");
    });
  }

  // console.log(self.rootId+" -> nodesByStep = "+JSON.stringify(nodesByStep));
  // console.log(self.rootId+" -> nodesAtStep = "+JSON.stringify(nodesAtStep));
  // console.log(self.rootId+" -> self.curStep = "+self.curStep);

  if (nodesByStep[self.curStep].some(self._isSubScenario, this)) {
    nodesByStep[self.curStep].forEach(function(nodeId) {
      if (self._isSubScenario(nodeId)) {
        var scnId = graph.getNode(nodeId).getScenarioId();
        var anim;
        if (self.subAnimations[scnId]) {
          anim = self.subAnimations[scnId];
        } else {
          anim = self.subAnimations[scnId] = new Animation(self.xdsms, scnId);
        }
        anim._step(dir);
      }
    }, this);
  }
  if (this.done()) {
    this._updateStatus(Animation.STATUS.DONE);
  } else if (this.ready()) {
    this._updateStatus(Animation.STATUS.READY);
  } else {
    this._updateStatus(Animation.STATUS.RUNNING_STEP);
  }
};

Animation.prototype.running = function() {
  return !this.ready() && !this.done();
};
Animation.prototype.ready = function() {
  return this.curStep === 0;
};
Animation.prototype.done = function() {
  return this.curStep === this.root.graph.nodesByStep.length - 1;
};
Animation.prototype.isStatus = function(status) {
  return this.status === status;
};

Animation.prototype.addObserver = function(observer) {
  if (observer) {
    this._observers.push(observer);
  }
};

Animation.prototype.renderNodeStatuses = function() {
  var self = this;
  var graph = self.xdsms[self.rootId].graph;
  graph.nodes.forEach(function(node) {
    var gnode = "g." + node.id;
    switch (node.status) {
      case Graph.NODE_STATUS.RUNNING:
        self._pulse(0, gnode + " > rect", "in", RUNNING_COLOR);
        break;
      case Graph.NODE_STATUS.FAILED:
        self._pulse(0, gnode + " > rect", "in", FAILED_COLOR);
        break;
      case Graph.NODE_STATUS.PENDING:
        self._pulse(0, gnode + " > rect", "in", PENDING_COLOR);
        break;
      case Graph.NODE_STATUS.DONE:
        self._pulse(0, gnode + " > rect", "in", DONE_COLOR);
        break;
      default:
        // nothing to do
    }
    if (self._isSubScenario(node.id)) {
      var scnId = graph.getNode(node.id).getScenarioId();
      var anim = new Animation(self.xdsms, scnId);

      anim.renderNodeStatuses();
    }
  });
};

Animation.prototype._updateStatus = function(status) {
  this.status = status;
  this._notifyObservers(status);
};

Animation.prototype._notifyObservers = function() {
  this._observers.map((function(obs) {
    obs.update(this.status);
  }).bind(this));
};

Animation.prototype._pulse = function(delay, toBeSelected, easeInOut, color) {
  if (color === undefined) {
    color = RUNNING_COLOR;
  }
  var sel = select("svg." + this.rootId)
      .selectAll(toBeSelected)
      .transition().delay(delay);
  if (easeInOut !== "out") {
    sel = sel.transition().duration(200)
        .style('stroke-width', '8px')
        .style('stroke', color)
        .style('fill', function(d) {
          if (d.id) {
            return color.brighter();
          }});
  }
  if (easeInOut !== "in") {
    sel.transition().duration(3 * PULSE_DURATION)
        .style('stroke-width', null)
        .style('stroke', null)
        .style('fill', null);
  }
};

Animation.prototype._pulseLink = function(delay, fromId, toId) {
  var graph = this.xdsms[this.rootId].graph;
  var from = graph.idxOf(fromId);
  var to = graph.idxOf(toId);
  this._pulse(delay, "path.link_" + from + "_" + to);
};

Animation.prototype._onAnimationStart = function(delay) {
  var title = select("svg." + this.rootId).select("g.title");
  title.select("text").transition().delay(delay).style("fill", RUNNING_COLOR);
  select("svg." + this.rootId).select("rect.border")
      .transition().delay(delay)
      .style("stroke-width", '5px').duration(200)
      .transition().duration(1000)
      .style("stroke", 'black').style("stroke-width", '0px');
};

Animation.prototype._onAnimationDone = function(delay) {
  var self = this;
  var title = select("svg." + this.rootId).select("g.title");
  title.select("text").transition()
      .delay(delay)
      .style("fill", null).on("end", function() {
        self._updateStatus(Animation.STATUS.DONE);
      });
};

Animation.prototype._isSubScenario = function(nodeId) {
  var gnode = "g.id" + nodeId;
  var nodeSel = select("svg." + this.rootId).select(gnode);
  return nodeSel.classed("mdo");
};

Animation.prototype._scheduleAnimation = function() {
  var self = this;
  var delay = this.initialDelay;
  var animDelay = SUB_ANIM_DELAY;
  var graph = self.xdsms[self.rootId].graph;

  self._onAnimationStart(delay);

  graph.nodesByStep.forEach(function(nodesAtStep, n, nodesByStep) {
    var offsets = [];
    nodesAtStep.forEach(function(nodeId) {
      var elapsed = delay + n * PULSE_DURATION;

      if (n > 0) {
        nodesByStep[n-1].forEach(function(prevNodeId) { // eslint-disable-line space-infix-ops
          self._pulseLink(elapsed, prevNodeId, nodeId);
        });

        var gnode = "g.id" + nodeId;
        if (self._isSubScenario(nodeId)) {
          self._pulse(elapsed, gnode + " > rect", "in");
          var scnId = graph.getNode(nodeId).getScenarioId();
          var anim = new Animation(self.xdsms, scnId, elapsed + animDelay);
          var offset = anim._scheduleAnimation();
          offsets.push(offset);
          self._pulse(offset + elapsed + animDelay, gnode + " > rect", "out");
        } else {
          self._pulse(elapsed, gnode + " > rect");
        }
      }
    }, this);

    if (offsets.length > 0) {
      delay += Math.max.apply(null, offsets);
    }
    delay += animDelay;
  }, this);

  self._onAnimationDone(graph.nodesByStep.length * PULSE_DURATION + delay);

  return graph.nodesByStep.length * PULSE_DURATION;
};

Animation.prototype._reset = function(all) {
  var svg = select("svg." + this.rootId);
  if (all) {
    svg = selectAll("svg");
  }
  svg.selectAll('rect').transition().duration(0)
      .style('stroke-width', null)
      .style('stroke', null);
  svg.selectAll('.title > text').transition().duration(0)
      .style('fill', null);

  svg.selectAll('.node > rect').transition().duration(0)
      .style('stroke-width', null)
      .style('stroke', null)
      .style('fill', null);
  svg.selectAll('path').transition().duration(0)
      .style('stroke-width', null)
      .style('stroke', null)
      .style('fill', null);
};

Animation.prototype._resetPreviousStep = function() {
  this.root.graph.nodesByStep[this.curStep - 1].forEach(function(nodeId) {
    var gnode = "g.id" + nodeId;
    this._pulse(0, gnode + " > rect", "out");
  }, this);
};

Animation.prototype._subAnimationInProgress = function() {
  var running = false;
  for (var k in this.subAnimations) {
    if (this.subAnimations.hasOwnProperty(k)) {
      running = running || this.subAnimations[k].running();
    }
  }
  return running;
};

export default Animation;
