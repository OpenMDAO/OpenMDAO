
import { select, selectAll } from 'd3-selection';
import { rgb } from 'd3-color';
import Graph from './graph';

const PULSE_DURATION = 700;
const SUB_ANIM_DELAY = 200;
const RUNNING_COLOR = rgb('seagreen');
const FAILED_COLOR = rgb('firebrick');
const PENDING_COLOR = rgb('darkseagreen');
const DONE_COLOR = rgb('darkcyan');

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

Animation.STATUS = {
  READY: 'ready',
  RUNNING_STEP: 'running_step',
  RUNNING_AUTO: 'running_auto',
  STOPPED: 'stopped',
  DONE: 'done',
  DISABLED: 'disabled',
};

Animation.prototype.reset = function reset() {
  this.curStep = 0;
  this.subAnimations = {};
  this._updateStatus(Animation.STATUS.READY);
};

Animation.prototype.start = function start() {
  this._scheduleAnimation();
  this._updateStatus(Animation.STATUS.RUNNING_AUTO);
};

Animation.prototype.stop = function stop() {
  this._reset('all');
  this._updateStatus(Animation.STATUS.STOPPED);
};

Animation.prototype.stepPrev = function stepPrev() {
  this._step('prev');
};

Animation.prototype.stepNext = function stepNext() {
  this._step('next');
};

Animation.prototype.setXdsmVersion = function setXdsmVersion(version) {
  Object.values(this.xdsms).forEach((xdsm) => {
    xdsm.setVersion(version);
    xdsm.refresh();
  }, this);
};

Animation.prototype._step = function _step(dir) {
  const backward = (dir === 'prev');
  const self = this;
  const { graph } = self.xdsms[self.rootId];
  const { nodesByStep } = graph;
  const incr = backward ? -1 : 1;

  // console.log("*************************************** STEP "+self.rootId);

  if ((!backward && self.done())
    || (backward && self.ready())) {
    return;
  }

  if (!self._subAnimationInProgress()) {
    self.curStep += incr;
    self._reset();

    const nodesAtStep = nodesByStep[self.curStep];
    nodesAtStep.forEach((nodeId) => {
      if (self.running()) {
        nodesByStep[self.curStep - incr].forEach((prevNodeId) => {
          if (backward) {
            self._pulseLink(0, nodeId, prevNodeId);
          } else {
            self._pulseLink(0, prevNodeId, nodeId);
          }
          const gnode = `g.id${prevNodeId}`;
          self._pulseNode(0, gnode, 'out');
        });
      }
      const gnode = `g.id${nodeId}`;
      self._pulseNode(0, gnode, 'in');
    });
  }

  if (nodesByStep[self.curStep].some(self._isSubXdsm, this)) {
    nodesByStep[self.curStep].forEach((nodeId) => {
      if (self._isSubXdsm(nodeId)) {
        const xdsmId = graph.getNode(nodeId).getSubXdsmId();
        if (!self.subAnimations[xdsmId]) {
          self.subAnimations[xdsmId] = new Animation(self.xdsms, xdsmId);
        }
        const anim = self.subAnimations[xdsmId];
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

Animation.prototype.running = function running() {
  return !this.ready() && !this.done();
};
Animation.prototype.ready = function ready() {
  return this.curStep === 0;
};
Animation.prototype.done = function done() {
  return this.curStep === this.root.graph.nodesByStep.length - 1;
};
Animation.prototype.isStatus = function isStatus(status) {
  return this.status === status;
};

Animation.prototype.addObserver = function addObserver(observer) {
  if (observer) {
    this._observers.push(observer);
  }
};

Animation.prototype.renderNodeStatuses = function renderNodeStatuses() {
  const self = this;
  const { graph } = self.xdsms[self.rootId];
  graph.nodes.forEach((node) => {
    const gnode = `g.${node.id}`;
    switch (node.status) {
      case Graph.NODE_STATUS.RUNNING:
        self._pulseNode(0, gnode, 'in', RUNNING_COLOR);
        break;
      case Graph.NODE_STATUS.FAILED:
        self._pulseNode(0, gnode, 'in', FAILED_COLOR);
        break;
      case Graph.NODE_STATUS.PENDING:
        self._pulseNode(0, gnode, 'in', PENDING_COLOR);
        break;
      case Graph.NODE_STATUS.DONE:
        self._pulseNode(0, gnode, 'in', DONE_COLOR);
        break;
      default:
      // nothing to do
    }
    if (self._isSubXdsm(node.id)) {
      const xdsmId = graph.getNode(node.id).getSubXdsmId();
      const anim = new Animation(self.xdsms, xdsmId);

      anim.renderNodeStatuses();
    }
  });
};

Animation.prototype._updateStatus = function _updateStatus(status) {
  this.status = status;
  this._notifyObservers(status);
};

Animation.prototype._notifyObservers = function _notifyObservers() {
  this._observers.map((obs) => obs.update(this.status));
};

Animation.prototype._pulse = function _pulse(delay, toBeSelected, easeInOut, color) {
  const colour = color || RUNNING_COLOR;
  let sel = select(`svg.${this.rootId}`)
    .selectAll(toBeSelected)
    .transition().delay(delay);
  if (easeInOut !== 'out') {
    sel = sel.transition().duration(200)
      .style('stroke-width', '8px')
      .style('stroke', colour)
      .style('fill', (d) => {
        if (d.id) {
          return colour.brighter();
        }
        return null;
      });
  }
  if (easeInOut !== 'in') {
    sel.transition().duration(3 * PULSE_DURATION)
      .style('stroke-width', null)
      .style('stroke', null)
      .style('fill', null);
  }
};

Animation.prototype._pulseNode = function _pulseNode(delay, gnode, easeInOut, color) {
  this._pulse(delay, `${gnode} > rect`, easeInOut, color);
  this._pulse(delay, `${gnode} > polygon`, easeInOut, color);
};

Animation.prototype._pulseLink = function _pulseLink(delay, fromId, toId) {
  const { graph } = this.xdsms[this.rootId];
  const from = graph.idxOf(fromId);
  const to = graph.idxOf(toId);
  this._pulse(delay, `path.link_${from}_${to}`);
};

Animation.prototype._onAnimationStart = function _onAnimationStart(delay) {
  const title = select(`svg.${this.rootId}`).select('g.title');
  title.select('text').transition().delay(delay).style('fill', RUNNING_COLOR);
  select(`svg.${this.rootId}`).select('rect.border')
    .transition().delay(delay)
    .style('stroke-width', '5px')
    .duration(200)
    .transition()
    .duration(1000)
    .style('stroke', 'black')
    .style('stroke-width', '0px');
};

Animation.prototype._onAnimationDone = function _onAnimationDone(delay) {
  const self = this;
  const title = select(`svg.${this.rootId}`).select('g.title');
  title.select('text').transition()
    .delay(delay)
    .style('fill', null)
    .on('end', () => {
      self._updateStatus(Animation.STATUS.DONE);
    });
};

Animation.prototype._isSubXdsm = function _isSubXdsm(nodeId) {
  const gnode = `g.id${nodeId}`;
  const nodeSel = select(`svg.${this.rootId}`).select(gnode);
  return nodeSel.classed('mdo') || nodeSel.classed('sub-optimization')
    || nodeSel.classed('group') || nodeSel.classed('implicit-group');
};

Animation.prototype._scheduleAnimation = function _scheduleAnimation() {
  const self = this;
  let delay = this.initialDelay;
  const animDelay = SUB_ANIM_DELAY;
  const { graph } = self.xdsms[self.rootId];

  self._onAnimationStart(delay);

  graph.nodesByStep.forEach((nodesAtStep, n, nodesByStep) => {
    const offsets = [];
    nodesAtStep.forEach((nodeId) => {
      const elapsed = delay + n * PULSE_DURATION;

      if (n > 0) {
        nodesByStep[n - 1].forEach((prevNodeId) => { // eslint-disable-line space-infix-ops
          self._pulseLink(elapsed, prevNodeId, nodeId);
        });

        const gnode = `g.id${nodeId}`;
        if (self._isSubXdsm(nodeId)) {
          self._pulseNode(elapsed, gnode, 'in');
          const xdsmId = graph.getNode(nodeId).getSubXdsmId();
          const anim = new Animation(self.xdsms, xdsmId, elapsed + animDelay);
          const offset = anim._scheduleAnimation();
          offsets.push(offset);
          self._pulseNode(offset + elapsed + animDelay, gnode, 'out');
        } else {
          self._pulseNode(elapsed, gnode);
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

Animation.prototype._reset = function _reset(all) {
  let svg = select(`svg.${this.rootId}`);
  if (all) {
    svg = selectAll('svg');
  }
  svg.selectAll('rect').transition().duration(0)
    .style('stroke-width', null)
    .style('stroke', null);
  svg.selectAll('polygon').transition().duration(0)
    .style('stroke-width', null)
    .style('stroke', null);
  svg.selectAll('.title > text').transition().duration(0)
    .style('fill', null);

  svg.selectAll('.node > rect').transition().duration(0)
    .style('stroke-width', null)
    .style('stroke', null)
    .style('fill', null);
  svg.selectAll('.node > polygon').transition().duration(0)
    .style('stroke-width', null)
    .style('stroke', null)
    .style('fill', null);
  svg.selectAll('path').transition().duration(0)
    .style('stroke-width', null)
    .style('stroke', null)
    .style('fill', null);
};

Animation.prototype._resetPreviousStep = function _resetPreviousStep() {
  this.root.graph.nodesByStep[this.curStep - 1].forEach((nodeId) => {
    const gnode = `g.id${nodeId}`;
    this._pulseNode(0, gnode, 'out');
  }, this);
};

Animation.prototype._subAnimationInProgress = function _subAnimationInProgress() {
  let running = false;
  for (const k in this.subAnimations) {
    if (Object.prototype.hasOwnProperty.call(this.subAnimations, k)) {
      running = running || this.subAnimations[k].running();
    }
  }
  return running;
};

export default Animation;
