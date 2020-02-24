
import { select, selectAll, event } from 'd3-selection';
import 'd3-transition';
import Labelizer from './labelizer';

export const VERSION1 = 'xdsm';
export const VERSION2 = 'xdsm2';

const WIDTH = 1000;
const HEIGHT = 500;
const X_ORIG = 100;
const Y_ORIG = 20;
const PADDING = 20;
const CELL_W = 250;
const CELL_H = 75;
const MULTI_OFFSET = 3;
const BORDER_PADDING = 4;
const ANIM_DURATION = 0; // ms
const TOOLTIP_WIDTH = 300;

function Cell(x, y, width, height) {
  this.x = x;
  this.y = y;
  this.width = width;
  this.height = height;
}

function Xdsm(graph, svgid, config) {
  this.graph = graph;
  this.version = config.version || VERSION2;

  const container = select(`.${this.version}`);
  this.svg = container.append('svg')
    .attr('width', WIDTH)
    .attr('height', HEIGHT)
    .attr('class', svgid);

  this.grid = [];
  this.nodes = [];
  this.edges = [];
  this.svgid = svgid;

  // TODO: Find better javascript way to do configuration.
  this.default_config = {
    labelizer: {
      ellipsis: 5,
      subSupScript: true,
      showLinkNbOnly: false,
    },
    layout: {
      origin: { x: X_ORIG, y: Y_ORIG },
      cellsize: { w: CELL_W, h: CELL_H },
      padding: PADDING,
    },
    titleTooltip: false, // allow to use external tooltip mechanism
  };

  this.config = this.default_config;
  if (config && config.labelizer) {
    this.config.labelizer.ellipsis = config.labelizer.ellipsis;
    this.config.labelizer.subSupScript = config.labelizer.subSupScript;
    this.config.labelizer.showLinkNbOnly = config.labelizer.showLinkNbOnly;
  }
  if (config && config.layout) {
    this.config.layout.origin.x = config.layout.origin.x;
    this.config.layout.origin.y = config.layout.origin.y;
    this.config.layout.cellsize.w = config.layout.cellsize.w;
    this.config.layout.cellsize.h = config.layout.cellsize.h;
    this.config.layout.padding = config.layout.padding;
  }
  this.config.titleTooltip = config.titleTooltip;

  // Xdsm built-in tooltip for variable connexions
  if (!this.config.titleTooltip) {
    this.tooltip = select('body').append('div').attr('class', 'xdsm-tooltip')
      .style('opacity', 0);
  }
  this._initialize();
}

Xdsm.prototype.setVersion = function setVersion(version) {
  this.version = version;
};

Xdsm.prototype.addNode = function addNode(nodeName) {
  this.graph.addNode(nodeName);
  this.draw();
};

Xdsm.prototype.removeNode = function removeNode(index) {
  this.graph.removeNode(index);
  this.draw();
};

Xdsm.prototype.hasWorkflow = function hasWorkflow() {
  return this.graph.chains.length !== 0;
};

Xdsm.prototype._initialize = function _initialize() {
  const self = this;

  if (self.graph.refname) {
    self._createTitle();
  }
  self.nodeGroup = self.svg.append('g').attr('class', 'nodes');
  self.edgeGroup = self.svg.append('g').attr('class', 'edges');
};

Xdsm.prototype.refresh = function refresh() {
  const self = this;
  self.svg.selectAll('g').remove();
  self.nodeGroup = self.svg.append('g').attr('class', 'nodes');
  self.edgeGroup = self.svg.append('g').attr('class', 'edges');
  self.draw();
};

Xdsm.prototype.draw = function draw() {
  const self = this;

  self.nodes = self._createTextGroup('node', self.nodeGroup, self._customRect);
  self.edges = self._createTextGroup('edge', self.edgeGroup, self._customTrapz);

  // Workflow
  self._createWorkflow();

  // Dataflow
  self._createDataflow();

  // Border (used by animation)
  self._createBorder();

  // update size
  const w = self.config.layout.cellsize.w * (self.graph.nodes.length + 1);
  const h = self.config.layout.cellsize.h * (self.graph.nodes.length + 1);
  self.svg.attr('width', w).attr('height', h);
  self.svg.selectAll('.border')
    .attr('height', h - BORDER_PADDING)
    .attr('width', w - BORDER_PADDING);
};

Xdsm.prototype._createTextGroup = function _createTextGroup(kind, group, decorate) {
  const self = this;

  const selection = group.selectAll(`.${kind}`)
    .data(this.graph[`${kind}s`], // DATA JOIN
      (d) => d.id);

  const labelize = Labelizer.labelize()
    .labelKind(kind)
    .ellipsis(self.config.labelizer.ellipsis)
    .subSupScript(self.config.labelizer.subSupScript)
    .linkNbOnly(self.config.labelizer.showLinkNbOnly);

  const textGroups = selection
    .enter() // ENTER
    .append('g').attr('class', (d) => {
      let klass = kind === 'node' ? d.type : 'dataInter';
      if (klass === 'dataInter' && (d.row === 0 || d.col === 0)) {
        klass = 'dataIO';
      }
      return `id${d.id} ${kind} ${klass}`;
    }).each(function makeLabel(/* d, i */) {
      const that = select(this); // eslint-disable-line no-invalid-this
      that.call(labelize); // eslint-disable-line no-invalid-this
    })
    .each(function makeLine(d1, i) {
      const { grid } = self;
      const item = select(this); // eslint-disable-line no-invalid-this
      if (grid[i] === undefined) {
        grid[i] = new Array(self.graph.nodes.length);
      }
      item.select('text').each(function makeCell(d2, j) {
        const that = select(this); // eslint-disable-line no-invalid-this
        const data = item.data()[0];
        const m = (data.row === undefined) ? i : data.row;
        const n = (data.col === undefined) ? i : data.col;
        const bbox = that.nodes()[j].getBBox();
        grid[m][n] = new Cell(-bbox.width / 2, 0, bbox.width, bbox.height);
        that
          .attr('width', () => grid[m][n].width)
          .attr('height', () => grid[m][n].height)
          .attr('x', () => grid[m][n].x)
          .attr('y', () => grid[m][n].y);
      });
    })
    .each(function makeDecoration(d, i) {
      const that = select(this); // eslint-disable-line no-invalid-this
      that.call(decorate.bind(self), d, i, 0);
      if (d.isMulti) {
        that.call(decorate.bind(self), d, i, 1 * Number(MULTI_OFFSET));
        that.call(decorate.bind(self), d, i, 2 * Number(MULTI_OFFSET));
      }
    })
    .merge(selection); // UPDATE + ENTER

  selection.exit().remove(); // EXIT

  if (self.tooltip) {
    selectAll('.ellipsized').on('mouseover', (d) => {
      self.tooltip.transition().duration(200).style('opacity', 0.9);
      const tooltipize = Labelizer.tooltipize()
        .subSupScript(self.config.labelizer.subSupScript)
        .text(d.name);
      self.tooltip.call(tooltipize)
        .style('width', `${TOOLTIP_WIDTH}px`)
        .style('left', `${event.pageX}px`)
        .style('top', `${event.pageY - 28}px`);
    }).on('mouseout', () => {
      self.tooltip.transition().duration(500).style('opacity', 0);
    });
  } else {
    selectAll('.ellipsized')
      .attr('title', (d) => d.name.split(',').join(', '));
  }
  self._layoutText(textGroups, decorate, selection.empty() ? 0 : ANIM_DURATION);
};

Xdsm.prototype._layoutText = function _layoutText(items, decorate, delay) {
  const self = this;
  items.transition().duration(delay).attr('transform', (d, i) => {
    const m = (d.col === undefined) ? i : d.col;
    const n = (d.row === undefined) ? i : d.row;
    const w = self.config.layout.cellsize.w * m + self.config.layout.origin.x;
    const h = self.config.layout.cellsize.h * n + self.config.layout.origin.y;
    return `translate(${self.config.layout.origin.x + w},${self.config.layout.origin.y + h})`;
  });
};

Xdsm.prototype._createWorkflow = function _createWorkflow() {
  const self = this;
  const workflow = this.svg.selectAll('.workflow')
    .data([self.graph])
    .enter()
    .insert('g', ':first-child')
    .attr('class', 'workflow');

  workflow.selectAll('g')
    .data(self.graph.chains)
    .enter()
    .insert('g')
    .attr('class', 'workflow-chain')
    .selectAll('path')
    .data((d) => d)
    .enter()
    .append('path')
    .attr('class', (d) => `link_${d[0]}_${d[1]}`)
    .attr('transform', (d) => {
      const max = Math.max(d[0], d[1]);
      const min = Math.min(d[0], d[1]);
      let w;
      let h;
      if (d[0] < d[1]) {
        w = self.config.layout.cellsize.w * max + self.config.layout.origin.x;
        h = self.config.layout.cellsize.h * min + self.config.layout.origin.y;
      } else {
        w = self.config.layout.cellsize.w * min + self.config.layout.origin.x;
        h = self.config.layout.cellsize.h * max + self.config.layout.origin.y;
      }
      return `translate(${self.config.layout.origin.x + w},${self.config.layout.origin.y + h})`;
    })
    .attr('d', (d) => {
      const w = self.config.layout.cellsize.w * Math.abs(d[0] - d[1]);
      const h = self.config.layout.cellsize.h * Math.abs(d[0] - d[1]);
      const points = [];
      if (d[0] < d[1]) {
        if (d[0] !== 0) {
          points.push(`${-w},0`);
        }
        points.push('0,0');
        if (d[1] !== 0) {
          points.push(`0,${h}`);
        }
      } else {
        if (d[0] !== 0) {
          points.push(`${w},0`);
        }
        points.push('0,0');
        if (d[1] !== 0) {
          points.push(`0,${-h}`);
        }
      }
      return `M${points.join(' ')}`;
    });
};

Xdsm.prototype._createDataflow = function _createDataflow() {
  const self = this;
  self.svg.selectAll('.dataflow')
    .data([self])
    .enter()
    .insert('g', ':first-child')
    .attr('class', 'dataflow');

  const selection = self.svg.select('.dataflow').selectAll('path')
    .data(self.graph.edges, (d) => d.id);

  selection.enter()
    .append('path')
    .merge(selection)
    .transition()
    .duration(selection.empty() ? 0 : ANIM_DURATION)
    .attr('transform', (d, i) => {
      const m = (d.col === undefined) ? i : d.col;
      const n = (d.row === undefined) ? i : d.row;
      const w = self.config.layout.cellsize.w * m + self.config.layout.origin.x;
      const h = self.config.layout.cellsize.h * n + self.config.layout.origin.y;
      return `translate(${self.config.layout.origin.x + w},${self.config.layout.origin.y + h})`;
    })
    .attr('d', (d) => {
      const w = self.config.layout.cellsize.w * Math.abs(d.col - d.row);
      const h = self.config.layout.cellsize.h * Math.abs(d.col - d.row);
      const points = [];
      if (d.iotype === 'in') {
        if (d.row !== 0) {
          points.push(`${-w},0`);
        }
        points.push('0,0');
        if (d.col !== 0) {
          points.push(`0,${h}`);
        }
      } else {
        if (d.row !== 0) {
          points.push(`${w},0`);
        }
        points.push('0,0');
        if (d.col !== 0) {
          points.push(`0,${-h}`);
        }
      }
      return `M${points.join(' ')}`;
    });
  selection.exit().remove();
};

Xdsm.prototype._customRect = function _customRect(node, d, i, offset) {
  const self = this;
  const { grid } = self;
  if (this.version === VERSION2
    && (d.type === 'group'
      || d.type === 'implicit-group'
      || d.type === 'sub-optimization'
      || d.type === 'mdo')) {
    const x0 = grid[i][i].x + offset - self.config.layout.padding;
    const y0 = -grid[i][i].height * (2 / 3) - self.config.layout.padding - offset;
    const x1 = grid[i][i].x + offset + self.config.layout.padding + grid[i][i].width;
    const y1 = -grid[i][i].height * (2 / 3) + self.config.layout.padding
      - offset + grid[i][i].height;
    const ch = 10;
    const points = `${x0 + ch},${y0} ${x1 - ch},${y0} ${x1},${y0 + ch} ${x1},${y1 - ch} ${x1 - ch},${y1} ${x0 + ch},${y1} ${x0},${y1 - ch} ${x0},${y0 + ch}`;
    node.insert('polygon', ':first-child')
      .classed('shape', true)
      .attr('points', points);
  } else {
    node.insert('rect', ':first-child')
      .classed('shape', true)
      .attr('x', () => grid[i][i].x + offset - self.config.layout.padding)
      .attr('y', () => -grid[i][i].height * (2 / 3) - self.config.layout.padding - offset)
      .attr('width', () => grid[i][i].width + (self.config.layout.padding * 2))
      .attr('height', () => grid[i][i].height + (self.config.layout.padding * 2))
      .attr('rx', () => {
        const rounded = d.type === 'driver'
          || d.type === 'optimization'
          || d.type === 'mda'
          || d.type === 'doe';
        return rounded ? (grid[i][i].height + (self.config.layout.padding * 2)) / 2 : 0;
      })
      .attr('ry', () => {
        const rounded = d.type === 'driver'
          || d.type === 'optimization'
          || d.type === 'mda'
          || d.type === 'doe';
        return rounded ? (grid[i][i].height + (self.config.layout.padding * 2)) / 2 : 0;
      });
  }
};

Xdsm.prototype._customTrapz = function _customTrapz(edge, dat, i, offset) {
  const { grid } = this;
  edge.insert('polygon', ':first-child')
    .classed('shape', true)
    .attr('points', (d) => {
      const pad = 5;
      const w = grid[d.row][d.col].width;
      const h = grid[d.row][d.col].height;
      const topleft = `${-pad - w / 2 + offset
      }, ${
        -pad - h * (2 / 3) - offset
      } `;
      const topright = `${w / 2 + pad + offset + 5}, ${
        -pad - h * (2 / 3) - offset
      } `;
      const botright = `${w / 2 + pad + offset - 5 + 5}, ${
        pad + h / 3 - offset
      } `;
      const botleft = `${-pad - w / 2 + offset - 5}, ${
        pad + h / 3 - offset
      } `;
      const tpz = [topleft, topright, botright, botleft].join(' ');
      return tpz;
    });
};

Xdsm.prototype._createTitle = function _createTitle() {
  const self = this;
  // do not display title if it is 'root'
  const ref = self.svg.selectAll('.title')
    .data([self.graph.refname])
    .enter()
    .append('g')
    .classed('title', true)
    .append('text')
    .text(self.graph.refname === 'root' ? '' : self.graph.refname);

  const bbox = ref.nodes()[0].getBBox();

  ref.insert('rect', 'text')
    .attr('x', bbox.x)
    .attr('y', bbox.y)
    .attr('width', bbox.width)
    .attr('height', bbox.height);

  ref.attr('transform',
    `translate(${self.config.layout.origin.x}, ${self.config.layout.origin.y + bbox.height})`);
};

Xdsm.prototype._createBorder = function _createBorder() {
  const self = this;
  const bordercolor = 'black';
  self.svg.selectAll('.border')
    .data([self])
    .enter()
    .append('rect')
    .classed('border', true)
    .attr('x', BORDER_PADDING)
    .attr('y', BORDER_PADDING)
    .style('stroke', bordercolor)
    .style('fill', 'none')
    .style('stroke-width', 0);
};

export default Xdsm;
