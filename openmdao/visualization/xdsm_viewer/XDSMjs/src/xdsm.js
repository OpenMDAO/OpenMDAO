'use strict';
import {select, selectAll, event} from 'd3-selection';
import 'd3-transition';
import Labelizer from './labelizer.js';

var WIDTH = 1000;
var HEIGHT = 500;
var X_ORIG = 100;
var Y_ORIG = 20;
var PADDING = 20;
var CELL_W = 250;
var CELL_H = 75;
var MULTI_OFFSET = 3;
var BORDER_PADDING = 4;
var ANIM_DURATION = 0; // ms
var TOOLTIP_WIDTH = 300;

function Cell(x, y, width, height) {
  this.x = x;
  this.y = y;
  this.width = width;
  this.height = height;
}

function Xdsm(graph, svgid, config) {
  this.graph = graph;
  var container = select(".xdsm");
  this.svg = container.append("svg")
      .attr("width", WIDTH)
      .attr("height", HEIGHT)
      .attr("class", svgid);

  this.grid = [];
  this.nodes = [];
  this.edges = [];

  // TODO: Find better javascript way to do configuration.
  this.default_config = {
    labelizer: {
      ellipsis: 5,
      subSupScript: true,
      showLinkNbOnly: false,
    },
    layout: {
      origin: {x: X_ORIG, y: Y_ORIG},
      cellsize: {w: CELL_W, h: CELL_H},
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
    this.tooltip = select("body").append("div").attr("class", "xdsm-tooltip")
        .style("opacity", 0);
  }
  this._initialize();
}

Xdsm.prototype.addNode = function(nodeName) {
  this.graph.addNode(nodeName);
  this.draw();
};

Xdsm.prototype.removeNode = function(index) {
  this.graph.removeNode(index);
  this.draw();
};

Xdsm.prototype.hasWorkflow = function() {
  return this.graph.chains.length !== 0;
};

Xdsm.prototype._initialize = function() {
  var self = this;

  if (self.graph.refname) {
    self._createTitle();
  }
  self.nodeGroup = self.svg.append('g').attr("class", "nodes");
  self.edgeGroup = self.svg.append('g').attr("class", "edges");
};

Xdsm.prototype.refresh = function() {
  var self = this;
  self.svg.selectAll("g").remove();
  self.nodeGroup = self.svg.append('g').attr("class", "nodes");
  self.edgeGroup = self.svg.append('g').attr("class", "edges");
  self.draw();
};

Xdsm.prototype.draw = function() {
  var self = this;

  self.nodes = self._createTextGroup("node", self.nodeGroup, self._customRect);
  self.edges = self._createTextGroup("edge", self.edgeGroup, self._customTrapz);

  // Workflow
  self._createWorkflow();

  // Dataflow
  self._createDataflow();

  // Border (used by animation)
  self._createBorder();

  // update size
  var w = self.config.layout.cellsize.w * (self.graph.nodes.length + 1);
  var h = self.config.layout.cellsize.h * (self.graph.nodes.length + 1);
  self.svg.attr("width", w).attr("height", h);
  self.svg.selectAll(".border")
      .attr("height", h - BORDER_PADDING)
      .attr("width", w - BORDER_PADDING);
};

Xdsm.prototype._createTextGroup = function(kind, group, decorate) {
  var self = this;

  var selection =
    group.selectAll("." + kind)
        .data(this.graph[kind + "s"], // DATA JOIN
            function(d) {return d.id;});

  var labelize = Labelizer.labelize()
      .labelKind(kind)
      .ellipsis(self.config.labelizer.ellipsis)
      .subSupScript(self.config.labelizer.subSupScript)
      .linkNbOnly(self.config.labelizer.showLinkNbOnly);

  var textGroups = selection
      .enter() // ENTER
      .append("g").attr("class", function(d) {
        var klass = kind === "node" ? d.type : "dataInter";
        if (klass === "dataInter" && (d.row === 0 || d.col === 0)) {
          klass = "dataIO";
        }
        return "id"+d.id + " " + kind + " " + klass;
      }).each(function(d, i) {
        var that = select(this); // eslint-disable-line no-invalid-this
        that.call(labelize); // eslint-disable-line no-invalid-this
      }).each(function(d, i) {
        var grid = self.grid;
        var item = select(this); // eslint-disable-line no-invalid-this
        if (grid[i] === undefined) {
          grid[i] = new Array(self.graph["nodes"].length);
        }
        item.select("text").each(function(d, j) {
          var that = select(this); // eslint-disable-line no-invalid-this
          var data = item.data()[0];
          var m = (data.row === undefined) ? i : data.row;
          var n = (data.col === undefined) ? i : data.col;
          var bbox = that.nodes()[j].getBBox();
          grid[m][n] = new Cell(-bbox.width / 2, 0, bbox.width, bbox.height);
          that
              .attr("width", function() {return grid[m][n].width;})
              .attr("height", function() {return grid[m][n].height;})
              .attr("x", function() {return grid[m][n].x;})
              .attr("y", function() {return grid[m][n].y;});
        });
      }).each(function(d, i) {
        var that = select(this); // eslint-disable-line no-invalid-this
        that.call(decorate.bind(self), d, i, 0);
        if (d.isMulti) {
          that.call(decorate.bind(self), d, i, 1 * Number(MULTI_OFFSET));
          that.call(decorate.bind(self), d, i, 2 * Number(MULTI_OFFSET));
        }
      })
      .merge(selection); // UPDATE + ENTER

  selection.exit().remove(); // EXIT

  if (self.tooltip) {
    selectAll(".ellipsized").on("mouseover", function(d) {
      self.tooltip.transition().duration(200).style("opacity", 0.9);
      var tooltipize = Labelizer.tooltipize()
          .subSupScript(self.config.labelizer.subSupScript)
          .text(d.name);
      self.tooltip.call(tooltipize)
          .style("width", TOOLTIP_WIDTH+"px")
          .style("left", (event.pageX) + "px")
          .style("top", (event.pageY - 28) + "px");
    }).on("mouseout", function() {
      self.tooltip.transition().duration(500).style("opacity", 0);
    });
  } else {
    selectAll(".ellipsized")
        .attr("title", function(d) {return d.name.split(',').join(', ');});
  }
  self._layoutText(textGroups, decorate, selection.empty() ? 0 : ANIM_DURATION);
};

Xdsm.prototype._layoutText = function(items, decorate, delay) {
  var self = this;
  items.transition().duration(delay).attr("transform", function(d, i) {
    var m = (d.col === undefined) ? i : d.col;
    var n = (d.row === undefined) ? i : d.row;
    var w = self.config.layout.cellsize.w * m + self.config.layout.origin.x;
    var h = self.config.layout.cellsize.h * n + self.config.layout.origin.y;
    return "translate(" + (self.config.layout.origin.x + w) + "," + (self.config.layout.origin.y + h) + ")";
  });
};

Xdsm.prototype._createWorkflow = function() {
  var self = this;
  var workflow = this.svg.selectAll(".workflow")
      .data([self.graph])
      .enter()
      .insert("g", ":first-child")
      .attr("class", "workflow");

  workflow.selectAll("g")
      .data(self.graph.chains)
      .enter()
      .insert('g').attr("class", "workflow-chain")
      .selectAll('path')
      .data(function(d) {return d;})
      .enter()
      .append("path")
      .attr("class", function(d) {
        return "link_" + d[0] + "_" + d[1];
      })
      .attr("transform", function(d) {
        var max = Math.max(d[0], d[1]);
        var min = Math.min(d[0], d[1]);
        var w;
        var h;
        if (d[0] < d[1]) {
          w = self.config.layout.cellsize.w * max + self.config.layout.origin.x;
          h = self.config.layout.cellsize.h * min + self.config.layout.origin.y;
        } else {
          w = self.config.layout.cellsize.w * min + self.config.layout.origin.x;
          h = self.config.layout.cellsize.h * max + self.config.layout.origin.y;
        }
        return "translate(" + (self.config.layout.origin.x + w) + "," + (self.config.layout.origin.y + h) + ")";
      })
      .attr("d", function(d) {
        var w = self.config.layout.cellsize.w * Math.abs(d[0] - d[1]);
        var h = self.config.layout.cellsize.h * Math.abs(d[0] - d[1]);
        var points = [];
        if (d[0] < d[1]) {
          if (d[0] !== 0) {
            points.push((-w) + ",0");
          }
          points.push("0,0");
          if (d[1] !== 0) {
            points.push("0," + h);
          }
        } else {
          if (d[0] !== 0) {
            points.push(w + ",0");
          }
          points.push("0,0");
          if (d[1] !== 0) {
            points.push("0," + (-h));
          }
        }
        return "M" + points.join(" ");
      });
};

Xdsm.prototype._createDataflow = function() {
  var self = this;
  self.svg.selectAll(".dataflow")
      .data([self])
      .enter()
      .insert("g", ":first-child")
      .attr("class", "dataflow");

  var selection =
    self.svg.select(".dataflow").selectAll("path")
        .data(self.graph.edges, function(d) {
          return d.id;
        });

  selection.enter()
      .append("path")
      .merge(selection)
      .transition().duration(selection.empty() ? 0 : ANIM_DURATION)
      .attr("transform", function(d, i) {
        var m = (d.col === undefined) ? i : d.col;
        var n = (d.row === undefined) ? i : d.row;
        var w = self.config.layout.cellsize.w * m + self.config.layout.origin.x;
        var h = self.config.layout.cellsize.h * n + self.config.layout.origin.y;
        return "translate(" + (self.config.layout.origin.x + w) + "," + (self.config.layout.origin.y + h) + ")";
      })
      .attr("d", function(d) {
        var w = self.config.layout.cellsize.w * Math.abs(d.col - d.row);
        var h = self.config.layout.cellsize.h * Math.abs(d.col - d.row);
        var points = [];
        if (d.iotype === "in") {
          if (d.row !== 0) {
            points.push((-w) + ",0");
          }
          points.push("0,0");
          if (d.col !== 0) {
            points.push("0," + h);
          }
        } else {
          if (d.row !== 0) {
            points.push(w + ",0");
          }
          points.push("0,0");
          if (d.col !== 0) {
            points.push("0," + (-h));
          }
        }
        return "M" + points.join(" ");
      });
  selection.exit().remove();
};

Xdsm.prototype._customRect = function(node, d, i, offset) {
  var self = this;
  var grid = self.grid;
  node.insert("rect", ":first-child")
      .classed("shape", true)
      .attr("x", function() {
        return grid[i][i].x + offset - self.config.layout.padding;
      })
      .attr("y", function() {
        return -grid[i][i].height * 2 / 3 - self.config.layout.padding - offset;
      })
      .attr("width", function() {
        return grid[i][i].width + (self.config.layout.padding * 2);
      })
      .attr("height", function() {
        return grid[i][i].height + (self.config.layout.padding * 2);
      })
      .attr("rx", function() {
        var rounded = d.type === 'driver' ||
                    d.type === 'optimization' ||
                    d.type === 'mda' ||
                    d.type === 'doe';
        return rounded ? (grid[i][i].height + (self.config.layout.padding * 2)) / 2 : 0;
      })
      .attr("ry", function() {
        var rounded = d.type === 'driver' ||
                    d.type === 'optimization' ||
                    d.type === 'mda' ||
                    d.type === 'doe';
        return rounded ? (grid[i][i].height + (self.config.layout.padding * 2)) / 2 : 0;
      });
};

Xdsm.prototype._customTrapz = function(edge, d, i, offset) {
  var grid = this.grid;
  edge.insert("polygon", ":first-child")
      .classed("shape", true)
      .attr("points", function(d) {
        var pad = 5;
        var w = grid[d.row][d.col].width;
        var h = grid[d.row][d.col].height;
        var topleft = (-pad - w / 2 + offset) + ", " +
                    (-pad - h * 2 / 3 - offset);
        var topright = (w / 2 + pad + offset + 5) + ", " +
                     (-pad - h * 2 / 3 - offset);
        var botright = (w / 2 + pad + offset - 5 + 5) + ", " +
                     (pad + h / 3 - offset);
        var botleft = (-pad - w / 2 + offset - 5) + ", " +
                    (pad + h / 3 - offset);
        var tpz = [topleft, topright, botright, botleft].join(" ");
        return tpz;
      });
};

Xdsm.prototype._createTitle = function() {
  var self = this;
  var ref = self.svg.selectAll(".title")
      .data([self.graph.refname])
      .enter()
      .append('g')
      .classed('title', true)
      .append("text").text(self.graph.refname);

  var bbox = ref.nodes()[0].getBBox();

  ref.insert("rect", "text")
      .attr('x', bbox.x)
      .attr('y', bbox.y)
      .attr('width', bbox.width)
      .attr('height', bbox.height);

  ref.attr('transform',
      'translate(' + self.config.layout.origin.x + ',' + (self.config.layout.origin.y + bbox.height) + ')');
};

Xdsm.prototype._createBorder = function() {
  var self = this;
  var bordercolor = 'black';
  self.svg.selectAll(".border")
      .data([self])
      .enter()
      .append("rect")
      .classed("border", true)
      .attr("x", BORDER_PADDING)
      .attr("y", BORDER_PADDING)
      .style("stroke", bordercolor)
      .style("fill", "none")
      .style("stroke-width", 0);
};

export default Xdsm;
