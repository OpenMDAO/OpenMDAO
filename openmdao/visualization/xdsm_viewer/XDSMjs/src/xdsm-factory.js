/*
 * XDSMjs
 * Copyright 2016-2019 Rémi Lafage
 */
import {json} from 'd3-fetch';
import {select, selectAll, event} from 'd3-selection';
import Graph from './graph';
import Xdsm from './xdsm';
import Animation from './animation';
import Controls from './controls';

class XdsmFactory {
  createXdsm(elt) {
    const mdostr = elt.attr('data-mdo');
    if (!mdostr) {
      const filename = elt.attr('data-mdo-file') || 'xdsm.json';
      json(filename).then((mdo) => this._createXdsm(mdo) );
    } else {
      const mdo = JSON.parse(mdostr);
      this._createXdsm(mdo);
    }
  }

  _createXdsm(mdo) {
    const config = {
      labelizer: {
        ellipsis: 5,
        subSupScript: true,
        showLinkNbOnly: false,
      },
    };

    const scenarioKeys = Object.keys(mdo).sort();

    // Optimization problem display setup
    select("body").selectAll("optpb").data(scenarioKeys).enter().append("div")
        .filter(function(d) {
          return mdo[d].optpb;
        }).attr("class", function(d) {
          return "optpb " + d;
        }).style("opacity", 0).on("click", function() {
          select(this).transition().duration(500) // eslint-disable-line
          // no-invalid-this
              .style("opacity", 0).style("pointer-events", "none");
        }).append("pre").html(function(d) {
          return mdo[d].optpb;
        });

    var xdsms = {};

    if (scenarioKeys.indexOf('root') === -1) {
      // old format: mono xdsm
      var graph = new Graph(mdo);
      xdsms.root = new Xdsm(graph, 'root', config);
      xdsms.root.draw();
    } else {
      // new format managing several XDSM
      scenarioKeys.forEach(function(k) {
        if (mdo.hasOwnProperty(k)) {
          var graph = new Graph(mdo[k], k);
          xdsms[k] = new Xdsm(graph, k, config);
          xdsms[k].draw();
          xdsms[k].svg.select(".optimization").on(
              "click",
              function() {
                var info = select(".optpb." + k);
                info.style("opacity", 0.9);
                info.style("left", (event.pageX) + "px").style("top",
                    (event.pageY - 28) + "px");
                info.style("pointer-events", "auto");
              });
        }
      }, this); // eslint-disable-line no-invalid-this
    }

    var anim = new Animation(xdsms);
    if (xdsms.root.hasWorkflow()) { // workflow is optional
      var ctrls = new Controls(anim); // eslint-disable-line no-unused-vars
    }
    anim.renderNodeStatuses();
  }
}

const XDSM_FACTORY = new XdsmFactory();

document.addEventListener("DOMContentLoaded",
    (event) => {
      const elts = selectAll(".xdsm");
      elts.each(function(d, i) {
        const elt = select(this); // eslint-disable-line no-invalid-this
        XDSM_FACTORY.createXdsm(elt);
      });
    }
);
