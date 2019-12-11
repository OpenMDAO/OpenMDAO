/*
 * XDSMjs
 * Copyright 2016-2019 Rémi Lafage
 */
import { json } from 'd3-fetch';
import { select, selectAll, event } from 'd3-selection';
import Graph from './graph';
import Xdsm, { VERSION1, VERSION2 } from './xdsm';


import Animation from './animation';
import Controls from './controls';

class XdsmFactory {
  createXdsm(elt, version) {
    const mdostr = elt.attr('data-mdo');
    if (!mdostr) {
      const filename = elt.attr('data-mdo-file') || 'xdsm.json';
      json(filename).then((mdo) => this._createXdsm(mdo, version));
    } else {
      const mdo = JSON.parse(mdostr);
      this._createXdsm(mdo, version);
    }
  }

  _createXdsm(mdo, version) {
    const config = {
      labelizer: {
        ellipsis: 5,
        subSupScript: true,
        showLinkNbOnly: false,
      },
      version,
    };

    const scenarioKeys = Object.keys(mdo).sort();

    // Optimization problem display setup
    select('body').selectAll('optpb').data(scenarioKeys).enter()
      .append('div')
      .filter((d) => mdo[d].optpb)
      .attr('class', (d) => `optpb ${d}`)
      .style('opacity', 0)
      .on('click', function makeTransition() {
        select(this).transition().duration(500) // eslint-disable-line
          // no-invalid-this
          .style('opacity', 0)
          .style('pointer-events', 'none');
      })
      .append('pre')
      .html((d) => mdo[d].optpb);

    const xdsms = {};

    if (scenarioKeys.indexOf('root') === -1) {
      // old format: mono xdsm
      const graph = new Graph(mdo, 'root');
      xdsms.root = new Xdsm(graph, 'root', config);
      xdsms.root.draw();
    } else {
      // new format managing several XDSM
      scenarioKeys.forEach((k) => {
        if (Object.prototype.hasOwnProperty.call(mdo, k)) {
          const graph = new Graph(mdo[k], k);
          xdsms[k] = new Xdsm(graph, k, config);
          xdsms[k].draw();
          xdsms[k].svg.select('.optimization').on(
            'click',
            () => {
              const info = select(`.optpb.${k}`);
              info.style('opacity', 0.9);
              info.style('left', `${event.pageX}px`).style('top',
                `${event.pageY - 28}px`);
              info.style('pointer-events', 'auto');
            },
          );
        }
      }, this); // eslint-disable-line no-invalid-this
    }

    const anim = new Animation(xdsms);
    if (xdsms.root.hasWorkflow()) { // workflow is optional
      const ctrls = new Controls(anim, version); // eslint-disable-line no-unused-vars
    }
    anim.renderNodeStatuses();
  }
}

const XDSM_FACTORY = new XdsmFactory();

function detectVersion() {
  if (select(`.${VERSION1}`).empty()) {
    return VERSION2;
  }
  return VERSION1;
}

function createXDSM(version) {
  const elts = selectAll(`.${version}`);
  elts.each(function create(/* d, i */) {
    const elt = select(this);
    XDSM_FACTORY.createXdsm(elt, version);
  });
}

document.addEventListener('DOMContentLoaded',
  (/* event */) => {
    createXDSM(detectVersion());
  });
