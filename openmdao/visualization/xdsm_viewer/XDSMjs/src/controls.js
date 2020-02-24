
import { select } from 'd3-selection';
import Animation from './animation';
import { VERSION1, VERSION2 } from './xdsm';

function Controls(animation, defaultVersion) {
  this.animation = animation;
  this.defaultVersion = defaultVersion || VERSION2;

  const buttonGroup = select('.xdsm-toolbar')
    .append('div')
    .classed('button_group', true);
  buttonGroup.append('button')
    .attr('id', 'start')
    .append('i').attr('class', 'icon-start');
  buttonGroup.append('button')
    .attr('id', 'stop')
    .append('i').attr('class', 'icon-stop');
  buttonGroup.append('button')
    .attr('id', 'step-prev')
    .append('i').attr('class', 'icon-step-prev');
  buttonGroup.append('button')
    .attr('id', 'step-next')
    .append('i').attr('class', 'icon-step-next');
  buttonGroup.append('label')
    .text('XDSM')
    .attr('id', 'xdsm-version-label');
  buttonGroup.append('select')
    .attr('id', 'xdsm-version-toggle');


  this.startButton = select('button#start');
  this.stopButton = select('button#stop');
  this.stepPrevButton = select('button#step-prev');
  this.stepNextButton = select('button#step-next');
  this.toggleVersionButton = select('select#xdsm-version-toggle');
  const versions = ['v1', 'v2'];
  const versionsMap = { v1: VERSION1, v2: VERSION2 };
  this.toggleVersionButton
    .selectAll('versions')
    .data(versions)
    .enter()
    .append('option')
    .text((d) => d)
    .attr('value', (d) => versionsMap[d])
    .property('selected', (d) => defaultVersion === versionsMap[d]);

  this.startButton.on('click', () => {
    this.animation.start();
  });
  this.stopButton.on('click', () => {
    this.animation.stop();
  });
  this.stepPrevButton.on('click', () => {
    this.animation.stepPrev();
  });
  this.stepNextButton.on('click', () => {
    this.animation.stepNext();
  });
  this.toggleVersionButton.on('change', () => {
    const selectVersion = select('select#xdsm-version-toggle').property('value');
    let xdsm = select(`.${selectVersion}`);
    if (xdsm.empty() && selectVersion === VERSION1) {
      xdsm = select(`.${VERSION2}`);
      xdsm.classed(VERSION2, false)
        .classed(VERSION1, true);
    }
    if (xdsm.empty() && selectVersion === VERSION2) {
      xdsm = select(`.${VERSION1}`);
      xdsm.classed(VERSION1, false)
        .classed(VERSION2, true);
    }
    this.animation.setXdsmVersion(selectVersion);
  });

  this.animation.addObserver(this);
  this.update(this.animation.status);
}

Controls.prototype.update = function update(status) {
  // console.log("Controls receives: "+status);
  switch (status) {
    case Animation.STATUS.STOPPED:
    case Animation.STATUS.DONE:
      this.animation.reset(); // trigger READY status
    case Animation.STATUS.READY: // eslint-disable-line no-fallthrough
      this._enable(this.startButton);
      this._disable(this.stopButton);
      this._enable(this.stepNextButton);
      this._enable(this.stepPrevButton);
      break;
    case Animation.STATUS.RUNNING_AUTO:
      this._disable(this.startButton);
      this._enable(this.stopButton);
      this._disable(this.stepNextButton);
      this._disable(this.stepPrevButton);
      break;
    case Animation.STATUS.RUNNING_STEP:
      this._disable(this.startButton);
      this._enable(this.stopButton);
      this._enable(this.stepNextButton);
      this._enable(this.stepPrevButton);
      break;
    default:
      console.log(`Unexpected Event: ${status}`);
      break;
  }
};

Controls.prototype._enable = function _enable(button) {
  button.attr('disabled', null);
};

Controls.prototype._disable = function _disable(button) {
  button.attr('disabled', true);
};

export default Controls;
