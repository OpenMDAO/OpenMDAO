'use strict';
import {select} from 'd3-selection';
import Animation from './animation';

function Controls(animation) {
  this.animation = animation;

  var buttonGroup = select(".xdsm-toolbar")
      .append("div")
      .classed("button_group", true);
  buttonGroup.append("button")
      .attr("id", "start")
      .append("i").attr("class", "icon-start");
  buttonGroup.append("button")
      .attr("id", "stop")
      .append("i").attr("class", "icon-stop");
  buttonGroup.append("button")
      .attr("id", "step-prev")
      .append("i").attr("class", "icon-step-prev");
  buttonGroup.append("button")
      .attr("id", "step-next")
      .append("i").attr("class", "icon-step-next");

  this.startButton = select('button#start');
  this.stopButton = select('button#stop');
  this.stepPrevButton = select('button#step-prev');
  this.stepNextButton = select('button#step-next');

  this.startButton.on('click', (function() {
    this.animation.start();
  }).bind(this));
  this.stopButton.on('click', (function() {
    this.animation.stop();
  }).bind(this));
  this.stepPrevButton.on('click', (function() {
    this.animation.stepPrev();
  }).bind(this));
  this.stepNextButton.on('click', (function() {
    this.animation.stepNext();
  }).bind(this));

  this.animation.addObserver(this);
  this.update(this.animation.status);
}

Controls.prototype.update = function(status) {
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
      console.log("Unexpected Event: " + status);
      break;
  }
};

Controls.prototype._enable = function(button) {
  button.attr("disabled", null);
};

Controls.prototype._disable = function(button) {
  button.attr("disabled", true);
};

export default Controls;
