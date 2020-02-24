
function Labelizer() { }

Labelizer.strParse = function strParse(str, subSupScript) {
  if (str === '') {
    return [{ base: '', sub: undefined, sup: undefined }];
  }
  const lstr = str.split(',');
  if (subSupScript === false) {
    return lstr.map((s) => ({ base: s, sub: undefined, sup: undefined }));
  }

  const underscores = /_/g;
  const rg = /([0-9-]+:)?([&#;A-Za-z0-9-.]+)(_[&#;A-Za-z0-9-._]+)?(\^.+)?/;

  const res = lstr.map((s) => {
    let base;
    let sub;
    let sup;

    if ((s.match(underscores) || []).length > 1) {
      const mu = s.match(/(.+)\^(.+)/);
      if (mu) {
        return { base: mu[1], sub: undefined, sup: mu[2] };
      }
      return { base: s, sub: undefined, sup: undefined };
    }
    const m = s.match(rg);
    if (m) {
      base = (m[1] ? m[1] : '') + m[2];
      if (m[3]) {
        sub = m[3].substring(1);
      }
      if (m[4]) {
        sup = m[4].substring(1);
      }
    } else {
      throw new Error(`Labelizer.strParse: Can not parse '${s}'`);
    }
    return { base, sub, sup };
  }, this);

  return res;
};

Labelizer._createVarListLabel = function _createVarListLabel(
  selection, name, text, ellipsis, subSupScript,
) {
  const tokens = Labelizer.strParse(name, subSupScript);

  tokens.every((token, i, ary) => {
    let offsetSub = 0;
    let offsetSup = 0;
    if (ellipsis < 1 || (i < 5 && text.nodes()[0].getBBox().width < 100)) {
      text.append('tspan').html(token.base);
      if (token.sub) {
        offsetSub = 10;
        text.append('tspan')
          .attr('class', 'sub')
          .attr('dy', offsetSub)
          .html(token.sub);
      }
      if (token.sup) {
        offsetSup = -10;
        text.append('tspan')
          .attr('class', 'sup')
          .attr('dx', -5)
          .attr('dy', -offsetSub + offsetSup)
          .html(token.sup);
        offsetSub = 0;
      }
    } else {
      text.append('tspan')
        .attr('dy', -offsetSub - offsetSup)
        .html('...');
      selection.classed('ellipsized', true);
      return false;
    }
    if (i < ary.length - 1) {
      text.append('tspan')
        .attr('dy', -offsetSub - offsetSup)
        .html(', ');
    }
    return true;
  }, this);
};

Labelizer._createLinkNbLabel = function _createLinkNbLabel(selection, name, text) {
  const lstr = name.split(',');
  let str = `${lstr.length} var`;
  if (lstr.length > 1) {
    str += 's';
  }
  text.append('tspan').html(str);
  selection.classed('ellipsized', true); // activate tooltip
};

Labelizer.labelize = function labelize() {
  let ellipsis = 0;
  let subSupScript = true;
  let linkNbOnly = false;
  let labelKind = 'node';

  function createLabel(selection) {
    selection.each((d) => {
      const text = selection.append('text');
      if (linkNbOnly && labelKind !== 'node') { // show connexion nb
        Labelizer._createLinkNbLabel(selection, d.name, text);
      } else {
        Labelizer._createVarListLabel(selection, d.name, text, ellipsis, subSupScript);
      }
    });
  }

  createLabel.ellipsis = function ellips(value) {
    if (!arguments.length) {
      return ellipsis;
    }
    ellipsis = value;
    return createLabel;
  };

  createLabel.subSupScript = function subsupscript(value) {
    if (!arguments.length) {
      return subSupScript;
    }
    subSupScript = value;
    return createLabel;
  };

  createLabel.linkNbOnly = function linknbonly(value) {
    if (!arguments.length) {
      return linkNbOnly;
    }
    linkNbOnly = value;
    return createLabel;
  };

  createLabel.labelKind = function labelkind(value) {
    if (!arguments.length) {
      return labelKind;
    }
    labelKind = value;
    return createLabel;
  };

  return createLabel;
};

Labelizer.tooltipize = function tooltipz() {
  let text = '';
  let subSupScript = false;

  function createTooltip(selection) {
    let html = [];
    if (subSupScript) {
      const tokens = Labelizer.strParse(text);
      tokens.forEach((token) => {
        let item = token.base;
        if (token.sub) {
          item += `<sub>${token.sub}</sub>`;
        }
        if (token.sup) {
          item += `<sup>${token.sup}</sup>`;
        }
        html.push(item);
      });
    } else {
      html = text.split(',');
    }
    selection.html(html.join(', '));
  }

  createTooltip.text = function txt(value) {
    if (!arguments.length) {
      return text;
    }
    text = value;
    return createTooltip;
  };

  createTooltip.subSupScript = function supsub(value) {
    if (!arguments.length) {
      return subSupScript;
    }
    subSupScript = value;
    return createTooltip;
  };

  return createTooltip;
};

export default Labelizer;
