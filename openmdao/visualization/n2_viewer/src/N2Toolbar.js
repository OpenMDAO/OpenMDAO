class N2Toolbar {
    constructor(sliderHeight = window.innerHeight * .95) {
        const self = this;
        this.modelHeightSlider = d3.select('#model-slider');
        this.toolbar = d3.select('#toolbarLoc');
        this.hideToolbarButton = d3.select('#hide-toolbar');
        this.hideToolbarIcon = this.hideToolbarButton.select('i');
        this.hidden = false;

        // Open expandable buttons when hovered over
        d3.selectAll('.expandable > div')
            .on('mouseover', function() {
                self.toolbar.style('z-index', '5');
                d3.select(this).style('max-width', '200px');
            })
            .on('mouseout', function() {
                d3.select(this).style('max-width', '0');
                self.toolbar.style('z-index', '1')
            })


        // When an expanded button is clicked, update the 'root' button
        // to the same icon and function.
        d3.selectAll('.toolbar-group-expandable > i')
            .on('click', function() {
                let container = d3.select(this.parentNode.parentNode);
                let button = d3.select(this);

                container.select('i')
                    .attr('class', button.attr('class'))
                    .attr('id', button.attr('id'))
                    .on('click', button.node.onclick);
            })

        this.hideToolbarButton.on('click', function () {
            if (self.hidden) { self.show(); }
            else { self.hide(); }
        })
    }

    /** Slide everything to the left offscreen 75px, rotate the button */
    hide() {
        this.toolbar.style('left', '-75px');
        this.hideToolbarButton.style('left', '-30px');
        this.hideToolbarIcon.style('transform', 'rotate(-180deg)');
        d3.select('#d3_content_div').style('margin-left', '-75px');
        this.hidden = true;
    }

    /** Slide everything to the right and rotate the button */
    show() {
        this.hideToolbarIcon.style('transform', 'rotate(0deg)');
        this.toolbar.style('left', '0px');
        this.hideToolbarButton.style('left', '45px');
        d3.select('#d3_content_div').style('margin-left', '0px');
        this.hidden = false;
    }
}