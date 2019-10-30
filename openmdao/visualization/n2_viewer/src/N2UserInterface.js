class N2UserInterface {
    constructor(n2Diag) {
        this.n2Diag = n2Diag;

        this._setupCollapseDepthElement();
    }

    _setupCollapseDepthElement() {
        let collapseDepthElement =
            this.n2Diag.dom.parentDiv.querySelector("#idCollapseDepthDiv");

        for (let i = 2; i <= this.n2Diag.model.maxDepth; ++i) {
            let option = document.createElement("span");
            option.className = "fakeLink";
            option.id = "idCollapseDepthOption" + i + "";
            option.innerHTML = "" + i + "";

            let f = function (idx) {
                return function () {
                    CollapseToDepthSelectChange(idx);
                };
            }(i);
            option.onclick = f;
            collapseDepthElement.appendChild(option);
        }
    }

    collapse() {
        let d = lastLeftClickedEle;
        if (!d.hasChildren()) return;
        if (d.depth > this.n2Diag.zoomedElement.depth) { //dont allow minimizing on root node
            lastRightClickedElement = d;
            FindRootOfChangeFunction = FindRootOfChangeForRightClick;
            N2TransitionDefaults.duration = N2TransitionDefaults.durationFast;
            lastClickWasLeft = false;
            d.toggleMinimize();
            this.n2Diag.update();
        }
    }
}