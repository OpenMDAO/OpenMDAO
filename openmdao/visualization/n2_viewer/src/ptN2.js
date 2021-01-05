let sharedTransition = null;

var enterIndex = 0;
var exitIndex = 0;

/*
const content = '<p style="margin: 10px; padding: 10px; border: 25px solid red; ">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed justo mauris, porttitor sed nibh non, interdum aliquet tellus. Duis eget est lectus. In ultrices finibus semper. Nullam dictum, tortor non placerat convallis, nibh diam sagittis risus, non ultricies diam nisi nec neque. Phasellus dapibus convallis metus. Proin cursus, metus quis ullamcorper suscipit, neque mauris dictum ex, a mattis velit lorem ac erat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. Praesent a ligula ut arcu rutrum venenatis. Morbi nec sapien turpis. Nunc tincidunt maximus venenatis. Phasellus facilisis imperdiet velit, nec cursus elit tincidunt pretium. Duis ligula metus, rutrum nec ullamcorper a, pretium eu massa. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aliquam condimentum, urna in congue dignissim, mi risus maximus lectus, interdum cursus neque turpis sed libero. Cras iaculis ornare accumsan. Sed tempor pretium est, eget aliquam purus feugiat ac.</p>';
myWin = new N2Window();
myWin.setList({ width: '300px', height: '300px', title: 'Hi Everyone! This is a very long title indeed', top: '100px', left: '10px'});
myWin.show();
myWin.body.html(content);
myWin.sizeToContent();

myWin2 = new N2WindowDraggable();
myWin2.setList({ width: '300px', height: '300px', title: 'Draggable Window', top: '100px', left: '350px'});
myWin2.show();
myWin2.body.html(content);
myWin2.sizeToContent();

myWin3 = new N2WindowResizable();
myWin3.setList({ width: '300px', height: '300px', title: 'Resizable Window', top: '100px', left: '700px'});
myWin3.showFooter();
myWin3.show();
myWin3.body.html(content);
myWin3.sizeToContent();

function thing() { */
// The modelData object is generated and populated by n2_viewer.py
var modelData  = ModelData.uncompressModel(compressedModel);
delete compressedModel;

let n2Diag = new N2Diagram(modelData);
let n2MouseFuncs = n2Diag.getMouseFuncs();
n2Diag.update(false);
// }