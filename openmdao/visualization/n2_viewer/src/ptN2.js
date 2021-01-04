let sharedTransition = null;

var enterIndex = 0;
var exitIndex = 0;

/*
myWin = new N2Window();
myWin.set({ width: '300px', height: '300px', title: 'Hi Everyone! This is a very long title indeed', top: '100px', left: '10px'});
myWin.show();

myWin2 = new N2WindowDraggable();
myWin2.set({ width: '300px', height: '300px', title: 'Draggable Window', top: '100px', left: '350px'});
myWin2.show();

myWin3 = new N2WindowResizable();
myWin3.set({ width: '300px', height: '300px', title: 'Resizable Window', top: '100px', left: '700px'});
myWin3.showFooter();
myWin3.show();

function thing() { */
// The modelData object is generated and populated by n2_viewer.py
var modelData  = ModelData.uncompressModel(compressedModel);
delete compressedModel;

let n2Diag = new N2Diagram(modelData);
let n2MouseFuncs = n2Diag.getMouseFuncs();
n2Diag.update(false);
// }