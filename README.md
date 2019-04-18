# Darknet cfg filer generator by a .xlsx file

Darknet is used to configure for YOLO structure in the .cfg file.

However, it's not easy to configure your own NN structure in the .cfg file, if your NN structure is complexity, i.e. Inception. You must use lot of [route] and [shortcut] components, and you may confuse in setting the parameter: layers=? or from=?

This cfg generator help to solve this problem.

The network structure can be configured by using the .xlsx file. ([model_yolov3.xlsx](https://github.com/TommyHuang821/Darknet-cfgGenerator/blob/master/model_yolov3.xlsx) is the example in configuring the YOLOv3.)

The "current_layer_code" is the code number for the current component. You must add the "previous_layer_code" for the current component, in meaning that this component is following by the "previous_layer_code" component.

The default parameter of each darknet component would be imported automatically by setting the sheet (Component). You can easy modified the parameter in the default value. 

The concatenation and elementwiseSum are easy to implement by the [route] ("route (Concate or Branch)" in my xlsx) and [shortcut] ("ElewiseSUM (shortcut, residual)" in my xlsx)  with more than two number in setting in "previous_layer_code", respectively. The branch of the NN structure would be implemented by set only one vaule in "previous_layer_code" in [route]. The default of the components "route (Concate or Branch)" and "ElewiseSUM (shortcut, residual)" doesn't need to modify in .xlsx, my code would parse the previous_layer_code and generate the correct parameter in cfg file.

I think it's easy to use, so I don't make the detail tutorial.

When your own .cfg is ready, the cfg NN structure can be visualized by ["Visualization-for-Darknet-network-structure-"](https://github.com/TommyHuang821/Visualization-for-Darknet-network-structure-). 

## main entry
main_Darknet_cfgGenerator.py
