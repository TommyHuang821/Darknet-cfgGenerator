# -*- coding: utf-8 -*-
"""
Darknet cfg generator
Model structure should be created in a defined '.xlsx' file. 
model_yolov3.xlsx is the example
You can modify/create a new NN structure by following the example file.

@author: Chih-Sheng (Tommy) Huang
chih.sheng.huang821@gmail.com
"""

from openpyxl import load_workbook
import fun_cfg_generator

filename="model_yolov3.xlsx"
save_cfg_filename="model_yolov3.cfg"


def cfg_writer(save_cfg_filename, cfg_list_net):
    with open(save_cfg_filename, 'w') as filewriter:
        for layer_list in cfg_list_net:
            for sub_layer_list in layer_list:
                filewriter.write(sub_layer_list)       
                filewriter.write('\n')       
            filewriter.write('\n') 


if __name__=='__main__': 
    # parse model.xlxs
    wb = load_workbook(filename,data_only = True)
    net = wb['net']
    modelstruct = wb['model strucutre']
    # xlxs sheet to python list 
    list_net = fun_cfg_generator.sheet2list(net)
    list_modelstruct=fun_cfg_generator.sheet2list(modelstruct)
    # python list to cfg format
    cfg_list_net = fun_cfg_generator.fun_list2cfg(list_net, list_modelstruct)
    # cfg writer
    cfg_writer(save_cfg_filename, cfg_list_net)       
        
 
      
        

            
                    
            
        
        

  
