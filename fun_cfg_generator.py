# -*- coding: utf-8 -*-
"""
Darknet cfg generator functions
main entry is main_Darknet_cfgGenerator.py

@author: Chih-Sheng (Tommy) Huang
chih.sheng.huang821@gmail.com
"""

def sheet2list(sheet):
    list_sheet=[]
    for row in sheet.rows:
        sub_list=[]
        for cell in row:
            sub_list.append(cell.internal_value)
        list_sheet.append(sub_list)
    return list_sheet


def fun_list2cfg(list_net, list_modelstruct):
    # init cfg parse engine
    f_cfg_generator=cfg_generator()
    cfg_list_net=[]
    # 1. [net]
    para_net={}
    for tmp_net in list_net:
        para_net[tmp_net[1]]=tmp_net[2]
    cfg_list_net.append(f_cfg_generator.gen_net(para_net))
    # 2. NN body
    for i_L, tmp_layer in enumerate(list_modelstruct):
        list_tmp_layer={}
        if i_L == 0 :
            parameter_name = tmp_layer
        elif i_L ==1 : pass
        else:
            for tmp_parameter_name,tmp_inf_layer in zip(parameter_name,tmp_layer):
                if tmp_inf_layer!='-':
                    list_tmp_layer[tmp_parameter_name]= tmp_inf_layer
            layer_type = list_tmp_layer['layer_type']
            if layer_type=="convolutional":
                cfg_list_net.append(f_cfg_generator.gen_conv(list_tmp_layer)) 
            elif layer_type=="maxpool":
                cfg_list_net.append(f_cfg_generator.gen_pool(list_tmp_layer)) 
            elif layer_type=="upsample":
                cfg_list_net.append(f_cfg_generator.gen_upsample(list_tmp_layer)) 
            elif layer_type=="avgpool":
                cfg_list_net.append(f_cfg_generator.gen_avgpool(list_tmp_layer)) 
            elif layer_type=="softmax":
                cfg_list_net.append(f_cfg_generator.gen_softmax(list_tmp_layer))     
            elif layer_type=="connected":
                cfg_list_net.append(f_cfg_generator.gen_connected(list_tmp_layer)) 
            elif layer_type=="dropout":
                cfg_list_net.append(f_cfg_generator.gen_dropout(list_tmp_layer)) 
            elif layer_type=="local":
                cfg_list_net.append(f_cfg_generator.gen_local(list_tmp_layer))
            elif layer_type=="rnn":
                cfg_list_net.append(f_cfg_generator.gen_rnn(list_tmp_layer))
            elif layer_type=="crnn":
                cfg_list_net.append(f_cfg_generator.gen_crnn(list_tmp_layer))
            elif layer_type=="yolo":
                cfg_list_net.append(f_cfg_generator.gen_yolo(list_tmp_layer))
            elif layer_type=="region":
                cfg_list_net.append(f_cfg_generator.gen_region(list_tmp_layer))
            elif layer_type=="detection":
                cfg_list_net.append(f_cfg_generator.gen_detection(list_tmp_layer))
            elif layer_type=="cost":
                cfg_list_net.append(f_cfg_generator.gen_cost(list_tmp_layer))
            elif layer_type=="ElewiseSUM (shortcut, residual)":
                lc_current = list_tmp_layer['current_layer_code']        
                lc_previous = str(list_tmp_layer['previous_layer_code'])
                lc_previous = [tmp.strip() for tmp in lc_previous.split(',')]
                if len(lc_previous)==1:
                    if list_tmp_layer['from']!=(int(lc_previous[0])-lc_current):
                        list_tmp_layer['from']=(int(lc_previous[0])-lc_current)
                else:
                    tmp_fromShortcut=[]
                    for tmp_lc_previous in lc_previous:
                        if (int(tmp_lc_previous)-lc_current)>-10:
                            tmp_fromShortcut.append(int(tmp_lc_previous)-lc_current)
                        else:
                            tmp_fromShortcut.append(int(tmp_lc_previous))
                    list_tmp_layer['from']=tmp_fromShortcut
                cfg_list_net.append(f_cfg_generator.gen_shortcut(list_tmp_layer))
            elif layer_type=="route (Concate or Branch)":
                lc_current = list_tmp_layer['current_layer_code']        
                lc_previous = str(list_tmp_layer['previous_layer_code'])
                lc_previous = [tmp.strip() for tmp in lc_previous.split(',')]
                if len(lc_previous)==1:
                    if list_tmp_layer['layers']!=(int(lc_previous[0])-lc_current):
                        list_tmp_layer['layers']=(int(lc_previous[0])-lc_current)
                else:
                    tmp_fromRotue=[]
                    for tmp_lc_previous in lc_previous:
                        if (int(tmp_lc_previous)-lc_current)>-10:
                            tmp_fromRotue.append(int(tmp_lc_previous)-lc_current)
                        else:
                            tmp_fromRotue.append(int(tmp_lc_previous))
                    list_tmp_layer['layers']=tmp_fromRotue 
                cfg_list_net.append(f_cfg_generator.gen_route(list_tmp_layer))
            else:
                print('Unknown Layer type: ' + layer_type)
    return cfg_list_net

class cfg_generator:
    def __init__(self): pass

    def list_addelement(self,list_net,**kwargs):
        name_element,inputvalue=[],[]
        for k, v in kwargs.items():
            if k=='name_ele': name_element = v
            if k=='in_value': inputvalue = v
        if len(name_element)!=0:
            if isinstance(inputvalue,list):
                val=''
                for tmp in inputvalue:
                    val+= str(tmp) + ',' 
                val=val[0:-1]    
                list_net.append(name_element + ' = ' + val)
            else:
                list_net.append(name_element + ' = ' + str(inputvalue))
        return list_net

    def fun_check_para(self,list_net, para_net, para_name, default=1):
        if para_name in para_net: 
            list_net = self.list_addelement(list_net,name_ele=para_name,in_value=para_net[para_name])
        else:
            list_net = self.list_addelement(list_net,name_ele=para_name,in_value=default) 
        return list_net
    
    def gen_net(self, para_net):
        list_net=[]
        list_net.append('[net]')
        list_net = self.fun_check_para(list_net, para_net, 'batch', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'subdivisions', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'width', default=416)
        list_net = self.fun_check_para(list_net, para_net, 'height', default=416)
        list_net = self.fun_check_para(list_net, para_net, 'channels', default=3)
        list_net = self.fun_check_para(list_net, para_net, 'momentum', default=0.9)
        list_net = self.fun_check_para(list_net, para_net, 'decay', default=0.0005)
        list_net = self.fun_check_para(list_net, para_net, 'angle', default=0)
        list_net = self.fun_check_para(list_net, para_net, 'saturation', default=1.5)
        list_net = self.fun_check_para(list_net, para_net, 'exposure', default=1.5)
        list_net = self.fun_check_para(list_net, para_net, 'hue', default=.1)
        list_net = self.fun_check_para(list_net, para_net, 'learning_rate', default=0.001)
        list_net = self.fun_check_para(list_net, para_net, 'burn_in', default=1000)
        list_net = self.fun_check_para(list_net, para_net, 'max_batches', default=500200)
        list_net = self.fun_check_para(list_net, para_net, 'policy', default='steps')
        # policy：CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
        list_net = self.fun_check_para(list_net, para_net, 'steps', default=[400000,450000])
        list_net = self.fun_check_para(list_net, para_net, 'scales', default=[.1,.1])
        return list_net
     
    def gen_conv(self, para_net):
        list_net=[]
        list_net.append('[convolutional]')
        list_net = self.fun_check_para(list_net, para_net, 'batch_normalize', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'filters', default=16)
        list_net = self.fun_check_para(list_net, para_net, 'size', default=3)
        list_net = self.fun_check_para(list_net, para_net, 'stride', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'pad', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'activation', default='leaky')
        return list_net  
    
    def gen_pool(self, para_net):
        list_net=[]
        list_net.append('[maxpool]')
        list_net = self.fun_check_para(list_net, para_net, 'size', default=2)
        list_net = self.fun_check_para(list_net, para_net, 'stride', default=2)
        return list_net
    
    def gen_route(self, para_net):
        list_net=[]
        list_net.append('[route]')
        list_net = self.fun_check_para(list_net, para_net, 'layers', default=-1)
        return list_net  
    
    def gen_upsample(self, para_net):
        list_net=[]
        list_net.append('[upsample]')
        list_net = self.fun_check_para(list_net, para_net, 'stride', default=2)
        return list_net 
     
    def gen_avgpool(self, para_net):
        list_net=[]
        list_net.append('[avgpool]')
        return list_net  
    
    def gen_reorg(self, para_net):
        list_net=[]
        list_net.append('[reorg]')
        list_net = self.fun_check_para(list_net, para_net, 'stride', default=2)
        return list_net 
    
    def gen_shortcut(self, para_net):
        list_net=[]
        list_net.append('[shortcut]')
        list_net = self.fun_check_para(list_net, para_net, 'from', default=-1)
        list_net = self.fun_check_para(list_net, para_net, 'activation', default='linear')
        return list_net 
    
    def gen_softmax(self, para_net):
        list_net=[]
        list_net.append('[softmax]')
        return list_net  
    
    def gen_connected(self, para_net):
        list_net=[]
        list_net.append('[connected]')
        list_net = self.fun_check_para(list_net, para_net, 'output', default=256)
        list_net = self.fun_check_para(list_net, para_net, 'activation', default='leaky')
        return list_net  
     
    def gen_dropout(self, para_net):
        list_net=[]
        list_net.append('[dropout]')
        list_net = self.fun_check_para(list_net, para_net, 'probability', default=.5)
        return list_net  
    
    def gen_local(self, para_net):
        list_net=[]
        list_net.append('[local]')
        list_net = self.fun_check_para(list_net, para_net, 'size', default=3)
        list_net = self.fun_check_para(list_net, para_net, 'stride', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'pad', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'filters', default=256)
        list_net = self.fun_check_para(list_net, para_net, 'activation', default='leaky')
        return list_net  
    
    def gen_rnn(self, para_net):
        list_net=[]
        list_net.append('[rnn]')
        list_net = self.fun_check_para(list_net, para_net, 'batch_normalize', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'output', default=1024)
        list_net = self.fun_check_para(list_net, para_net, 'hidden', default=1024)
        list_net = self.fun_check_para(list_net, para_net, 'activation', default='leaky')
        return list_net  
    
    def gen_crnn(self, para_net):
        list_net=[]
        list_net.append('[crnn]')
        list_net = self.fun_check_para(list_net, para_net, 'batch_normalize', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'size', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'pad', default=0)
        list_net = self.fun_check_para(list_net, para_net, 'output', default=1024)
        list_net = self.fun_check_para(list_net, para_net, 'hidden', default=1024)
        list_net = self.fun_check_para(list_net, para_net, 'activation', default='leaky')
        return list_net 
    
    def gen_yolo(self, para_net):
        list_net=[]
        list_net.append('[yolo]')
        list_net = self.fun_check_para(list_net, para_net, 'mask', default=[0,1,2])
        list_net = self.fun_check_para(list_net, para_net, 'anchors', default=[10,14,  23,27,  37,58,  81,82,  135,169,  344,319])
        list_net = self.fun_check_para(list_net, para_net, 'classes', default=80)
        list_net = self.fun_check_para(list_net, para_net, 'num', default=6)
        list_net = self.fun_check_para(list_net, para_net, 'jitter', default=.3)
        list_net = self.fun_check_para(list_net, para_net, 'ignore_thresh', default=.7)
        list_net = self.fun_check_para(list_net, para_net, 'truth_thresh', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'random', default=1)
        return list_net 
    
    def gen_region(self, para_net):
        list_net=[]
        list_net.append('[region]')
        list_net = self.fun_check_para(list_net, para_net, 'anchors', default=[1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071])
        list_net = self.fun_check_para(list_net, para_net, 'bias_match', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'classes', default=20)
        list_net = self.fun_check_para(list_net, para_net, 'coords', default=4)
        list_net = self.fun_check_para(list_net, para_net, 'num', default=5)
        list_net = self.fun_check_para(list_net, para_net, 'jitter', default=.3)
        list_net = self.fun_check_para(list_net, para_net, 'softmax', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'rescore', default=1)
        return list_net 
    
    def gen_detection(self, para_net):
        list_net=[]
        list_net.append('[detection]')
        list_net = self.fun_check_para(list_net, para_net, 'classes', default=20)
        list_net = self.fun_check_para(list_net, para_net, 'coords', default=4)
        list_net = self.fun_check_para(list_net, para_net, 'rescore', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'side', default=7)
        list_net = self.fun_check_para(list_net, para_net, 'num', default=3)
        list_net = self.fun_check_para(list_net, para_net, 'softmax', default=0)
        list_net = self.fun_check_para(list_net, para_net, 'sqrt', default=1)  
        list_net = self.fun_check_para(list_net, para_net, 'jitter', default=.2)
        list_net = self.fun_check_para(list_net, para_net, 'object_scale', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'noobject_scale', default=.5)
        list_net = self.fun_check_para(list_net, para_net, 'class_scale', default=1)
        list_net = self.fun_check_para(list_net, para_net, 'coord_scale', default=5)
        return list_net  
    
    def gen_cost(self, para_net):
        list_net=[]
        list_net.append('[cost]')
        list_net = self.fun_check_para(list_net, para_net, 'type', default='sse')
        return list_net  


