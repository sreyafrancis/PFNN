# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:15:32 2019

@author: Administrator
"""

import os
import pandas as pd
import numpy as np
import re


def gait(bvh,txt):        
    with open(txt) as f1:    
        list1=[]
        list2=[]
        list3=[]
        list4=[]
        for line in f1.readlines():
            if line[0].isdigit():
                if line.endswith('scol\n') or line.endswith('ecol\n') or line.endswith('ecol'):
                    a=re.findall(r'\d+',line)
                    list4+=a
                    b=re.findall(r'[a-z]+',line)
                    list3+=b                    
                else:
                    a=re.findall(r'\d+',line)
                    list2+=a
                    b=re.findall(r'[a-z]+',line)
                    list1+=b
                
    with open(bvh) as f2:    
        for line in f2.readlines():
            if line.startswith('Frames'):
                #print(re.split(': |:\t',line)[1])
                frames=re.split(': |:\t',line)[1]
                frames=eval(frames)
#    print(list1)  
#    print(list2)  
#    print(list3)
#    print(list4)
    print(bvh)
    
    columns=['stand','walk','jog','run','crouch','jump','crawl','unknown']
    columns_index=[1,2,4,5,6,7,8]
    feature_dict=dict(zip(columns,columns_index))
    
    #->dataframe
    a = np.zeros((frames,8))
    A = pd.DataFrame(a,columns=['stand','walk','jog','run','crouch','jump','crawl','unknown'],index=list(range(1,frames+1)))
    
    #->gait
    for i in range(len(list1)-1):
        if list1[i] == list1[i+1]:
            A[list1[i]].loc[eval(list2[i]):eval(list2[i+1])-1] = A[list1[i]].loc[eval(list2[i]):eval(list2[i+1])-1].apply(lambda x:x+1)
            
        elif feature_dict[list1[i]]<feature_dict[list1[i+1]]:
            A[list1[i]].loc[eval(list2[i]):eval(list2[i+1])-1] = np.linspace(1,0,eval(list2[i+1])-eval(list2[i]),endpoint=True)
            A[list1[i+1]].loc[eval(list2[i]):eval(list2[i+1])-1] = np.linspace(0,1,eval(list2[i+1])-eval(list2[i]),endpoint=True)
        
        elif feature_dict[list1[i]]>feature_dict[list1[i+1]]:
            A[list1[i]].loc[eval(list2[i]):eval(list2[i+1])-1] = np.linspace(1,0,eval(list2[i+1])-eval(list2[i]),endpoint=True)
            A[list1[i+1]].loc[eval(list2[i]):eval(list2[i+1])-1] = np.linspace(0,1,eval(list2[i+1])-eval(list2[i]),endpoint=True)    
    A[list1[-1]].loc[eval(list2[-1]):] = A[list1[-1]].loc[eval(list2[-1]):].apply(lambda x:x+1)
    
    #->unknown
    if list3 and txt!='WalkingUpSteps10_000_gait.txt':          
        for i in range(len(list3)-1)[::2]:
                A['unknown'].loc[eval(list4[i]):eval(list4[i+1])-1] = A['unknown'].loc[eval(list4[i]):eval(list4[i+1])-1].apply(lambda x:x+1)
    else:
        list3=list3[:16]+list3[18:]
        list4=list4[:16]+list4[18:]
        for i in range(len(list3)-1)[::2]:
            A['unknown'].loc[eval(list4[i]):eval(list4[i+1])-1] = A['unknown'].loc[eval(list4[i]):eval(list4[i+1])-1].apply(lambda x:x+1) 
    
    A.to_csv(r'{}.gait'.format(os.path.splitext(bvh)[0]),float_format='%.5f',header=False, index=False,sep=' ')
    A.to_csv(r'{}_mirror.gait'.format(os.path.splitext(bvh)[0]),float_format='%.5f',header=False, index=False,sep=' ')
    

if __name__=='__main__':
    path = r'./data/animations_xxx'
    l1=[]
    for file in os.listdir(path):
        if re.match(r'(\w+).bvh|(\w+)_gait.txt',file):        
            l1.append(file)  
    #print(l1)
            
    for i in range(len(l1))[::2]:
        if l1[i+1].split('.')[0] == l1[i].split('.')[0]+'_gait':
            gait(path+'/'+l1[i],path+'/'+l1[i+1])
     
    