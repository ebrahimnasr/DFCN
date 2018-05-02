import copy
import numpy as np
import os
import sys
caffe_root = '/usr/local/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
from PIL import Image
import scipy.io
from os import listdir
from os.path import isfile, join

def Train(fold):
    
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver1 = caffe.AdamSolver('./Model/solver_DFCN.prototxt')
    
    print "start training"
    deviate=0;
    minLoss=10000;
    solver1.step(1)   

    
    
  
    for i in range(10000):
        loss=0;
        for j in range(10):        
            solver1.step(100)
        if(i%10==0):
            solver1.net.save("./Model/Deploy/Fold"+str(fold)+"/DFCN_"+str(i+1)+".caffemodel")
        loss=0;
        solver1.test_nets[0].share_with(solver1.net)        
        for j in range(1000):
            solver1.test_nets[0].forward()
            loss=loss+solver1.test_nets[0].blobs['loss'].data
             
    
        if(i>=0):
    
            print loss
            print minLoss
            print deviate

            if(loss<=minLoss+0.1*minLoss):
                if(loss<=minLoss):
                    minLoss=copy.deepcopy(loss)
                deviate=0
                print "best is"
                print i
                print "min loss is",minLoss
                solver1.net.save("./Model/Deploy/Fold"+str(fold)+"/DFCN_"+str(i+1)+".caffemodel")


            else:
                deviate=deviate+1
                if(deviate==3000):
                    solver1.net.save("./Model/Deploy/Fold"+str(fold)+"/DFCN_"+str(i+1)+".caffemodel")
                    break
    return

    
    
modelAdd='./Model/'
os.makedirs('./Model/Deploy/')
os.makedirs('./Model/Deploy/Fold1')
os.makedirs('./Model/Deploy/Fold2')
os.makedirs('./Model/Deploy/Fold3')
os.makedirs('./Model/Deploy/Fold4')

for fold in  xrange(0, 1):
    trainFile = open(modelAdd+'TrainDatasetAddress.txt','w')
    testFile = open(modelAdd+'TestDatasetAddress.txt','w')
    for i in range(4):
        if(i==fold):
            continue
        if(i==fold+1 or (i==0 and fold==3)):
            trainFile.write('./Dataset/DataBaseFold'+ str(i+1) +'Train.h5\n');   
            testFile.write('./Dataset/DataBaseFold'+ str(i+1)+ 'Test.h5\n');   
            continue;   
        trainFile.write('./Dataset/DataBaseFold'+ str(i+1) +'Train.h5\n');   
        trainFile.write('./Dataset/DataBaseFold'+ str(i+1)+ 'Test.h5\n');  

    trainFile.close()
    testFile.close()
    Train(fold+1)
 
        
    
    
