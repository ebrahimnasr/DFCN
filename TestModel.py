import numpy as np
import scipy.io
#import matplotlib.pyplot as plt
import sys
import time
import os

caffe_root = '/usr/local/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from os import listdir
from os.path import isfile, join
import PIL

def ImagePaddingAndBGRtoRGB(im,pad,fitTo): 
    r=im[:,:,0]
    r= np.lib.pad(r, pad,'constant')
    g=im[:,:,1]
    g= np.lib.pad(g, pad,'constant')
    b=im[:,:,2]
    b= np.lib.pad(b, pad,'constant')
    res=np.zeros((fitTo[0],fitTo[1],3))
    res[0:r.shape[0],0:r.shape[1],0]=r
    res[0:r.shape[0],0:r.shape[1],1]=g
    res[0:r.shape[0],0:r.shape[1],2]=b
    return res


for deployNum in  xrange(100,101): 
    caffe.set_device(0)
    caffe.set_mode_gpu()
    targetDeploy=deployNum
    
    file_path = "./Model/Deploy/Result/Fold4_"+str(targetDeploy)+"k"
    
    os.makedirs(file_path)
    for fold in  xrange(4, 5):
        directoryAddress='./Dataset/Original/Fold'+str(fold)+'/'
        print "fold=",directoryAddress
        net =  caffe.Net('/Model/DFCN_deploy.prototxt',  './Model/Deploy/Fold'+str(fold)+'/DFCN_'+str(targetDeploy)+'.caffemodel', caffe.TEST)
        fileList = [f for f in listdir(directoryAddress) if isfile(join(directoryAddress, f))]
        numPack=2;
        
        resultSize=32*4+256
        patchSize=155+256
     
        paddLocal=(26)/2
        
        resSize=resultSize
        centerPad=22
        
        for fileName in fileList:
        
            temp=fileName.split('.', 1 ) 
            if temp[1]=='jpg':
                print fileName     
                im=mpimg.imread(directoryAddress+fileName)
                im=np.array(im,dtype=float)
                tempIm=im[:,:,0]
                tempIm=(tempIm-np.mean(tempIm))/np.std(tempIm)
                im[:,:,0]=tempIm

                tempIm=im[:,:,1]
                tempIm=(tempIm-np.mean(tempIm))/np.std(tempIm)
                im[:,:,1]=tempIm
                
                tempIm=im[:,:,2]
                tempIm=(tempIm-np.mean(tempIm))/np.std(tempIm)
                im[:,:,2]=tempIm
                
                tic = time.clock()
                inputImage=ImagePaddingAndBGRtoRGB(im,paddLocal,[im.shape[0]+2*paddLocal+400,im.shape[1]+2*paddLocal+400] )
                
                scale=np.max(inputImage)

                patch=np.zeros((1,3,patchSize,patchSize))
                
                res=np.zeros((im.shape[0]+resultSize, im.shape[1]+resultSize))
                numVoter=np.zeros((im.shape[0]+resultSize, im.shape[1]+resultSize))
                for y in range(0,im.shape[0],resultSize/2):
                    for x in range(0,im.shape[1],resultSize/2): 
                        tempLocal=inputImage[y:y+patchSize,x:x+patchSize,:]
                        patch[0,0,:,:]=np.transpose (tempLocal[:,:,0])
                        patch[0,1,:,:]=np.transpose (tempLocal[:,:,1])
                        patch[0,2,:,:]=np.transpose (tempLocal[:,:,2])
        
    
                        #patch=patch/scale
                        
                        net.blobs['GlobalPatch'].data[...] = patch
                        #net.blobs['LocalPatch'].data[...] = localpatch
                        net.forward()
                        score= net.blobs['prob'].data

                        
                        res[y:y+resSize,x:x+resSize]=np.transpose (score[0,1,:,:])

                res=res[0:im.shape[0],0:im.shape[1]]
                mpimg.imsave('./Model/Deploy/Result/Fold'+str(fold)+'_'+str(targetDeploy)+'k/'+temp[0]+'.jpg', res)
                scipy.io.savemat('./Model/Deploy/Result/Fold'+str(fold)+'_'+str(targetDeploy)+'k/'+fileName+'.mat', mdict={'res': res})
                toc = time.clock()
                elapsedtime = toc - tic
                print "time=",elapsedtime
                
                
        
        
