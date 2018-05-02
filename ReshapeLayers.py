import numpy as np
import sys


caffe_root = '/usr/local/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')

import caffe
sys.path.insert(0, caffe_root)
import yaml



class ReshapeLayerLevel1(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")

    def reshape(self, bottom, top):       
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],130,130)
 
    def forward(self, bottom, top):
        B=top[0].shape[0]
        C=bottom[0].shape[1]
        for y in range(2):
            for x in range(2):
                top[0].data[0:B,0:C,y*65:(y+1)*65,65*x:(x+1)*65]=bottom[0].data[0:B,0:C,y:130:2,x:130:2]
 
    def backward(self, top, propagate_down, bottom):
        B=top[0].shape[0]        
        C=bottom[0].shape[1]
        for y in range(2):
            for x in range(2):
                bottom[0].diff[0:B,0:C,y:130:2,x:130:2]=top[0].diff[0:B,0:C,y*65:(y+1)*65,65*x:(x+1)*65]
#===========================================================================================      

class TestLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")
            

    def reshape(self, bottom, top):       
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],bottom[0].shape[2],bottom[0].shape[3])
      
    def forward(self, bottom, top):
        top[0].data[...]=2*bottom[0].data

    def backward(self, top, propagate_down, bottom):
        pass
#===========================================================================================             
class ReshapePoolingLayer(caffe.Layer):
    def setup(self, bottom, top):
        print "----------------------------------"
        
        if len(bottom) == 100000:
            raise Exception("Need one ...inputs to compute distance.")
        self.level = yaml.load(self.param_str)["level"]
        self.sizeOut=bottom[0].shape[2]-(2**self.level)+1

        

    def reshape(self, bottom, top):       
        print "----------------------------------"
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],self.sizeOut,self.sizeOut)
      
    def forward(self, bottom, top):
        
        B=top[0].shape[0]
        C=bottom[0].shape[1]
        P=(self.sizeOut)/(2**(self.level+1))
        xTop=yTop=xBottom=yBottom=0      


        
        for yStep in range(2**self.level):
            xBottom=xTop=0
            for xStep in range(2**self.level):           
                for y in range(2):
                    for x in range(2):
                        top[0].data[0:B,0:C,yTop+y*P:yTop+(y+1)*P,xTop+x*P:xTop+(x+1)*P]=bottom[0].data[0:B,0:C,yBottom+y:yBottom+2*P:2,xBottom+x:xBottom+2*P:2]
                xTop+=2*P
                xBottom+=2*P+1
            yTop+=2*P
            yBottom+=2*P+1
    def backward(self, top, propagate_down, bottom):

        B=top[0].shape[0]
        C=bottom[0].shape[1]
        P=(self.sizeOut)/(2**(self.level+1))
        xTop=yTop=xBottom=yBottom=0

        for yStep in range(2**self.level):
            xBottom=xTop=0
            for xStep in range(2**self.level):           
                for y in range(2):
                    for x in range(2):
                        bottom[0].diff[0:B,0:C,yBottom+y:yBottom+2*P:2,xBottom+x:xBottom+2*P:2]=top[0].diff[0:B,0:C,yTop+y*P:yTop+(y+1)*P,xTop+x*P:xTop+(x+1)*P]
                xTop+=2*P
                xBottom+=2*P+1
            yTop+=2*P
            yBottom+=2*P+1

#===========================================================================================             
class DeleteInvalidDataConvLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")
        self.level = yaml.load(self.param_str)["level"]
        self.filterSize = yaml.load(self.param_str)["kernel_size"]
        self.sizeOut=bottom[0].shape[2]-((2**self.level)-1)*(self.filterSize-1)

           

    def reshape(self, bottom, top):       
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],self.sizeOut,self.sizeOut)
      
    def forward(self, bottom, top):

            
        B=top[0].shape[0]
        C=bottom[0].shape[1]
        P=(self.sizeOut)/(2**self.level)
        shift=self.filterSize-1
        yBottom=yTop=0
        for yStep in range(2**self.level):
            xBottom=xTop=0
            for xStep in range(2**self.level):           
                top[0].data[0:B,0:C,yTop:yTop+P,xTop:xTop+P]=bottom[0].data[0:B,0:C,yBottom:yBottom+P,xBottom:xBottom+P]
                xTop+=P
                xBottom+=P+shift
            yTop+=P
            yBottom+=P+shift

    def backward(self, top, propagate_down, bottom):

        B=top[0].shape[0]
        C=bottom[0].shape[1]
        P=(self.sizeOut)/(2**self.level)
        shift=self.filterSize-1
        yBottom=yTop=0
        for yStep in range(2**self.level):
            xBottom=xTop=0
            for xStep in range(2**self.level):           
                bottom[0].diff[0:B,0:C,yBottom:yBottom+P,xBottom:xBottom+P]=top[0].diff[0:B,0:C,yTop:yTop+P,xTop:xTop+P]
                xTop+=P
                xBottom+=P+shift
            yTop+=P
            yBottom+=P+shift
         
#===========================================================================================             
class RestoreLayer(caffe.Layer):
    def setup(self, bottom, top):
        if(1==2):
            self.sizeOut=bottom[0].shape[2];   
            return    
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")
        self.level = yaml.load(self.param_str)["level"]
        self.sizeOut=bottom[0].shape[2]
        bottom=np.zeros((self.sizeOut, self.sizeOut,2))
        top=np.zeros((self.sizeOut, self.sizeOut,2))
        for y in range(self.sizeOut):
            for x in range(self.sizeOut):
                bottom[y,x,0]=y
                bottom[y,x,1]=x
        level=0
        if(self.level==0):
            top[...]=bottom
            
        while(level<self.level):
            P=(self.sizeOut)/(2**(level+1))
            yTop=yBottom=0
            for yStep in range(2**level):
                xBottom=xTop=0
                for xStep in range(2**level):           
                    for y in range(2):
                        for x in range(2):
                            top[yTop+y*P:yTop+(y+1)*P,xTop+x*P:xTop+(x+1)*P,0:2]=bottom[yBottom+y:yBottom+2*P:2,xBottom+x:xBottom+2*P:2,0:2]
                    xTop+=2*P
                    xBottom+=2*P
                yTop+=2*P
                yBottom+=2*P
            bottom[...]=top
            level=level+1        
        self.index=top


    def reshape(self, bottom, top):       
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],self.sizeOut,self.sizeOut)
      
    def forward(self, bottom, top):
        if(1==2):
            top[0].data[...]=bottom[0].data
            return
        B=top[0].shape[0]
        C=bottom[0].shape[1]
        for y in range(self.sizeOut):
            for x in range(self.sizeOut):  
                top[0].data[0:B,0:C,int(self.index[y,x,0]),int(self.index[y,x,1])]=bottom[0].data[0:B,0:C,y,x]

    def backward(self, top, propagate_down, bottom):
        if(1==2):
            bottom[0].diff[...]=top[0].diff
            return
        B=top[0].shape[0]
        C=bottom[0].shape[1]
        for y in range(self.sizeOut):
            for x in range(self.sizeOut):  
                bottom[0].diff[0:B,0:C,y,x]=top[0].diff[0:B,0:C,self.index[y,x,0],self.index[y,x,1]]
            
 

#===========================================================================================             


 

            
#===========================================================================================             
  
class CropInLevelLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")

    def reshape(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")
        self.offset = yaml.load(self.param_str)["offset"]
        self.level = yaml.load(self.param_str)["level"]
        self.evenFilter=0
        #self.sizeOut=bottom[0].shape[2]-self.offset*(2**(self.level+1)) # for odd filter
        self.sizeOut=bottom[0].shape[2]-self.offset*(2**(self.level+1))+2**(self.level)*self.evenFilter #for even filter
        #print "size out is=",self.sizeOut,self.level,bottom[0].shape[2]

        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],self.sizeOut,self.sizeOut)

            
        
    def forward(self, bottom, top):
        if(1==2):
            ss=(bottom[0].shape[2]-1)/2
            top[0].data[:,:,:,:]=bottom[0].data[:,:,ss-2:ss+3,ss-2:ss+3]
            return

        B=top[0].shape[0]
        C=bottom[0].shape[1]
        P=(self.sizeOut)/(2**self.level)
        #print "P is=",P
        
        yBottom=self.offset
        yTop=0
        for yStep in range(2**self.level):
            xBottom=self.offset
            xTop=0
            for xStep in range(2**self.level):           
                top[0].data[0:B,0:C,yTop:yTop+P,xTop:xTop+P]=bottom[0].data[0:B,0:C,yBottom:yBottom+P,xBottom:xBottom+P]
                xTop+=P
                xBottom+=P+2*self.offset-self.evenFilter
            yTop+=P
            yBottom+=P+2*self.offset-self.evenFilter
    def backward(self, top, propagate_down, bottom):
        if(1==2):
            ss=(bottom[0].shape[2]-1)/2
            bottom[0].diff[:,:,ss-2:ss+3,ss-2:ss+3]=top[0].diff[...]
            return
        B=top[0].shape[0]
        C=bottom[0].shape[1]
        P=(self.sizeOut)/(2**self.level)
        yBottom=self.offset
        yTop=0
        for yStep in range(2**self.level):
            xBottom=self.offset
            xTop=0
            for xStep in range(2**self.level):           
                bottom[0].diff[0:B,0:C,yBottom:yBottom+P,xBottom:xBottom+P]=top[0].diff[0:B,0:C,yTop:yTop+P,xTop:xTop+P]
                xTop+=P
                xBottom+=P+2*self.offset-self.evenFilter
            yTop+=P
            yBottom+=P+2*self.offset-self.evenFilter
#===========================================================================================             
  
class CropLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")

    def reshape(self, bottom, top):
        if len(bottom) == 100:
            raise Exception("Need one ...inputs to compute distance.")
        self.offset = yaml.load(self.param_str)["offset"]
        self.sizeOut = yaml.load(self.param_str)["size_out"]
        top[0].reshape(bottom[0].shape[0],bottom[0].shape[1],self.sizeOut,self.sizeOut)
        
    def forward(self, bottom, top):
        B=top[0].shape[0]        
        C=bottom[0].shape[1]
        top[0].data[...]=bottom[0].data[0:B,0:C,self.offset:self.sizeOut+self.offset,self.offset:self.sizeOut+self.offset]

    def backward(self, top, propagate_down, bottom):
        B=top[0].shape[0]        
        C=bottom[0].shape[1]
        bottom[0].diff[0:B,0:C,self.offset:self.sizeOut+self.offset,self.offset:self.sizeOut+self.offset]=top[0].diff

#===========================================================================================             
  
class MySoftMaxWithLossLayer(caffe.Layer):
    def setup(self, bottom, top):
        if len(bottom) != 3:
            raise Exception("Need one ...inputs to compute distance.")

    def reshape(self, bottom, top):
        if bottom[0].num != bottom[1].num:
            raise Exception("Inputs must have the same dimension.")
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.scale=bottom[0].shape[0]*bottom[0].shape[2]*bottom[0].shape[3]
        top[0].reshape(1)
        
    def forward(self, bottom, top):

            
        scores = bottom[0].data[:,:,:,:]
            
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) 
        label=bottom[1].data[:,0,:,:]*probs[:,1,:,:]

        label=label+np.abs(bottom[1].data[:,0,:,:]-1)*probs[:,0,:,:]
        
        data_loss=np.sum(-np.log(label))
        self.diff[...] = probs
        top[0].data[...] = data_loss/(self.scale) 
        
        delta = self.diff        
        delta[:,1,:,:]-=bottom[1].data[:,0,:,:]
        delta[:,0,:,:]-=abs(bottom[1].data[:,0,:,:]-1)   
        delta[:,0,:,:]=delta[:,0,:,:]*bottom[2].data[:,0,:,:]
        delta[:,1,:,:]=delta[:,1,:,:]*bottom[2].data[:,0,:,:] 
        self.diff=delta
    
    def backward(self, top, propagate_down, bottom):
        '''delta = self.diff        
        delta[:,1,:,:]-=bottom[1].data[:,0,:,:]
        delta[:,0,:,:]-=abs(bottom[1].data[:,0,:,:]-1)

        #bottom[2].data[:,:,:,:]=0.71
        delta[:,0,:,:]=delta[:,0,:,:]*bottom[2].data[:,0,:,:]
        delta[:,1,:,:]=delta[:,1,:,:]*bottom[2].data[:,0,:,:]
        m=np.sum(bottom[2].data)
        delta=delta/m'''
        
        

        bottom[0].diff[...] = self.diff



#===========================================================================================             
