import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
import pdb
from numpy import linalg as LA
from scipy.misc import logsumexp

data = pd.DataFrame(np.zeros((5000, 3)), columns=['x1', 'x2', 'y'])

# Let's make up some noisy XOR data to use to build our binary classifier
for i in range(len(data.index)):
    x1 = random.randint(0,1)
    x2 = random.randint(0,1)
    if x1 == 1 and x2 == 0:
        y = 0
    elif x1 == 0 and x2 == 1:
        y = 0
    elif x1 == 0 and x2 == 0:
        y = 1
    else:
        y = 2
    x1 = 1.0 * x1 + 0.20 * np.random.normal()
    x2 = 1.0 * x2 + 0.20 * np.random.normal()
    data.iloc[i,0] = x1
    data.iloc[i,1] = x2
    data.iloc[i,2] = y
    
for i in range(int(0.25 *len(data.index))):
    k = np.random.randint(len(data.index)-1)  
    data.iloc[k,0] = 1.5 + 0.20 * np.random.normal()
    data.iloc[k,1] = 1.5 + 0.20 * np.random.normal()
    data.iloc[k,2] = 1

for i in range(int(0.25 *len(data.index))):
    k = np.random.randint(len(data.index)-1)  
    data.iloc[k,0] = 0.5 + 0.20 * np.random.normal()
    data.iloc[k,1] = -0.75 + 0.20 * np.random.normal()
    data.iloc[k,2] = 2
    
# Now let's normalize this data.
data.iloc[:,0] = (data.iloc[:,0] - data['x1'].mean()) / data['x1'].std()
data.iloc[:,1] = (data.iloc[:,1] - data['x2'].mean()) / data['x2'].std()
        
data.head()

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# The cost function is expecting numpy matrices so we need to convert X and y before we can use them.  
X = np.matrix(X.values)
y = np.matrix(y.values)

def plot_data(X, y_predict):
    
    fig, ax = plt.subplots(figsize=(12,8))
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    indices_0 = [k for k in range(0, X.shape[0]) if y_predict[k] == 0]
    indices_1 = [k for k in range(0, X.shape[0]) if y_predict[k] == 1]
    indices_2 = [k for k in range(0, X.shape[0]) if y_predict[k] == 2]

    ax.plot(X[indices_0, 0], X[indices_0,1], marker='o', linestyle='', ms=5, label='0')
    ax.plot(X[indices_1, 0], X[indices_1,1], marker='o', linestyle='', ms=5, label='1')
    ax.plot(X[indices_2, 0], X[indices_2,1], marker='o', linestyle='', ms=5, label='2')

    ax.legend()
    ax.legend(loc=2)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Tricky 3 Class Classification')
    plt.show()

######################################
class HiddenLayer:
    def __init__(self, dim_in, dim_out,method=1):
        self.dim_in,self.dim_out=dim_in, dim_out
        # initialise weights and biases
        if method==1:
            self.w=np.random.random((dim_in, dim_out)) - 0.5
            self.b=np.random.random((dim_out,1)) - 0.5
            #self.w=np.random.random((dim_in, dim_out)) - 0.5
            #self.b = np.random.randn(dim_out,1)
            #self.b = 0.1 * np.random.random()

            #self.b = np.zeros([dim_out,1]) #0.1*np.random.random((dim_out,1))
        elif method==2:
            self.w = np.asmatrix(np.random.randn(dim_in, dim_out) / np.sqrt(dim_in))
            self.w = np.vstack((self.w, np.random.randn(1, dim_out)))
        else:
            self.w=np.random.randn(dim_in, dim_out)
            self.b = 3 * np.random.randn(dim_out,1) 
        self.dldw_net=np.zeros([dim_in, dim_out])
        self.dldb_net=np.zeros([dim_out,1])

    def forward_prop(self,x):
        self.x=x
        self.u = np.dot(np.transpose(self.w), x) + self.b
        #self.h = np.fmax(self.u, 0.01*self.u) 
        self.h = np.fmax(self.u, 0) 
        return self.h


    def backward_prop(self,dldh):
        #print self.h
        fu_2=(self.u>0).astype(float)
        #fu_2 = 1.0 * (self.u > 0.0) + 0.01 * (self.u < 0.0) 
        #print np.max(self.u)
        if np.isnan(np.min(self.u)):
            pdb.set_trace()
        int1=np.multiply(fu_2,dldh)
        dldw=self.x*int1.T
        dldb=int1
        self.dldw_net+=dldw
        self.dldb_net+=dldb
        dldh= np.matrix(self.w)*np.matrix(int1)
        return dldh
    
    def backward_prop_prev(self,dldh):
        f=(self.u>0).astype(float)
        #fu_2 = 1.0 * (self.u > 0.0) + 0.01 * (self.u < 0.0) 
   
        if np.isnan(np.min(self.u)):
            pdb.set_trace()
        dldb= np.multiply(f, dldh)
        dldw = np.dot(self.x, np.transpose(np.multiply(f, dldh)))
        dldh = self.w*np.multiply(f, dldh)
#       
        return dldh

    def update_weights(self,lr,reg_param):
        self.w=self.w-lr * self.dldw_net+reg_param*self.w
        self.b=self.b-lr * self.dldb_net
        self.dldw_net=np.zeros([self.dim_in, self.dim_out])
        self.dldb_net=np.zeros([self.dim_out,1])
    
class OutputLayer:
    def __init__(self, dim_in, dim_out,method=1):
        self.dim_in,self.dim_out=dim_in, dim_out
        # initialise weights and biases
        if method==1:
            self.w=np.random.random((dim_in, dim_out)) - 0.5
            self.b=np.random.random((dim_out,1)) - 0.5
            #self.w=np.random.random((dim_in, dim_out)) - 0.5
            #self.b = 0.1 * np.random.random()
            #self.b = np.random.randn(dim_out,1)
            #self.b =np.zeros([dim_out,1])#0.1*np.random.random((dim_out,1))
        elif method ==2:
            self.w = np.asmatrix(np.random.randn(dim_in, dim_out) / np.sqrt(dim_in))
            self.w = np.vstack((self.w, np.random.randn(1, dim_out)))
        else:
            self.w=np.random.randn(dim_in, dim_out)
            self.b = 3* np.random.randn(dim_out,1) 
        self.b = 0.1 * np.random.random((dim_out,1))
        self.dldw_net=np.zeros([dim_in, dim_out])
        self.dldb_net=np.zeros([dim_out,1])

    def forward_prop(self,x):
        self.x=x
        # here x would be 3 x 1
        h = np.dot(np.transpose(self.w), x) + self.b
        return h

    def backward_prop(self,dldz):
        dldw=self.x*dldz.T # self.x is the input to the network
        dldb=dldz
        self.dldw_net+=dldw
        self.dldb_net+=dldb
        dldh= np.matrix(self.w)*np.matrix(dldz)
        return dldh

    def update_weights(self,lr,reg_param):
        self.w=self.w-lr * self.dldw_net+reg_param*self.w
        self.b=self.b-lr * self.dldb_net
        self.dldw_net=np.zeros([self.dim_in, self.dim_out])
        self.dldb_net=np.zeros([self.dim_out,1])

    
    
class LossLayer:
    def __init__(self, dim_in, dim_out):
        self.dim_in=dim_in
        self.dim_out=dim_out

    def forward_prop(self,x,y):
        self.x=x
        loss=-1*x.item(y)+logsumexp(x)
        if loss >1:
            pass
        return loss
    

    def backward_prop(self,y):
        # y is the correct label
        indicator=np.zeros([self.dim_in,1])
        indicator[y]=-1
        self.x-=np.max(self.x) # to make it stable
        dldz= indicator+np.exp(self.x.item(y))/np.sum(np.exp(self.x))
        return dldz
              
class MLP:
    def __init__(self):
        self.layers=[]
        self.num_layers=0
        
    
    def add_layer(self,layer_type,dim_in,dim_out):
        if layer_type =='Hidden':
            layer=HiddenLayer(dim_in,dim_out)
            self.hidden_units=dim_out
            self.num_layers+=1
        elif layer_type == 'Output':
            layer=OutputLayer(dim_in,dim_out)
        elif layer_type == 'Loss':
            layer=LossLayer(dim_in,dim_out)
        self.layers.append(layer)
        
            
    def training(self,num_epochs,bsize,reg_param=0, learning_rate=0.001):
        import copy
        x = copy.deepcopy(NN.layers[1].w)
        learning_rate=learning_rate/bsize
        self.reg_param=reg_param
        N=X.shape[0]
        losses=[]
        # include gradient descent here
        for t in range(1, num_epochs):
            eta = .001 * np.exp(-t/1000.0)
            learning_rate=eta/bsize
            loss=0      
            prev_loss=0                    #eta = 1.0 * np.exp(-t/100.0)
            samples_indexes=np.random.permutation(N)
            for bdx in range(0,N,bsize):
                batch_indexes=samples_indexes[bdx:bdx+bsize]
                for k in batch_indexes:  
                    y_bdx=int(y.item(k))
                    l=self.forward_prop(np.transpose(X[k,:]),y_bdx)
                    loss+=l
                    self.back_prop(y_bdx)
                
                for i,layer in enumerate(self.layers):
                    if i == len(NN.layers) - 1:
                        break
                    layer.update_weights(learning_rate,reg_param)
            if t % 10 ==0:
                print t, loss
                #print NN.layers[1].w
                #if np.any(np.equal(x,NN.layers[1].w))==False:
                #    pdb.set_trace()
                
                    
            if prev_loss>loss:
                break
                #self.prediction()
            #losses.append(loss)
        #plt.plot(range(1, num_epochs),losses)
    
    def forward_prop(self,x, y):
        # 1. iterate over hidden layers and output layer
        y=int(y)
        for i,layer in enumerate(self.layers):
            if i < len(NN.layers) - 1:
                x = layer.forward_prop(x)
        # 2. output layer
        loss=self.layers[-1].forward_prop(x,y)
        
        return loss

    def back_prop(self,y):
        
        # dL = loss_layer_backward(z, y)
        dldz=self.layers[-1].backward_prop(y)
        
        # output layer
        dldh=self.layers[-2].backward_prop(dldz)

        # successively all hidden layers
        for ilayer in range(-3,-1*len(self.layers)-1,-1):
            dldh=self.layers[ilayer].backward_prop(dldh)
 
    
    def prediction(self):
        # include forwardprop here, returns label 
        fig, ax = plt.subplots(figsize=(12,8))
        ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
        self.plot_predict(ax)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_title('Hidden units:'+str(self.hidden_units)+' No. of Layers:'+str(self.num_layers)+' Reg. value:'+str(self.reg_param))
        plt.show()  
    
    def plot_predict(self,ax):
        #ploting              
        for index in range(X.shape[0]):
            x1=X[index,:]
            x=np.transpose(x1)
            for i,layer in enumerate(self.layers):
                if i < len(NN.layers) - 1:
                    x = layer.forward_prop(x)
                else:
                    prediction=np.argmax(x)
            if prediction==0:
                ax.plot(x1.item(0),x1.item(1),label='1',marker='o', linestyle='', ms=5,color='r')
            elif prediction==1:
                ax.plot(x1.item(0),x1.item(1),label='0',marker='o', linestyle='', ms=5,color='g')
            elif prediction ==2:
                ax.plot(x1.item(0),x1.item(1),label='-1',marker='o', linestyle='', ms=5,color='y')
    
  
    
testing=True     
if testing:
    NN = MLP()
    NN.add_layer('Hidden', dim_in=2, dim_out=16)
 
    NN.add_layer('Output', dim_in=16, dim_out=3)
    NN.add_layer('Loss', dim_in=3, dim_out=1)
    NN.training(10000,100,0)
    NN.prediction()

for hidden_units in [3,8,16]:
    # constructing the network
    for layer in [1,2]:
        for reg_param in [0.01]:
            print hidden_units, layer,reg_param
            NN = MLP()
            NN.add_layer('Hidden', dim_in=2, dim_out=hidden_units)
            if layer ==2:
                NN.add_layer('Hidden', dim_in=hidden_units, dim_out=hidden_units)
            NN.add_layer('Output', dim_in=hidden_units, dim_out=3)
            NN.add_layer('Loss', dim_in=3, dim_out=1)
            
            # training and prediction
            NN.training(1200,100,reg_param)
            NN.prediction()

