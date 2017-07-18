# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 22:23:19 2017

@author: Manoj_pc
"""
import numpy as np

class Network(object):
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.rand(y, 1) for y in sizes[1:]]
        self.weights = [np.random.rand(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]
        pass
    """
    tanh derivative function
    """
    def tanh(self,z):
        return np.tanh(z)
    """
    tanh derivative 
    """
    def tanh_derivative(self,z):
        layers_derv = []
        for layer in range(0,len(z)):
            layers_derv.append((1 - z[layer]*z[layer]))
        return layers_derv
    
    """
    Feed Forward Network for neutral network.Here input in passed onto next \
    stages of network and tanh is used as transfer function.
    """
    
    def feedforward(self,a):
        sigmoid_out = []
        prev_inp = a
        for b,w in zip(self.biases, self.weights):
           # print(prev_inp)
            prev_inp=np.array(self.tanh(np.dot(w,prev_inp)+b))
            sigmoid_out.append(prev_inp)
        return sigmoid_out
    
    """
    calculates mean square error of output with expected output
    """
    @staticmethod
    def mean_squared_error(y,y_hat):
        return np.sum(((y_hat - y)**2)/2)
        pass
    
    """
    calculate net gardients for all network
    """
    def evaluate_gradients(self,y,y_hat,z):
        gradient = []
        for layer in range(len(z)-1,-1,-1):
            if layer is (len(z)-1):
                gradient.append(z[layer] * (y_hat - y))
            else:
                gradient.append(z[layer] *  np.dot(net.weights[layer+1].T,gradient[len(gradient)-1]) )      
               
        
        gradient.reverse()
        return gradient
    
    """
    calculate delta weights for each network 
    """
    def evaluate_weights(self,gradient,eta,epoch,and_inp):
        delta_weights=[]
        for layer in range(0,len(gradient)):
            if layer is 0:
                delta_weights.append(np.dot(gradient[layer]*eta,and_inp.T))
            else:
                delta_weights.append(np.dot(gradient[layer]*eta,epoch[layer-1].T))
                
        return delta_weights
        pass
    
    """
    calculate delta biases for each network
    """
    def evaluate_biases(self,gradient,eta):
        delta_biases=[]
        for layer in range(0,len(gradient)):
            delta_biases.append(np.dot(gradient[layer],np.ones((len(gradient[layer][0]),1)) )*eta )
        
        return delta_biases
        pass
            
  
    def update_weights(self,delta_weights):
        updated_weight=self.weights;
        for layer in range(0,len(delta_weights)):
            updated_weight[layer]=updated_weight[layer]-delta_weights[layer]
        self.weights=updated_weight
        pass 
    
    
    def update_biases(self,delta_biases):
        updated_biases=self.biases;
        for layer in range(0,len(delta_biases)):
            updated_biases[layer]=updated_biases[layer]-delta_biases[layer]
        self.biases=updated_biases
        pass 

        
np.random.seed(3)
net = Network([2,2,1])

"""
Calcution of Xor of two inputs
"""
and_inp = np.array([[-1,-1,1,1],[-1,1,-1,1]])
and_out = np.array([-1,1,1,-1])

error=1
count=0
"""
checking for mean square error less than 0.001
"""
while error > 0.001:
    epoch_1 = net.feedforward(and_inp)
    epoch1_derivative = net.tanh_derivative(epoch_1)
    gradiants= net.evaluate_gradients(and_out,epoch_1[len(epoch_1)-1],epoch1_derivative)
    delta_weights=net.evaluate_weights(gradiants,0.8389019,epoch_1,and_inp)
    delta_biases=net.evaluate_biases(gradiants,0.8389019)
    net.update_weights(delta_weights)
    net.update_biases(delta_biases)
    error=net.mean_squared_error(and_out,epoch_1[len(epoch_1)-1] )
    count+=1
    print("epcoh: %d error: %f"%(count,error))
    
    pass



