import tensorflow as tf
from tensorflow import keras

# Common imports
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import numpy as np
np.random.seed(123)
from sklearn.model_selection import train_test_split
import seaborn as sns

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    
    return 1/(1+np.exp(-z))
    
def relu(x):

    return np.maximum(0.0, x)
"""
Arguments:
X -- input dataset of shape (input size, number of examples)
Y -- labels of shape (output size, number of examples)

Returns:
n_x -- the size of the input layer
n_h -- the size of the hidden layer
n_y -- the size of the output layer
"""

def layer_sizes(X, Y, hidden_units = 4):

  n_x = X.shape[1] # size of input layer
  n_h = hidden_units # we have 3 hidden units
  n_y = 1 # size of output layer
    
  return (n_x, n_h, n_y)
"""
Argument:
n_x -- size of the input layer
n_h -- size of the hidden layer
n_y -- size of the output layer

Returns:
params -- python dictionary containing the parameters:
                W1 -- weight matrix of shape (n_h, n_x)
                b1 -- bias vector of shape (n_h, 1)
                W2 -- weight matrix of shape (n_y, n_h)
                b2 -- bias vector of shape (n_y, 1)

"""
def initialize_parameters(n_x, n_h, n_y):

  np.random.seed(2) 
  W1 = np.random.randn(n_h,n_x) * 0.01
  b1 = np.zeros((n_h,1))
  W2 = np.random.randn(n_y,n_h) * 0.01
  b2 = np.zeros((n_y,1))
  parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
  return parameters
"""
Argument:
X -- input data of size (n_x, m)
parameters -- python dictionary containing the parameters (output of initialization function)

Returns:
A2 -- The sigmoid output of the second activation
cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
"""
def forward_propagation(X, parameters, activation="tanh"):

  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
  Z1 = np.dot(W1,X.T) + b1
  if activation=="relu":
    A1 = relu(Z1)
  elif activation=="tanh":
    A1 = np.tanh(Z1)
  Z2 = np.dot(W2,A1) + b2
  A2 = sigmoid(Z2)
  # Values needed in the backpropagation are stored in "cache". This will be given as an input to the backpropagation
  cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
    
  return A2, cache
"""
Computes the cross-entropy cost 

Arguments:
A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
Y -- "true" labels vector of shape (1, number of examples)
parameters -- python dictionary containing your parameters W1, b1, W2 and b2
[Note that the parameters argument is not used in this function, 
but the auto-grader currently expects this parameter.
Future version of this notebook will fix both the notebook 
and the auto-grader so that `parameters` is not needed.
For now, please include `parameters` in the function signature,
and also when invoking this function.]

Returns:
cost -- cross-entropy
"""
def compute_cost(A2, Y):

  m = Y.shape[0] # number of example
  # Compute the cross-entropy cost
  logprobs = np.multiply(Y ,np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
  cost = (-1/m) * np.sum(logprobs)  
  cost = float(np.squeeze(cost))  
  
  return cost
"""
Implement the backward propagation using the instructions above.

Arguments:
parameters -- python dictionary containing our parameters 
cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
X -- input data of shape (2, number of examples)
Y -- "true" labels vector of shape (1, number of examples)

Returns:
grads -- python dictionary containing your gradients with respect to different parameters
"""
def backward_propagation(parameters, cache, X, Y,activation="tanh"):

  m = X.shape[0]
  # First, retrieve W1 and W2 from the dictionary "parameters".
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
  # Retrieve also A1 and A2 from dictionary "cache".
  A1 = cache["A1"]
  A2 = cache["A2"]
  Z1 = cache["Z1"]
  Z2 = cache["Z2"]

  # Backward propagation: calculate dW1, db1, dW2, db2. 
  dZ2 = A2 - Y
  dW2 = (1/m) * np.dot(dZ2,A1.T)
  db2 = (1/m) *(np.sum(dZ2,axis=1,keepdims=True))
  dA1 = np.dot(W2.T,dZ2) 
  
  if activation=="relu":
    dZ1 = np.array(dA1, copy=True)
    dZ1[Z1 <= 0] = 0 # gradient of relu
  elif activation=="tanh":
    dZ1 = np.dot(W2.T,dZ2) * (1 - np.power(A1,2))

  dW1 = (1/m) *(np.dot(dZ1,X))
  db1 = (1/m) *(np.sum(dZ1, axis=1, keepdims=True))
  grads = {"dW1": dW1,"db1": db1,"dW2": dW2,"db2": db2}

  return grads
  
