import numpy as np
from sigmoid import sigmoid
from sklearn.preprocessing import OneHotEncoder

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):
#NNCOSTFUNCTION Implements the neural network cost function for a two layer
#neural network which performs classification
#   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
#   X, y, lambda) computes the cost and gradient of the neural network. The
#   parameters for the neural network are "unrolled" into the vector
#   nn_params and need to be converted back into the weight matrices. 
# 
#   The returned parameter grad should be a "unrolled" vector of the
#   partial derivatives of the neural network.
#

    # =======================Part 1===============================
    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = nn_params[0:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size, input_layer_size + 1)
    #the theta1 in the nn_params is input_layer_size+1 by hidden_layer_size because it contains the bias term    
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:].reshape(num_labels, hidden_layer_size + 1)
    
    # Setup some useful variables
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))         
    A1 = X
    Z2 = np.matmul(A1, Theta1.T)
    A2 = sigmoid(Z2)
    A2 = np.hstack((np.ones((m, 1)), A2)) #5000 by 26
    Z3 = np.matmul(A2, Theta2.T)
    A3 = sigmoid(Z3)   #since theta1 and 2 are both matrix, h is also matrix
    h = A3
    #h is 5000 by 10
    # You need to return the following variables correctly 
    #y cannot be a vector
    onehotencoder = OneHotEncoder(categorical_features = [0]) #0 specifies which column(country) to categorize
    # the y is from the dataset prepared for matlab, which index starts at 1. so for use here, need to adjust to y-1 
    y_matrix = onehotencoder.fit_transform(y-1).toarray()
    J =  np.sum(-y_matrix * np.log(h) - (1 - y_matrix) * np.log(1 - h)) / m
    #remove bias from theta
    Theta1_nobias = Theta1[:,1:]
    Theta2_nobias = Theta2[:,1:]
    J_reg = (np.sum(Theta1_nobias*Theta1_nobias) + np.sum(Theta2_nobias*Theta2_nobias))*lmbda*0.5/m
    J = J + J_reg
    return J
# =======================Part 2===============================
def nnGradFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmbda):

    Theta1 = nn_params[0:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size, input_layer_size + 1)
    #the theta1 in the nn_params is input_layer_size+1 by hidden_layer_size because it contains the bias term    
    Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:].reshape(num_labels, hidden_layer_size + 1)
    
        # Setup some useful variables
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))         
    A1 = X
    Z2 = np.matmul(A1, Theta1.T)
    A2 = sigmoid(Z2)
    A2 = np.hstack((np.ones((m, 1)), A2)) #5000 by 26
    Z3 = np.matmul(A2, Theta2.T)
    A3 = sigmoid(Z3)   #since theta1 and 2 are both matrix, h is also matrix
    
    onehotencoder = OneHotEncoder(categorical_features = [0]) #0 specifies which column(country) to categorize
    # the y is from the dataset prepared for matlab, which index starts at 1. so for use here, need to adjust to y-1 
    y_matrix = onehotencoder.fit_transform(y-1).toarray()
    
    DELTA1 = np.zeros(Theta1.shape)   #25 by 401
    DELTA2 = np.zeros(Theta2.shape)   #10 by 26
    delta3 = A3 - y_matrix #5000 by 10
    #A2 = A2(:,2:end)   #5000 by 25
    g_prime = A2 * (1 - A2)    #5000 by 26
    delta2 = delta3 @ Theta2 * g_prime  #5000 by 26
    delta2 = delta2[:,1:]#5000 by 25
    DELTA2 = DELTA2 + delta3.T @ A2
    #A1 = A1(:,2:end)
    DELTA1 = DELTA1 + delta2.T @ A1   #25 by 401
    Theta1_grad = DELTA1 / m
    Theta2_grad = DELTA2 / m
    #regularization
    Theta1_temp = Theta1.copy()
    Theta2_temp = Theta2.copy()
    Theta1_temp[:,0] = 0
    Theta2_temp[:,0] = 0
    Theta1_grad = Theta1_grad + Theta1_temp * lmbda / m
    Theta2_grad = Theta2_grad + Theta2_temp * lmbda / m
    #unroll theta_grad
    grad = np.hstack((Theta1_grad.ravel(), Theta2_grad.ravel()))
    return grad[:,None]
