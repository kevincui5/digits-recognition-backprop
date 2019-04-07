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
## ====================== YOUR CODE HERE ======================
## Instructions: You should complete the code by working through the
##               following parts.
##
## Part 1: Feedforward the neural network and return the cost in the
##         variable J. After implementing Part 1, you can verify that your
##         cost function computation is correct by verifying the cost
##         computed in ex4.m
##
## Part 2: Implement the backpropagation algorithm to compute the gradients
##         Theta1_grad and Theta2_grad. You should return the partial derivatives of
##         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
##         Theta2_grad, respectively. After implementing Part 2, you can check
##         that your implementation is correct by running checkNNGradients
##
##         Note: The vector y passed into the function is a vector of labels
##               containing values from 1..K. You need to map this vector into a 
##               binary vector of 1's and 0's to be used with the neural network
##               cost function.
##
##         Hint: We recommend implementing backpropagation using a for-loop
##               over the training examples if you are implementing it for the 
##               first time.
##
## Part 3: Implement regularization with the cost function and gradients.
##
##         Hint: You can implement this around the code for
##               backpropagation. That is, you can compute the gradients for
##               the regularization separately and then add them to Theta1_grad
##               and Theta2_grad from Part 2.
##
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
## -------------------------------------------------------------
#
## =========================================================================
#
## Unroll gradients
#grad = [Theta1_grad(:)  Theta2_grad(:)]


# below from https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-5/
from sigmoidGradient import sigmoidGradient
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    
    return a1, z2, a2, z3, h

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    ##### this section is identical to the cost function logic we already saw #####
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    
    # initializations
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
    
    # compute the cost
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    
    J = J / m
    
    # add the cost regularization term
    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))

    ##### end of cost function logic, below is the new part #####
    
    # perform backpropagation
    for t in range(m):
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)
        
        d3t = ht - yt  # (1, 10)
        
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoidGradient(z2t))  # (1, 26)
        
        delta1 = delta1 + (d2t[:,1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t
        
    delta1 = delta1 / m
    delta2 = delta2 / m
    
    # add the gradient regularization term
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * learning_rate) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * learning_rate) / m
    
    # unravel the gradient matrices into a single array
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    
    return J, grad