# Neural Network Learning


import scipy.io
import numpy as np
from nnCostFunction import nnCostFunction, nnGradFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients
from predict import predict


input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)


trainingset = scipy.io.loadmat('ex4data1.mat');
X = trainingset.get('X')
y = trainingset.get('y')
# not implemented

print('\nLoading Saved Neural Network Parameters ...\n')

# Load the weights into variables Theta1 and Theta2
weights = scipy.io.loadmat('ex4weights.mat');
Theta1 = weights.get('Theta1')
Theta2 = weights.get('Theta2')

# Unroll parameters 
nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#

print('\nFeedforward Using Neural Network ...\n')

# Weight regularization parameter (we set this to 0 here).
lmbda = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lmbda)

print(['Cost at parameters (loaded from ex4weights): %f '\
         '\n(this value should be about 0.287629)\n'], J)


## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
lmbda = 1

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lmbda)
grad = nnGradFunction(nn_params, input_layer_size, hidden_layer_size, \
                   num_labels, X, y, lmbda)

print(['Cost at parameters (loaded from ex4weights): %f '\
         '\n(this value should be about 0.383770)\n'], J)


## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.


print('\nEvaluating sigmoid gradient...\n')

g = sigmoidGradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ')
print('%f ', g)
print('\n\n')

### ================ Part 6: Initializing Pameters ================
##  In this part of the exercise, you will be starting to implment a two
##  layer neural network that classifies digits. You will start by
##  implementing a function to initialize the weights of the neural network
##  (randInitializeWeights.m)

print('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack((initial_Theta1.ravel(), initial_Theta2.ravel()))


## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#
print('\nChecking Backpropagation... \n')

#  Check gradients by running checkNNGradients
checkNNGradients()


### =============== Part 8: Implement Regularization ===============
##  Once your backpropagation implementation is correct, you should now
##  continue to implement the regularization with the cost and gradient.
##

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
lmbda = 3
checkNNGradients(lmbda)

# Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, \
                          hidden_layer_size, num_labels, X, y, lmbda)

print(['\n\nCost at (fixed) debugging parameters (w/ lmbda = %f): %f ' \
         '\n(for lmbda = 3, this value should be about 0.576051)\n\n'], lmbda, debug_J)


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural 
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print('\nTraining Neural Network... \n')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.

#  You should also try different values of lmbda
lmbda = 1
args = (input_layer_size, hidden_layer_size, num_labels, X, y, lmbda)
from scipy import optimize
#nn_params = optimize.fmin_cg(nnCostFunction, nn_params, fprime=nnGradFunction, args=args, maxiter=50)
result = optimize.minimize(fun=nnCostFunction, x0=nn_params, args=(input_layer_size, hidden_layer_size, num_labels, X, y, lmbda), 
                method='TNC', jac=nnGradFunction, options={'maxiter': 150})

nn_params = result.x
# Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[0:(input_layer_size+1)*hidden_layer_size].reshape(hidden_layer_size, input_layer_size + 1)
#the theta1 in the nn_params is input_layer_size+1 by hidden_layer_size because it contains the bias term    
Theta2 = nn_params[(input_layer_size+1)*hidden_layer_size:].reshape(num_labels, hidden_layer_size + 1)


### ================= Part 9: Visualize Weights =================
##  You can now "visualize" what the neural network is learning by 
##  displaying the hidden units to see what features they are capturing in 
##  the data.
#
## not implemented
#
#
# predict the labels and compute the training set accuracy.

pred = predict(Theta1, Theta2, X)
correct = np.array([1 if (a + 1 == b) else 0 for (a, b) in zip(pred, y)])
print('\nTraining Set Accuracy: %f\n', correct.mean(axis = 0) * 100)


