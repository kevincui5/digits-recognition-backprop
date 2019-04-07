from sigmoid import sigmoid
import numpy as np

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
#   trained weights of a neural network (Theta1, Theta2)

    # Useful values
    m = X.shape[0]    
    # You need to return the following variables correctly 
    #p = np.zeros(size(X, 1), 1)
    
    h1 = sigmoid(np.hstack((np.ones((m, 1)), X)) @ Theta1.T)
    h2 = sigmoid(np.hstack((np.ones((m, 1)), h1)) @ Theta2.T)
    p = np.argmax(h2, axis = 1)
    return p

