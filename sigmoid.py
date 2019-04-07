import numpy as np

def sigmoid(z):
#SIGMOID Compute sigmoid function
#   g = SIGMOID(z) computes the sigmoid of z.

# You need to return the following variables correctly 
    g = 1/(1 + np.exp(-z))
    return g
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the sigmoid of each value of z (z can be a matrix,
#               vector or scalar).





# =============================================================


