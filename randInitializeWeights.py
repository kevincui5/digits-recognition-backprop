import numpy as np

def randInitializeWeights(L_in, L_out):
#RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
#incoming connections and L_out outgoing connections
#   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
#   of a layer with L_in incoming connections and L_out outgoing 
#   connections. 
#
#   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
#   the first column of W handles the "bias" terms
#

# You need to return the following variables correctly 
#W = zeros(L_out, 1 + L_in);
# a good choice of e_init is by sqrt(6)/sqrt(L_out + L_in)
    epsilon_init = 0.12
    W = np.random.random_sample((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    return W
# ====================== YOUR CODE HERE ======================
# Instructions: Initialize W randomly so that we break the symmetry while
#               training the neural network.
#
# Note: The first column of W corresponds to the parameters for the bias unit
#









# =========================================================================

