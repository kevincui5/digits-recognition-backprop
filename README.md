
The problem come from Andrew Ng's machine learning course projects from Coursera, and 
I'd like to implement them in python instead of matlab/octave

In the digits-recognition repository I implemented a neural network to recognize 
hand-written digits with pre-trained parameters. In this exercise I will implement
 the back-propagation to train the parameters and make prediction on the digits.
 
The dataset is in the file ex4data1.mat.  It is a subset of MNIST hand-written 
digits, containing 5000 training sets. Each digit image is a 20x20 gray scale 
image but in ex3data1.mat image is converted to float64, each row represent a image,
and there are 400 columns, each as a pixel, a feature.

the file ex4weights.mat is still given for the purpose of implementing cost
function and gradient in the begining of the code.
The training set accuracy turned out to be 99.5%

To execute, just run ex4.py

For the second part, we are to implement a simple nural network to recognize the digits.
We don't need to implement the back-propagation, saving for another exercise.  
We are given the pre-trained parameters and they are saved in ex3weights.mat.  
I implemented the forward-propagation in the predict.py. so no optimization objection
algorithm library needed for this exercise.
run ex3_nn.py to try

DO NOT USE THIS SOURCE CODE FOR THE EXERCISES/PROJECTS IN COURSERA MACHINE
 LEARNING COURSE.