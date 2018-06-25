#Created by Matthew Li
#My version of a concise 3 layer Neural Network attempted by memory
#06/25/18

import numpy as np

def nonlin(x, deriv=False):
	if deriv==True:
		return x*(1-x)
	return 1/(1+np.exp(-x))

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]]) #dimension is 4x3
y = np.array([[0,1,1,0]]).T #dimension is 4*1

np.random.seed(1)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
			   
for i in range(60000): #<-number of iterations
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	
	#prediction error, dimension 4*1
	l2_error = y - l2

	#special sauce (error based derivative/confidence based error of prediction), dimension 4*1
	l2_delta = l2_error * nonlin(l2, True)
	
	#error of l1 based on effect on l2
	l1_error = np.dot(l2_delta, syn1.T)

	#special sauce (error based derivative/confidence based error of prediction), dimension 4*3
	l1_delta = l1_error * nonlin(l1,True)

	syn0 += np.dot(l0.T, l1_delta) #matrix multiplication (3x4)x(3x4) = (3x4)
	syn1 += np.dot(l1.T, l2_delta) #matrix multiplication ()

print l2

