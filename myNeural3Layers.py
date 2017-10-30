#Importando libs
import numpy as np


#Sigmoidal
def nonlin(x, deriv = False):
	if(deriv == True):
		return x*(1-x)

	return 1/(1+np.exp(-x))


#Input data
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

Y = np.array([[0],[1],[1],[0]])

np.random.seed(1)

#Synapses
syn0 = 2*np.random.random((3,4)) -1
syn1 = 2*np.random.random((4,200)) -1
syn2 = 2*np.random.random((200,1)) -1

#Training steps

for j in xrange(0,60000):
	l0 = X
	l1 = nonlin(np.dot(l0,syn0))
	l2 = nonlin(np.dot(l1,syn1))
	l3 = nonlin(np.dot(l2,syn2))

	l3_error = Y - l3

	if((j %10000)==0 ):
		#print("ERROR:" + str(np.mean(np.abs(l2_error))))
		print("ERROR:" + str(np.mean(np.abs(l3_error))))

	
	#BACKPROPAGATION
	l3_delta = l3_error*nonlin(l3,deriv = True)
	
	l2_error = l3_delta.dot(syn2.T)
	l2_delta = l2_error*nonlin(l2,deriv = True)
	
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * nonlin(l1,deriv = True)
	
	#Atualizando pesos -> Gradiente de decaimento
	syn2 += l2.T.dot(l3_delta)
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

#Result
print("Output After training")
print (l3)


print("TEST BEFORE TRAINING")

#Testing Before training
Z = np.array([[1,1,0]])

l0x = Z
l1x = nonlin(np.dot(l0x,syn0))
l2x = nonlin(np.dot(l1x,syn1))
l3x = nonlin(np.dot(l2x,syn2))

print(l3x)







