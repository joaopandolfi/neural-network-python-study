import numpy as np
import random
#import math

import logging
logging.getLogger().setLevel(logging.DEBUG)

df_path = "../data/trunkline.df.csv"
label_col = 0
id_col = 0
clumns_to_remove = [id_col]
batch_size = 1
METHODS = ["MINHA","MXNET","RMSE"] 

METHOD = METHODS[0]

#costas
for i in range(240, 300):
	clumns_to_remove.append(i)

#barriga
for i in range(61, 120):
	clumns_to_remove.append(i)



# ======== FUNCOES DE MANIPULACAO DE ARQUIVO ===========

#le arquivo completo => Retorna lista de linhas
def read_arq(arquivo):
	conteudo =""
	lista= []
	arq = open(arquivo , 'rt')
	conteudo = arq.readline()
	while conteudo != '':
		if(conteudo[-1] =='\n'):
			conteudo = conteudo[:-1].replace(",",".")
		lista.append(conteudo)
		conteudo = arq.readline()
	arq.close()
	return lista


#Transforma dados lidos no csv em matriz
def arq_to_mat(lista,separator):
	i = 1
	tam = len(lista)
	result = []
	while(i < tam):
		aux = lista[i].split(separator) #linha -> [a,b,c,d]
		aux2 = []
		j =1
		while(j < len(aux)):
			#print(aux[j])
			try:
				aux2.append(float(aux[j].replace("\"","")))
			except:
				pass
			j+=1

		result.append(aux2)

		i +=1
	return result


# Limpa colunas da lista
def clean_list(lista,to_remove):
	i = 1
	tam = len(lista)
	while(i < tam):
		aux = lista[i]+[]
		for rem in to_remove:
			lista[i].remove(aux[rem])
		i = i+1

	lista.remove(lista[0])
	return lista

# Recupera a coluna
def get_colum(lista, coluna):
	i = 1
	tam = len(lista)
	newList = []
	while(i < tam):
		#print(lista[i])
		newList.append([lista[i][coluna]])
		i = i+1

	return newList


# Retorna dois agrupamentos aleatorios
# result -> tamanho definido
# aux -> resto
def get_random_data(lista1,lista2,percentage):
	result = []
	result2 = []
	aux = lista1 + []
	aux2 = lista2 + []
	quant = int(len(lista1)*percentage)
	i = 0
	while(i< quant):
		res = random.randrange(0,len(aux))
		result.append(aux[res])
		result2.append(aux2[res])
		aux.remove(aux[res])
		aux2.remove(aux2[res])
		i+=1

	return {"l1":[result,aux],"l2":[result2,aux2]}

# Retorna subgrupos do tamanho definido
# Ou seja o tamanho do agrupamento
def sub_groups(lista1,lista2, tam):
	result1 = []
	result2 = []

	quant = int(len(lista1)/tam)


	i = 0
	j = 0
	if(tam ==1 ):
		result1.append(lista1)
		result2.append(lista2)

	elif(tam == len(lista1)):
		quant = tam
		while(j < quant):
			result1.append([lista1[j]])
			result2.append([lista2[j]])
			j+=1

	else:
		while(j < quant):
			result1.append(lista1[i:i+quant])
			result2.append(lista2[i:i+quant])
			i += tam
			j+=1

	return result1,result2


# ============ EXECUCAO DO PROBLEMA ==================

print("LENDO AQUIVO")
lines = read_arq(df_path)
lines = arq_to_mat(lines,";")
print("PEGANDO COLUNA")
#print(lines[0])

label_list = get_colum(lines,label_col)
print("LIMPANDO LISTA")
print(len(lines[0]))

cleaned_list = clean_list(lines,clumns_to_remove)

'''
# DEBUG
print("PESOS")
print(len(label_list))
print("LISTA LIMPA")
print(len(cleaned_list[0]))

#print( cleaned_list)
'''
#cleaned_list = [[1,2,3],[4,5,6],[7,8,9]]
#label_list = [1,2,3]
#batch_size = 1
'''
print(cleaned_list)

print ("=====================")
print(label_list)
'''


subgroup_len = 500

result_randomization = get_random_data(cleaned_list,label_list,0.7)
#result_randomization = {"l1":[cleaned_list[:-1],cleaned_list[-1]],"l2":[label_list[:-1],label_list[-1]]}
#print(result_randomization["l1"][0][0])

if(METHOD == METHODS[0]):
	tam1 = 1 #len(result_randomization["l1"][0])
	tam2 = 1 #len(result_randomization["l2"][0])
	result_randomization["l1"][0],result_randomization["l2"][0] = sub_groups(result_randomization["l1"][0],result_randomization["l2"][0],tam1)
	result_randomization["l1"][1],result_randomization["l2"][1] = sub_groups(result_randomization["l1"][1],result_randomization["l2"][1],tam2)

# print(len(result_randomization["l1"][0][0][0]))
# print(len(result_randomization["l1"][0][0]))
# print(len(result_randomization["l1"][0]))
# print(result_randomization["l1"][0][0])
# input()

#print(result_randomization["l1"][0])
#print("TAMANHO DOS AGRUPAMENTOS")
#print(len(result_randomization["l1"][0]))
#print(len(result_randomization["l1"][1]))

#Defining Data Train
train_data = np.array(result_randomization["l1"][0])#np.array(cleaned_list)
train_label = np.array(result_randomization["l2"][0])#np.array(label_list)


#print(train_label)

#print(train_data)
#print(train_label)

#Eval Data
eval_data = np.array(result_randomization["l1"][1])
eval_label = np.array(result_randomization["l2"][1])

#print(train_data)

#print(train_data[0])
#print(train_data[1])



#res1, res2 = sub_groups(result_randomization["l1"][0],result_randomization["l2"][0],50)

# ========== MINHA LIB
if(METHOD == METHODS[0]):
	#Importando libs
	import numpy as np
	import pylab


	#Sigmoidal
	def nonlin(x, deriv = False):
		if(deriv == True):
			#return ( (1/(np.cosh(x/2)**2))/4) # sech -> 1/tanh
			return x*(1-x)
			#return 1/(2*np.cosh(x)+2)
			#return 1/(2*(np.cosh(x)+1))
			#return (np.cosh(x/2)**2)/((np.cosh(x)+1)**2)
		
		#return  1/(1+math.e**-x) #Overflow
		#return 1/(1+np.exp(-x))  #Overflow
		return (0.5+(np.tanh(x/2))/2)
		#== N FUNCIONA ==
		#return np.log((0.5+(np.tanh(x/2))/2))
		#return np.log(1+np.exp(-x))

	log_err = []
	#Input data
	X = train_data#np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

	Y = train_label#np.array([[0],[1],[1],[0]])

	np.random.seed(3)

	data_lines = len(train_data[0])
	data_columns = len(train_data[0][0])
	label_lines = data_lines
	label_columns = 1
	data_train_len = data_lines

	#Synapses
	syn0 = 2*np.random.random((data_columns,data_train_len)) -1
	syn1 = 3*np.random.random((data_train_len,128)) -2
	syn2 = 4*np.random.random((128,64)) -3
	syn3 = 5*np.random.random((64,16)) -4

	syn4 = 1*np.random.random((16,1)) -5

	#Training steps
	i = 0
	print("GENERATIONS: "+str(len(train_data)))
	while(i < len(train_data)):	
		print("Generation:" + str(i))

		for j in range(0,100000):
			#for k in range(0,len(X[i])):
			#activation
			l0 = X[i]
			l1 = nonlin(np.dot(l0,syn0))
			l2 = nonlin(np.dot(l1,syn1))
			l3 = nonlin(np.dot(l2,syn2))
			l4 = nonlin(np.dot(l3,syn3))
			l5 = nonlin(np.dot(l4,syn4))

			ln = l5
			#l2_error = Y - l2


			ln_error = Y[i] - ln

			if((j %1000)==0 ):
				aux_err = np.mean(np.abs(ln_error))
				print("["+ str(j) +"] ERROR:" + str(aux_err))
				log_err.append(aux_err)

			#print(ln_error)
			#print(l5)
		
			#BACKPROPAGATION
			l5_delta = ln_error*nonlin(l5,deriv = True)

			l4_error = l5_delta.dot(syn4.T)
			l4_delta = l4_error*nonlin(l4,deriv = True)

			l3_error = l4_delta.dot(syn3.T)
			l3_delta = l3_error*nonlin(l3,deriv = True)
		
			l2_error = l3_delta.dot(syn2.T)
			l2_delta = l2_error*nonlin(l2,deriv = True)
		
			l1_error = l2_delta.dot(syn1.T)
			l1_delta = l1_error*nonlin(l1,deriv = True)
		
	
			#Atualizando pesos -> Gradiente de decaimento
			syn4 += l4.T.dot(l5_delta)
			syn3 += l3.T.dot(l4_delta)
			syn2 += l2.T.dot(l3_delta)
			syn1 += l1.T.dot(l2_delta)
			syn0 += l0.T.dot(l1_delta)

		i+=1


	#Result
	pylab.plot(log_err) 
	pylab.show()


	print("Output After training")
	if(len(eval_data) >0):
		for ed in range(0, len(eval_data)):
			l0 = eval_data[ed]
			l1 = nonlin(np.dot(l0,syn0))
			l2 = nonlin(np.dot(l1,syn1))
			l3 = nonlin(np.dot(l2,syn2))
			l4 = nonlin(np.dot(l3,syn3))
			l5 = nonlin(np.dot(l4,syn4))
			lx = l5

			lx_error = eval_label[0][ed] - lx[ed]		
			print("ERROR:" + str(np.mean(np.abs(lx_error))))
			print("PREDICTED: "+str(lx[ed]))
			print("REAL: "+ str(eval_label[0][ed]))

	#print (l2)
	#print (l3)


#print("TEST BEFORE TRAINING")

#Testing Before training
#Z = np.array([[1,1,0]])

#l0x = Z
#l1x = nonlin(np.dot(l0x,syn0))
#l2x = nonlin(np.dot(l1x,syn1))

#print(l2x)





# ================ MXNET
if(METHOD == METHODS[1]):
	import mxnet as mx
	train_iter = mx.io.NDArrayIter(train_data,train_label, batch_size, shuffle=True,label_name='lin_reg_label')
	eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)


	## PURO TESTE

	X = mx.sym.Variable('data')
	Y = mx.symbol.Variable('lin_reg_label')
	
	fully_connected_layer  = mx.sym.FullyConnected(data=X, name='fc1', num_hidden = 1)
	
	#DEBUG
	act1 = mx.sym.Activation(data = fully_connected_layer,name="ac1",act_type =	"tanh")

	fully_connected_layer_2 = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 128)
	act2 = mx.sym.Activation(data =	fully_connected_layer_2,name="ac2",act_type= "sigmoid")

	fully_connected_layer_3 = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden = 64)
	act3 = mx.sym.Activation(data =	fully_connected_layer_3,name="ac3",act_type= "tanh")

	fully_connected_layer_4 = mx.sym.FullyConnected(data=act3, name='fc4', num_hidden = 16)
	act4 = mx.sym.Activation(data = fully_connected_layer_4,name="ac4",act_type= "sigmoid")

	fully_connected_layer_x = mx.sym.FullyConnected(data=act4, name='fc5', num_hidden = 8)
	actx = mx.sym.Activation(data = fully_connected_layer_x,name="ac4",act_type= "relu")

	fully_connected_layer_x = mx.sym.FullyConnected(data=actx, name='fc6', num_hidden = 1)
	#actx = mx.sym.Activation(data = fully_connected_layer_x,name="ac4",act_type= "relu")


	fully_connected_layer_r = fully_connected_layer_x

	#fully_connected_layer_r = fully_connected_layer_4
	#fully_connected_layer_r = act4
	#fully_connected_layer_r = act3
	#fully_connected_layer_r = act2


	#fully_connected_layer_r = fully_connected_layer

	lro = mx.sym.LinearRegressionOutput(data=fully_connected_layer_r, label=Y, name="lro")
	
	model = mx.mod.Module(
	    symbol = lro ,
	    data_names=['data'],
	    label_names = ['lin_reg_label']# network structure
	)

	mx.viz.plot_network(symbol=lro)

	#TREINANDO SAPORRA

	model.fit(train_iter, eval_iter,
	            optimizer_params={'learning_rate':0.005, 'momentum': 0.9},
	            num_epoch=50,
	            #eval_metric='mse',
	            eval_metric='rmse',
	            batch_end_callback = mx.callback.Speedometer(1, 50))
	            #batch_end_callback = mx.callback.Speedometer(batch_size, columns_size))
	            #batch_end_callback = mx.callback.Speedometer(1, 3))

	model.predict(eval_iter).asnumpy()

	metric = mx.metric.RMSE()
	b= model.score(eval_iter, metric)

	print(b)


#==================== RMSE =================

# Define the loss function 
def squared_loss(y, y_hat):
	return np.dot((y - y_hat),(y - y_hat))

 # Output Layer
def binary_cross_entropy(y, y_hat): 
	return np.sum(-((y * np.log(y_hat)) + ((1-y) * np.log(1 - y_hat))))


# Wraper around the Neural Network 
def neural_network(x, theta):
	w1, b1, w2, b2 = theta
	return np.tanh(np.dot((np.tanh(np.dot(x,w1) + b1)), w2) + b2)

# Wrapper around the objective function to be optimised 
def objective(theta, idx):
	return squared_loss(D[1][idx], neural_network(D[0][idx], theta))

# Update
def update_theta(theta, delta, alpha): 
	w1, b1, w2, b2 = theta
	w1_delta, b1_delta, w2_delta, b2_delta = delta 
	w1_new = w1 - alpha * w1_delta 
	b1_new = b1 - alpha * b1_delta 
	w2_new = w2 - alpha * w2_delta 
	b2_new = b2 - alpha * b2_delta 
	new_theta = (w1_new,b1_new,w2_new,b2_new) 
	return new_theta


if(METHOD == METHODS[2]):
	import autograd.numpy as np
	import autograd.numpy.random as npr
	from autograd import grad
	import sklearn.metrics
	import pylab
	examples = len(train_data[0])
	features = label_col
	#D = (npr.randn(examples, features), npr.randn(examples))
	D =  (train_data, train_label)
	# Specify the network 
	layer1_units = 128
	layer2_units = 1
	w1 = npr.rand(features, layer1_units) 
	b1 = npr.rand(layer1_units)
	w2 = npr.rand(layer1_units, layer2_units) 
	b2 = 0.0
	theta = (w1, b1, w2, b2)


	# Compute Gradient 
	grad_objective = grad(objective)
	# Train the Neural Network 
	epochs = 50
	print ("MSE before training:", sklearn.metrics.mean_absolute_error(D[1],neural_network(D[0], theta)) )
	#print ("RMSE before training:", sklearn.metrics.mean_squared_error(D[1],neural_network(D[0], theta)) )
	rmse = []
	for i in range(0, epochs): 
		for j in range(0, examples): 
			delta = grad_objective(theta, j) 
			theta = update_theta(theta,delta, 0.01)

			#rmse.append(sklearn.metrics.mean_squared_error(D[1],neural_network(D[0], theta))) 
			rmse.append(sklearn.metrics.mean_absolute_error(D[1],neural_network(D[0], theta))) 
	
	print ("MSE after training:", sklearn.metrics.mean_absolute_error(D[1],neural_network(D[0], theta)))
	print ("MSE PREDICTED:", sklearn.metrics.mean_absolute_error(eval_label,neural_network(eval_data, theta)))
	#print ("RMSE after training:", sklearn.metrics.mean_squared_error(D[1],neural_network(D[0], theta)))

	print("TEST: ",sklearn.metrics.mean_absolute_error(eval_label[0],neural_network(eval_data[0], theta)))
	print("VALOR (P,R): ", neural_network(eval_data[0], theta), eval_label[0])


	pylab.plot(rmse) 
	pylab.show()
